"""Educational SKEMPI baseline. No structure model, novelty or clinical claims."""
from pathlib import Path
from urllib.request import Request, urlopen
import argparse, hashlib, json, re, platform
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge

URL='https://life.bsc.es/pid/skempi2/database/download/skempi_v2.csv'
R=8.314/4184 # kcal / mol / kelvin, following upstream FAQ
AA='ACDEFGHIKLMNPQRSTVWY'
MUT=re.compile(r'^(['+AA+r'])([A-Za-z0-9])(-?\d+[A-Za-z]?)(['+AA+r'])$')
FEATURES=['wt_aa','mut_aa','location']
MANIFEST_VERSION=1
SPLIT_ALGORITHM='greedy_group_balance_size_desc_group_asc_fold_index_v1'

def exact_float(value):
    try:
        x=float(str(value).strip())
        return x if np.isfinite(x) else np.nan
    except (ValueError, TypeError): return np.nan

def ddg(kmut,kwt,temp):
    if min(kmut,kwt,temp)<=0: raise ValueError('Positive inputs required')
    return R*temp*(np.log(kmut)-np.log(kwt))

class UnionFind:
    def __init__(self):self.parent={}
    def find(self,x):
        self.parent.setdefault(x,x)
        if self.parent[x]!=x:self.parent[x]=self.find(self.parent[x])
        return self.parent[x]
    def union(self,a,b):self.parent[self.find(b)]=self.find(a)

def holdout_groups(raw):
    """Connected components from full raw metadata, including excluded records.
    Shared PDB structures and explicit upstream holdout tokens cannot cross folds.
    Not a sequence-identity clustering guarantee.
    """
    u=UnionFind()
    for _,row in raw.iterrows():
        c=str(row['#Pdb']).strip();u.union(c,'PDB:'+c.split('_')[0].upper())
        for col in ['Hold_out_type','Hold_out_proteins']:
            if pd.isna(row[col]):continue
            for token in str(row[col]).split(','):
                token=token.strip()
                if token:u.union(c,token)
    complexes=sorted({str(c) for c in raw['#Pdb'].unique()})
    components={}
    for complex_name in complexes:
        components.setdefault(u.find(complex_name),[]).append(complex_name)
    canonical={root:min(members) for root,members in components.items()}
    return {complex_name:canonical[u.find(complex_name)] for complex_name in complexes}

def canonical_sha256(value):
    encoded=json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()

def example_id(complex_name,mutation):
    return hashlib.sha256(f'{complex_name}\0{mutation}'.encode()).hexdigest()

def example_set_fingerprint(clean):
    columns=['example_id','complex','mutation','pdb_group','related_group']
    records=clean[columns].sort_values('example_id').to_dict(orient='records')
    return canonical_sha256(records)

def deterministic_group_folds(clean,group_column,nfold=5):
    """Allocate whole groups without targets, with explicit stable tie-breaking."""
    counts={str(group):int(size) for group,size in clean.groupby(group_column,sort=False).size().items()}
    if len(counts)<2:raise ValueError('Not enough independent groups')
    nfold=min(nfold,len(counts));fold_rows=[0]*nfold;group_to_fold={}
    for group,size in sorted(counts.items(),key=lambda item:(-item[1],item[0])):
        fold=min(range(nfold),key=lambda candidate:(fold_rows[candidate],candidate))
        group_to_fold[group]=fold;fold_rows[fold]+=size
    return clean[group_column].astype(str).map(group_to_fold).astype(int).to_numpy()

def build_split_manifest(clean,dataset_sha256):
    if clean.example_id.duplicated().any():raise ValueError('Duplicate example IDs in processed data')
    manifest={'manifest_version':MANIFEST_VERSION,'algorithm':SPLIT_ALGORITHM,
        'dataset_sha256':dataset_sha256,'example_set_sha256':example_set_fingerprint(clean),
        'n_examples':len(clean),'groupings':{}}
    for group_column in ['pdb_group','related_group']:
        folds=deterministic_group_folds(clean,group_column)
        assignments=[{'example_id':eid,'group':str(group),'fold':int(fold)}
            for eid,group,fold in zip(clean.example_id,clean[group_column],folds)]
        assignments.sort(key=lambda row:row['example_id'])
        manifest['groupings'][group_column]={'n_folds':int(max(folds)+1),'assignments':assignments}
    manifest['split_fingerprint']=canonical_sha256(manifest)
    return manifest

def validate_split_manifest(clean,manifest,dataset_sha256):
    supplied_fingerprint=manifest.get('split_fingerprint')
    unsigned={key:value for key,value in manifest.items() if key!='split_fingerprint'}
    if supplied_fingerprint!=canonical_sha256(unsigned):raise ValueError('Split manifest fingerprint mismatch')
    if manifest.get('manifest_version')!=MANIFEST_VERSION:raise ValueError('Unsupported split manifest version')
    if manifest.get('algorithm')!=SPLIT_ALGORITHM:raise ValueError('Unexpected split allocation algorithm')
    if manifest.get('dataset_sha256')!=dataset_sha256:raise ValueError('Raw dataset fingerprint mismatch')
    if clean.example_id.duplicated().any():raise ValueError('Duplicate example IDs in processed data')
    if manifest.get('n_examples')!=len(clean):raise ValueError('Split manifest row count mismatch')
    if manifest.get('example_set_sha256')!=example_set_fingerprint(clean):raise ValueError('Processed example fingerprint mismatch')
    expected=set(clean.example_id);validated={}
    for group_column in ['pdb_group','related_group']:
        spec=manifest.get('groupings',{}).get(group_column)
        if not spec:raise ValueError(f'Missing grouping: {group_column}')
        assignments=spec.get('assignments',[]);ids=[row.get('example_id') for row in assignments]
        if len(ids)!=len(set(ids)):raise ValueError(f'Duplicate IDs in {group_column} assignments')
        if set(ids)!=expected:raise ValueError(f'Incomplete or unexpected IDs in {group_column} assignments')
        by_id={row['example_id']:row for row in assignments};folds=[]
        for eid,group in zip(clean.example_id,clean[group_column].astype(str)):
            row=by_id[eid]
            if str(row.get('group'))!=group:raise ValueError(f'Group mismatch for {eid}')
            fold=row.get('fold')
            if not isinstance(fold,int) or not 0<=fold<spec['n_folds']:raise ValueError(f'Invalid fold for {eid}')
            folds.append(fold)
        folds=np.asarray(folds,dtype=int);groups=clean[group_column].astype(str)
        for fold in range(spec['n_folds']):
            test_groups=set(groups[folds==fold]);train_groups=set(groups[folds!=fold])
            if test_groups & train_groups:raise ValueError(f'Train/test group overlap in {group_column} fold {fold}')
        validated[group_column]=folds
    return validated

def prepare(raw):
    work=raw.copy(); audit={'raw_rows':len(raw),'exclusions_sequential':{}}
    def keep(mask,reason):
        nonlocal work
        audit['exclusions_sequential'][reason]=int((~mask).sum())
        work=work.loc[mask].copy()
    parsed=work['Mutation(s)_cleaned'].astype(str).str.extract(MUT)
    work=work.assign(wt_aa=parsed[0],mut_aa=parsed[3])
    keep(work.wt_aa.notna() & work.mut_aa.notna() & (work.wt_aa!=work.mut_aa),'not_single_standard_amino_acid_substitution')
    work=work.assign(kmut=work['Affinity_mut (M)'].map(exact_float),kwt=work['Affinity_wt (M)'].map(exact_float))
    keep(work.kmut.notna() & work.kwt.notna() & (work.kmut>0) & (work.kwt>0),'missing_censored_or_nonpositive_affinity')
    work=work.assign(temperature_k=work.Temperature.map(exact_float))
    keep(work.temperature_k.notna() & work.temperature_k.between(250,350),'missing_assumed_or_invalid_temperature')
    work=work.assign(ddg_kcal_mol=ddg_array(work.kmut,work.kwt,work.temperature_k))
    keep(np.isfinite(work.ddg_kcal_mol),'nonfinite_target')
    work=work.assign(location=work['iMutation_Location(s)'].fillna('UNKNOWN').astype(str),
        complex=work['#Pdb'].astype(str),mutation=work['Mutation(s)_cleaned'].astype(str))
    audit['retained_measurements']=len(work)
    # Consolidate repeated complex/substitution measurements with a documented median.
    clean=work.groupby(['complex','mutation'],as_index=False).agg(
        wt_aa=('wt_aa','first'),mut_aa=('mut_aa','first'),location=('location','first'),
        ddg_kcal_mol=('ddg_kcal_mol','median'),replicates=('ddg_kcal_mol','size'),
        target_min=('ddg_kcal_mol','min'),target_max=('ddg_kcal_mol','max'))
    groups=holdout_groups(raw)
    clean=clean.assign(pdb_group=clean.complex.str.split('_').str[0].str.upper(),
        related_group=clean.complex.map(groups))
    clean=clean.assign(example_id=[example_id(c,m) for c,m in zip(clean.complex,clean.mutation)])
    if clean.example_id.duplicated().any():raise ValueError('Duplicate stable example IDs')
    audit.update(unique_complex_mutations=len(clean),repeated_measurements_collapsed=len(work)-len(clean),
        pdb_groups=int(clean.pdb_group.nunique()),related_groups=int(clean.related_group.nunique()),
        largest_related_group_rows=int(clean.related_group.value_counts().max()))
    return clean,audit

def ddg_array(kmut,kwt,temp):return R*temp*(np.log(kmut)-np.log(kwt))

def metrics(y,p):
    e=np.asarray(y)-np.asarray(p)
    return {'mae_kcal_mol':float(np.abs(e).mean()),'rmse_kcal_mol':float(np.sqrt(np.mean(e**2)))}

def evaluate(clean,group_column,fold_ids):
    groups=clean[group_column];fold_ids=np.asarray(fold_ids,dtype=int);nfold=int(fold_ids.max()+1)
    if len(fold_ids)!=len(clean):raise ValueError('Fold assignment length mismatch')
    pred={name:np.zeros(len(clean)) for name in ['zero_change','training_median','ridge_mutation_location']}
    folds=[]
    y=clean.ddg_kcal_mol.to_numpy()
    for fold in range(nfold):
        te=np.flatnonzero(fold_ids==fold);tr=np.flatnonzero(fold_ids!=fold)
        if not len(te) or not len(tr):raise ValueError(f'Empty train/test partition in fold {fold}')
        assert not set(groups.iloc[tr]) & set(groups.iloc[te])
        enc=ColumnTransformer([('categorical',OneHotEncoder(handle_unknown='ignore',sparse_output=False),FEATURES)],remainder='drop')
        model=make_pipeline(enc,Ridge(alpha=1.0)) # fixed in advance; no test-set tuning
        model.fit(clean.iloc[tr][FEATURES],y[tr])
        pred['training_median'][te]=np.median(y[tr])
        pred['ridge_mutation_location'][te]=model.predict(clean.iloc[te][FEATURES])
        folds.append({'fold':fold,'train_rows':len(tr),'test_rows':len(te),
            'test_groups':sorted(set(groups.iloc[te])),
            'metrics':{name:metrics(y[te],values[te]) for name,values in pred.items()}})
    perrow=clean[['example_id','complex','mutation',group_column,'ddg_kcal_mol']].copy();perrow['fold']=fold_ids
    for name,values in pred.items():perrow[name]=values
    return {'grouping':group_column,'n_folds':int(nfold),
        'pooled_out_of_fold':{name:metrics(y,values) for name,values in pred.items()},'folds':folds},perrow

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--download',action='store_true')
    parser.add_argument('--write-split-manifest',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];rawpath=root/'data/raw/skempi_v2.csv';rawpath.parent.mkdir(parents=True,exist_ok=True)
    if args.download or not rawpath.exists():
        b=urlopen(Request(URL,headers={'User-Agent':'Educational baseline study'}),timeout=30).read()
        if b'Affinity' not in b[:2000]:raise ValueError('Unexpected download')
        rawpath.write_bytes(b)
    dataset_sha256=hashlib.sha256(rawpath.read_bytes()).hexdigest()
    raw=pd.read_csv(rawpath,sep=';');clean,audit=prepare(raw)
    manifest_path=root/'splits/split_manifest_v1.json'
    if args.write_split_manifest:
        manifest_path.parent.mkdir(exist_ok=True)
        manifest=build_split_manifest(clean,dataset_sha256)
        manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    if not manifest_path.exists():raise FileNotFoundError(f'Missing split manifest: {manifest_path}')
    manifest=json.loads(manifest_path.read_text())
    fold_assignments=validate_split_manifest(clean,manifest,dataset_sha256)
    resultdir=root/'results';resultdir.mkdir(exist_ok=True)
    clean.to_csv(resultdir/'processed.csv',index=False)
    runs={}
    for grouping in ['pdb_group','related_group']:
        summary,predictions=evaluate(clean,grouping,fold_assignments[grouping]);runs[grouping]=summary
        predictions.to_csv(resultdir/f'predictions_{grouping}.csv',index=False)
    result={'dataset_sha256':dataset_sha256,'split_manifest':str(manifest_path.relative_to(root)),
        'split_manifest_version':manifest['manifest_version'],'split_fingerprint':manifest['split_fingerprint'],
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__},
        'audit':audit,'features':FEATURES,'target':'RT ln(Kd_mut / Kd_wt), kcal/mol; positive means weaker binding',
        'primary_evaluation':'related_group','experiments':runs}
    (resultdir/'metrics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'audit':audit,'primary_metrics':runs['related_group']['pooled_out_of_fold']},indent=2))
if __name__=='__main__':main()
