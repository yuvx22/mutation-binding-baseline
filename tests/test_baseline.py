import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from baseline import (exact_float,ddg,holdout_groups,prepare,FEATURES,evaluate,
    deterministic_group_folds,build_split_manifest,validate_split_manifest)
import numpy as np
import pandas as pd

class ScientificChecks(unittest.TestCase):
    def test_sign_and_units(self):
        self.assertAlmostEqual(ddg(1e-8,1e-9,298),1.363485,places=6)
        self.assertAlmostEqual(ddg(1e-9,1e-9,298),0)
        self.assertLess(ddg(1e-10,1e-9,298),0)
    def test_qualifiers_not_silently_stripped(self):
        for x in ['>1e-6','<1e-9','298(assumed)','nan','inf','']:
            self.assertTrue(np.isnan(exact_float(x)))
    def test_nonpositive_target_inputs_rejected(self):
        with self.assertRaises(ValueError):ddg(0,1e-9,298)
    def test_transitive_holdout_groups(self):
        r=pd.DataFrame({'#Pdb':['1AAA_A_B','2BBB_A_B','3CCC_A_B','4DDD_A_B'],
            'Hold_out_type':[np.nan]*4,'Hold_out_proteins':['2BBB_A_B','3CCC_A_B',np.nan,np.nan]})
        g=holdout_groups(r)
        self.assertEqual(g['1AAA_A_B'],g['3CCC_A_B']);self.assertNotEqual(g['1AAA_A_B'],g['4DDD_A_B'])
    def test_same_structure_chain_variants_grouped(self):
        r=pd.DataFrame({'#Pdb':['1AAA_A_B','1AAA_C_D'],'Hold_out_type':[np.nan]*2,'Hold_out_proteins':[np.nan]*2})
        g=holdout_groups(r);self.assertEqual(g['1AAA_A_B'],g['1AAA_C_D'])
        self.assertEqual(g,holdout_groups(r.iloc[::-1].reset_index(drop=True)))
    def test_target_inputs_not_features(self):
        self.assertFalse(set(FEATURES)&{'kmut','kwt','ddg_kcal_mol','temperature_k','Reference'})
    def test_filtering_and_duplicate_median(self):
        row={'#Pdb':'1AAA_A_B','Mutation(s)_cleaned':'AA1G','Hold_out_type':np.nan,'Hold_out_proteins':np.nan,
             'Affinity_mut (M)':'1e-8','Affinity_wt (M)':'1e-9','Temperature':'298','iMutation_Location(s)':'COR'}
        rows=[row,row.copy(),dict(row,**{'Mutation(s)_cleaned':'AA1G,GA2A'}),dict(row,Temperature='298(assumed)'),dict(row,**{'Affinity_mut (M)':'<1e-8'})]
        c,a=prepare(pd.DataFrame(rows));self.assertEqual(len(c),1);self.assertEqual(c.iloc[0].replicates,2)
        self.assertEqual(a['retained_measurements'],2)
    def test_folds_keep_groups_disjoint_and_cover_rows(self):
        rows=[]
        for i in range(10):
            for j in range(2):rows.append({'complex':f'{i}','mutation':f'A{j}G','related_group':str(i),
                'ddg_kcal_mol':float(i+j),'wt_aa':'A','mut_aa':'G','location':'COR'})
        c=pd.DataFrame(rows);c['example_id']=[f'id-{i}' for i in range(len(c))]
        folds=deterministic_group_folds(c,'related_group');s,p=evaluate(c,'related_group',folds)
        self.assertEqual(len(p),20);self.assertEqual(p.groupby('related_group').fold.nunique().max(),1)
        self.assertTrue(np.isfinite(p.ridge_mutation_location).all())
    def test_deterministic_assignments_survive_row_reordering(self):
        rows=[{'example_id':f'id-{i}-{j}','complex':f'c{i}','mutation':f'A{j}G',
            'pdb_group':f'p{i}','related_group':f'r{i//2}'} for i in range(12) for j in range(i%3+1)]
        clean=pd.DataFrame(rows)
        first=build_split_manifest(clean,'abc')
        reordered=build_split_manifest(clean.sample(frac=1,random_state=17).reset_index(drop=True),'abc')
        self.assertEqual(first,reordered)
    def test_deterministic_assignments_repeat_and_validate(self):
        rows=[{'example_id':f'id-{i}','complex':f'c{i}','mutation':'A1G',
            'pdb_group':f'p{i%7}','related_group':f'r{i%4}'} for i in range(20)]
        clean=pd.DataFrame(rows);first=build_split_manifest(clean,'abc');second=build_split_manifest(clean,'abc')
        self.assertEqual(first,second)
        validated=validate_split_manifest(clean,first,'abc')
        self.assertEqual(set(validated),{'pdb_group','related_group'})
if __name__=='__main__':unittest.main()
