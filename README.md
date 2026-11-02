# Mutation binding baseline

An AI-assisted educational experiment predicting mutation-associated changes in protein-protein binding affinity using SKEMPI 2.0. This first version is a tabular baseline, not a novel biological model, paper reproduction, clinically validated tool, or structure-based predictor.

## Setup and commands

Use Python 3.12 for parity with continuous integration. The exact canonical-run environment is recorded in `results/metrics.json`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the scientific checks:

```bash
python -m unittest discover -s tests -v
```

Reproduce the evaluation from the bundled raw CSV and frozen split manifest:

```bash
python src/baseline.py
```

Evaluation loads the versioned `splits/split_manifest_v1.json` file and refuses
to run if its raw-data fingerprint, processed-example fingerprint, coverage,
unique example IDs, or group separation checks fail. `--write-split-manifest`
is reserved for an explicit split-version update; ordinary reproduction runs must
not use it or `--download`.

The included raw CSV makes the experiment runnable without a fresh dataset download. To explicitly refresh upstream data, run `python src/baseline.py --download`; this overwrites the raw CSV and will change the recorded hash if upstream data changes.

## Research question

Can a fixed linear model using only wild-type amino acid, mutant amino acid, and mutation location outperform simple constant predictors when related complexes are held together during evaluation?

## Data and target

Upstream: https://life.bsc.es/pid/skempi2/database/download/skempi_v2.csv

7,085 raw records. This version retains single substitutions between standard amino acids; it excludes censored, invalid, or missing affinity values and excludes temperatures annotated as assumed or otherwise nonnumeric. Exact temperatures must fall between 250 and 350 kelvin. Those are explicit first-experiment choices, not universal SKEMPI preprocessing rules.

Target: `delta_delta_G = (8.314/4184) * temperature_K * ln(Kd_mut / Kd_wt)` in kcal/mol, following the upstream FAQ. Positive values indicate weaker mutant binding. Experimental affinities are used to derive labels and NEVER enter model features.

There are 3,336 retained measurements. Repeated measurements for the same cleaned complex/mutation are consolidated by median target, giving 2,861 examples. The processed data preserves replicate counts and target ranges. This aggregation ignores some differences in assay, reference and temperature and is a stated limitation. Location uses the first observed annotation.

## Splits and model protocol

The fixed model is one-hot encoded amino-acid identities and location followed by Ridge(alpha=1). Preprocessing fits on training folds only; unseen categories are ignored at inference. No hyperparameter tuning or test-result-based model selection was performed in this version.

Two manifest-backed deterministic five-fold evaluations are reported:

1. PDB-structure grouping, including chain variants sharing the same PDB accession.
2. Primary: connected groups built from all raw records using shared PDB accession plus upstream Hold_out_type and Hold_out_proteins links. These group assignments do not use target values. Related examples linked by that metadata cannot cross folds.

Groups are allocated largest first. Equal-sized groups are ordered by their
canonical group label, and equal-load folds are ordered by fold number. This
explicit tie-breaking is independent of input row order and does not use target
values. Stable example IDs bind each processed complex/mutation to its fold.

The conservative primary grouping produces 60 groups, but one connected component has 1,323 of 2,861 examples. Primary test-fold sizes are 1,323 / 385 / 385 / 384 / 384. The broad holdout types can merge unrelated complexes, making this a conservative and imbalanced stress test. It is not an official split, sequence-identity clustering, or proof that all biological leakage is eliminated.

Constant baselines predict zero change and the training-fold target median. The median baseline is an appropriate simple comparator for mean absolute error. Model outputs for every held-out example and the exact fold assignments are saved.

## Canonical split v1 results

Primary related-group out-of-fold results, pooled over examples:

| Model | MAE kcal/mol | RMSE kcal/mol |
| --- | ---: | ---: |
| Zero change | 1.3814 | 2.1342 |
| Training median | 1.2644 | 1.9609 |
| Mutation/location ridge | 1.2047 | 1.8224 |

PDB-only ridge MAE is 1.1519; related-group ridge MAE is 1.2047. This descriptive difference illustrates sensitivity to grouping. Do not treat it as a causal estimate of leakage. The ridge has modestly lower pooled error than constants in this run. No statistical significance, state-of-the-art, or broad generalization claim is established.

These are cross-validation results, not a separate untouched final test. Before future iterative tuning, freeze an independent final holdout or use nested grouped evaluation. Do not repeatedly optimize against these reported folds and continue calling them untouched test data.

## Files

- src/baseline.py: ingestion, cleaning, target derivation, grouping, and evaluation.
- tests/test_baseline.py: ten checks covering units/sign, censored values, invalid inputs, transitive grouping, shared structures, feature boundaries, aggregation, fold coverage, repeated allocation, and reordered input rows.
- results/metrics.json: complete audit, dependency versions, data hash, pooled and per-fold scores and held-out groups.
- results/processed.csv: retained and consolidated examples.
- results/predictions_*.csv: held-out labels, fold assignments and predictions.
- splits/split_manifest_v1.json: versioned example-to-fold assignments and fingerprints.
- data/raw/provenance.json: dataset provenance and attribution.
- .github/workflows/tests.yml: configured CI workflow. Local tests passed; this workflow has not run on GitHub yet.

## Reproducibility and development disclosure

The repository preserves the raw-data SHA-256, stable processed-example IDs,
canonical split fingerprint, dependency versions, per-row predictions, fold
membership, and pooled/per-fold metrics. A reproduction run validates the raw
dataset and processed-example fingerprints, complete manifest coverage, unique
IDs, and train/test group separation before fitting any model. The split allocator
uses group sizes and canonical group labels, never target values.

The implementation and documentation were developed with AI assistance and then
reviewed through local tests and artifact comparisons. The saved GitHub workflow
defines CI but a green local run must not be represented as a successful GitHub
Actions run.

## Ownership and next steps

Before presenting this work, inspect the code, reproduce it locally, and be able
to explain the scientific decisions. Do not imply unaided authorship or completed
scientific expertise.

First extensions: investigate fold imbalance and annotation consistency; inspect error by mutation and holdout group; document data exclusions; add a frozen final holdout before tuning; compare biologically motivated features. Sequence embeddings and neural networks come after a sound baseline. Validate single versus multiple substitutions separately if adding the latter.

## Dataset attribution and license

SKEMPI2 copyright 2018 BSC. Data authors: Jankauskaite J, Jimenez-Garcia B, Dapkunas J, Fernandez-Recio J, Moal IH. (2019). SKEMPI 2.0: an updated benchmark of changes in protein-protein binding energy, kinetics and thermodynamics upon mutation. Bioinformatics 35, 462-469. https://doi.org/10.1093/bioinformatics/bty635

Upstream terms: https://life.bsc.es/pid/skempi2/info/terms
CC BY 4.0: https://creativecommons.org/licenses/by/4.0/
FAQ and format: https://life.bsc.es/pid/skempi2/info/faq_and_help

The upstream terms apply the Creative Commons Attribution 4.0 International
license and permit sharing with its conditions. The bundled raw CSV is included
unmodified. Processed examples are filtered and aggregated derivatives as
explained above. Attribution, copyright, source, license link, modification
status, and the upstream warranty disclaimer are retained here. Upstream provides
the data as-is and without warranties. This educational project is not endorsed
by BSC or the dataset authors.
