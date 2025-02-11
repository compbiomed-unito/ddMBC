# Supplementary material

Supplementary material for the ddMBC paper on Protein Science

ref: doi_to_be_assigned
 
## ESM-IF directory ##

./ESM-IF:

### input data

S461.mut                                  
s669_af_muts.csv
s669_muts.csv

### ESM-IF evaluation results

esmif_fixed.csv
esmif_s669_af.csv
esmif_s669_fixed.csv

### evaluate and analyze ddMBC correction

esmif_plain_double.ipynb
esmif_s669.csv

### ESM-IF evaluation on training set and test set

esmif_run.py                              
esmif_s669_run.py
esmif_s669_af_run.py

## FoldX directory ##

./FoldX:

### input data

s461_muts.csv
s669_muts.csv
Data_s669_with_predictions.csv
Ssym+_experimental.csv

### evaluate and analyze ddMBC correction

foldx_ddfep.ipynb

### estimate max pearson on set

pcc_max.ipynb

## Pythia directory ##

./Pythia:

### input data

S461.mut
all_muts_pythia_in.csv
augd_pythia_in.csv
s669_pythia_af_in.csv
s669_pythia_in.csv
esmif_late_fixed.csv
pythia_all_train.csv
pythia_s669_af.csv
pythia_s669_augd.csv

### evaluate and analyze ddMBC correction

pythia_double.ipynb

### estimate baseline ddmbc performance

aa_baseline.ipynb

### analysis outputs for graphs

pythia_ddmbc_test_af.csv
pythia_ddmbc_test_pdb.csv
pythia_ddmbc_train.csv

## Stability Oracle directory ##

./StabilityOracle:

### input data

S669.csv
S669_r.csv
s461_muts.csv

### evaluate and analyze ddMBC correction

so_s669.ipynb

