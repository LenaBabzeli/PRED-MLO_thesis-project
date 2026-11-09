# PRED-MLO

**PRED-MLO**: A tool for multi-trait post-GWAS analysis that combines summary statistics files using statistical combination methods in order to account for pleiotropy across traits, with optional Z-score and LD-based pruning.

# Source code

The code is available at: [PRED-MLO_OPTIMIZED.py](https://github.com/LenaBabzeli/PRED-MLO_thesis-project/blob/main/PRED-MLO_OPTIMIZED.py)

# Installation guide

PRED-MLO is written in Python (ver. 3.13.5).

1. Clone or download PRED-MLO from: https://github.com/LenaBabzeli/PRED-MLO_thesis-project
2. After downloading the `.zip` folder of PRED-MLO from GitHub, extract it to a working directory.
3. To install the requirements, pip needs to be installed. Download the script for pip from: https://bootstrap.pypa.io/get-pip.py
4. Open a terminal/command prompt, `cd` to the folder containing the `get-pip.py` file and run:

```bash
python3 get-pip.py
```

5. To install the mentioned requirements with pip, open a terminal/command prompt and run:

```bash
pip install -r requirements.txt
```

# Usage

## Input file structure

Ensure each summary statistics file follows the structure below. You should include at least two GWAS summary statistics files as input studies. The `SNP` column should contain rsIDs specifically if LD-pruning should be performed.

| SNP | CHR | POS | A1 | A2 | BETA | OR | SE | P | Z |
|---|---|---|---|---|---|---|---|---|---|
| rsID1 | 10 | 9918166 | A | C | 0.030 | 1.030 | 0.040 | 0.4533 | 0.750 |
| rsID2 | 10 | 98254058 | T | G | 0.070 | 1.073 | 0.035 | 0.0455 | 2.000 |
| rsID3 | 5 | 233875471 | C | T | -0.100 | 0.905 | 0.050 | 0.0455 | -2.000 |
| rsID4 | 9 | 63787744 | A | G | 0.140 | 1.150 | 0.060 | 0.0196 | 2.333 |
| rsID5 | 4 | 98261776 | G | A | 0.200 | 1.221 | 0.080 | 0.0124 | 2.500 |

## Execution command

```bash
python PRED-MLO_OPTIMIZED.py -i [input_filename1.tsv] [input_filename2.tsv] [input_filename3.tsv] -o [output_filename.tsv.gz] --method [method]
```

## Arguments

PRED-MLO accepts the following command-line arguments:

- `-i`: Input tab-separated files with columns: `SNP`, `CHR`, `POS`, `A1`, `A2`, `BETA`, `OR`, `SE`, `P`, `Z`
- `-o`: Output tab-separated file to save the results
- `--method`: Statistical method to analyze the data and calculate the combined p-value
  - `CAUCHY`: (Cauchy Method) Statistical method that combines p-values using the Cauchy transformation, with the combined p-value obtained from the standard Cauchy distribution.
  - `MINP`: (Minimum P-value) Statistical method that combines the minimum p-value across all studies using the Sidak correction to account for multiple testing
  - `HMP`: (Harmonic Mean P-value) Statistical method that uses the harmonic mean of p-values to combine association evidence while allowing for dependence between tests
  - `FISHER`: (Fisher Method) Statistical method that combines independent p-values using Fisher's statistic, which follows a chi-squared distribution with 2K degrees of freedom under the null hypothesis
  - `MCM`: (Minimum Combination Method) Statistical method that combines the Cauchy and MinP results using a minimum-based heuristic, to capture both distributed and sparse association signals
  - `CMC`: (Cauchy Method of Combination) Statistical method that combines the Cauchy and MinP p-values using a second-stage Cauchy combination test, followed by the standard Cauchy transformation
  - `LANCASTER`: (Lancaster Method) Statistical method that generalizes Fisher's method by transforming each p-value into a chi-squared statistic with a specified degree-of-freedom weight, then sums the statistics and evaluates the results against a chi-square distribution
  - `EMPIRICAL_BROWNS`: (Empirical Browns Method) Statistical method that extends Fisher's method to dependent p-values by estimating the variance of Fisher's statistic from empirical covariance matrices and approximating its null distribution using a scaled chi-square distribution
  - `YANG`: (Browns Method by Yang) Statistical method that applies a dependence-adjusted Fisher-type combination using a Gamma distribution, with correlation corrections estimated from the empirical Z-score correlation matrix
  - `WALD_Z_SCORES`: (Wald Test Statistic using Z-scores) Statistical method that performs a multivariate Wald test using the Z-score vector and its estimated covariance matrix, evaluated against a chi-squared distribution with K degrees of freedom
  - `WALD_BETAS_SES`: (Wald Test Statistic using Betas and Standard Errors) Statistical method that performs a multivariate Wald test statistic using effect estimates, standard errors and the correlation matrix to calculate the combined p-value

### Additional arguments

The following options are not required for the tool to perform the analysis. They are optional and allow the user to choose if an option will be used and what value it will have.

- `--R2_cut_off`: Optional LD-based SNP pruning using a predefined linkage disequilibrium (LD) dataset. SNP pairs with (R²) values greater than or equal to the specified threshold are identified, and one SNP from each pair is removed to reduce redundancy due to correlated genetic variants.

  Default: no pruning
- `--ld_dir`: Directory containing the chromosome-specific `UNIVERSAL_LD` parquet files used for LD-based pruning. Required when `--R2_cut_off` is specified
- `--z_cut_off`: Excludes SNPs with large association Z-scores from the correlation estimation dataset to reduce the influence of potentially associated variants on the estimated correlation matrix. This filtering affects the estimation of (R), rather than the final association dataset

  Default: no z_cut_off
- `--study_fraction`: Controls the minimum number of studies in which a SNP must appear. This filtering is applied during the merging process, before the association combination methods are executed.

  Default: no study fraction cut_off

## Example

LD-based pruning:

```bash
python PRED-MLO_OPTIMIZED.py -i [input_filename1.tsv] [input_filename2.tsv] [input_filename3.tsv] -o [output_filename.tsv.gz] --method [method] --R2_cut_off [LD-pruning cut-off value] --ld_dir [LD-files path]
```

## Example

Z-score filtering and study fraction:

```bash
python PRED-MLO_OPTIMIZED.py -i [input_filename1.tsv] [input_filename2.tsv] [input_filename3.tsv] -o [output_filename.tsv.gz] --method [method] --z_cut_off [Z-score cut-off value] --study_fraction [number of studies in which a SNP must appear in]
```
