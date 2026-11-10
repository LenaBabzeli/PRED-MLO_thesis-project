import pandas as pd
import numpy as np
import os
os.environ["POLARS_MAX_THREADS"] = "10"
import polars as pl
import argparse
import math
import sys
import gc
import psutil
import threading
from scipy.stats import norm
from scipy.stats import gamma
from scipy.stats import chi2
from functools import reduce
from collections import defaultdict
import concurrent.futures
import multiprocessing
from scipy.special import ndtr, ndtri
import concurrent.futures
from numba import jit

def main():
    parser = argparse.ArgumentParser(description='Merge GWAS summary statistics files by SNPID_UKB')

    parser.add_argument('-i', '--input', nargs='+', required=True,
                        help='Input requires at least two GWAS files, .tsv format. Example: gwas1.tsv, gwas2.tsv')

    parser.add_argument('-o', '--output', default='results.tsv.gz', help='Output file path. Example: results.tsv.gz')

    parser.add_argument('--method', type=str,
                        choices=['CAUCHY', 'MINP', 'HMP', 'FISHER','MCM', 'CMC', 'LANCASTER',
                                'EMPIRICAL_BROWNS', 'YANG', 'WALD_Z_SCORES', 'WALD_BETAS_SES'],
                        help='Select a multiple outcome method: CAUCHY, MINP, HMP, FISHER, MCM, CMC, LANCASTER, EMPIRICAL_BROWNS, YANG, WALD_Z_SCORES, WALD_BETAS_SES')

    parser.add_argument('--R2_cut_off', type=float,
                        help='R2 threshold for LD-based pruning when estimating R correlation matrix. Default: no pruning.')

    parser.add_argument('--ld_dir', required=False, help='Directory containing UNIVERSAL_LD files. Ensure that the base_dir in the code and ld_dir path are the same.')

    parser.add_argument('--z_cut_off', type=float, default=None,
                        help='Z-score threshold for null SNP filtering when estimating R. Default: no filtering.')

    parser.add_argument('--study_fraction', type=float, default=None,
                        help='Minimum number of studies a SNP must appear in. Used as a percentage. Example: 0.4, SNPs that appear in 40 percent of the studies are kept.')

    args = parser.parse_args()

    if len(args.input) < 2:
        print("More than two input files are required.", file=sys.stderr)
        sys.exit(1)

    K = len(args.input)
    if args.study_fraction is not None:
        min_study_count = max(1, math.ceil(args.study_fraction * K))
        print(f" Filtering SNPs: must appear in at least {min_study_count}/{K} studies.")
    else:
        min_study_count = 1

        #Enable the cache before joining the dataframes
        pl.enable_string_cache()

        #Start joining the dataframes based on their 'SNP' column
        required_cols = ['SNP', 'CHR', 'POS', 'A1', 'A2', 'BETA', 'SE', 'Z']
        #required_cols = ['SNP', 'A1', 'A2', 'BETA', 'SE', 'Z']

        pl_dtypes = {'CHR': pl.UInt8, 'POS': pl.Int32, 'A1': pl.Categorical, 'A2': pl.Categorical,
                    'BETA': pl.Float32, 'SE': pl.Float32, 'Z': pl.Float32}
        
        #pl_dtypes = {'A1': pl.Categorical, 'A2': pl.Categorical,
        #            'BETA': pl.Float32, 'SE': pl.Float32, 'Z': pl.Float32}

        lf_merged = pl.scan_csv(args.input[0], separator='\t', schema_overrides=pl_dtypes).select(required_cols)
        rename_map = {'BETA': 'BETA1', 'Z': 'Z1', 'SE': 'SE1'}
        lf_merged = lf_merged.rename(rename_map)

        join_keys = ['SNP', 'CHR', 'POS', 'A1', 'A2']
        #join_keys = ['SNP', 'A1', 'A2']

        for i, file_path in enumerate(args.input[1:], start=2):
            lf_next = pl.scan_csv(file_path, separator='\t', schema_overrides=pl_dtypes).select(required_cols)
            rename_map = {'BETA': f'BETA{i}', 'Z': f'Z{i}', 'SE': f'SE{i}'}
            lf_next = lf_next.rename(rename_map)
            lf_merged = lf_merged.join(lf_next, on=join_keys, how="full", coalesce=True)

        beta_cols_pl = [f'BETA{i}' for i in range(1, K + 1)]

        lf_merged = lf_merged.with_columns(
            pl.sum_horizontal(pl.col(beta_cols_pl).is_not_null()).alias("N")
        ).filter(pl.col("N") >= min_study_count)

        #Process data in batches to prevent crashing
        df_merged = lf_merged.collect(engine="streaming")

        #Disable the cache after joining the dataframes
        pl.disable_string_cache()

        df_merged = df_merged.sort(["CHR", "POS"])

        z_cols = [f'Z{i}' for i in range(1, K + 1)]
        beta_cols = [f'BETA{i}' for i in range(1, K + 1)]
        se_cols = [f'SE{i}' for i in range(1, K + 1)]

        final_columns = ['SNP', 'CHR', 'POS', 'A1', 'A2'] + beta_cols + se_cols + z_cols + ['N']
        #final_columns = ['SNP', 'A1', 'A2'] + beta_cols + se_cols + z_cols + ['N']

        df_merged = df_merged.select(final_columns)
        df_for_corr = df_merged.select(["SNP", "CHR"] + z_cols)
        #df_for_corr = df_merged.select(["SNP"] + z_cols)

    #Z-BASED FILTERING, to avoid polygenic inflation (optional -- provide threshold)
    if args.z_cut_off is not None:
        print(f"Applying Z-based filtering (|Z| <= {args.z_cut_off}) for R estimation.")

        z_cols = [col for col in df_for_corr.columns if col.startswith('Z')]
        if z_cols:
            Z_matrix = df_for_corr.select(z_cols).to_numpy().astype(np.float32)
            within_threshold = (np.abs(Z_matrix) <= args.z_cut_off) | (~np.isfinite(Z_matrix))
            keep_rows = np.all(within_threshold, axis=1)
            #Use polars boolean for filtering
            df_for_corr = df_for_corr.filter(pl.Series(keep_rows))
        else:
            print(f"No Z-columns found. Skipping Z-filtering.")
    else:
        print(f"Skipping Z-based filtering.")

    #R2-BASED FILTERING (optional -- provide threshold)
    if args.R2_cut_off is not None:
        print(f"Applying R2-based filtering (Pruning R2 >= {args.R2_cut_off})...")

        if 'SNP' not in df_for_corr.columns:
            print("No SNP columns found. Skipping R2-based filtering.")
            sys.exit(1)

        total_snps = df_for_corr.height
        print(f"Total SNPs in merged dataset before pruning: {total_snps}")

        snps_to_exclude = []

        CHROM_LIST = [str(c) for c in range(1, 23)]

        if total_snps == 0:
            print("No SNPs to check for LD.")

        else:
            #---------------CHROMOSOME PROCESSING---------------

            #Drop nulls, strip 'rs' and cast to pure Int64 natively
            df_valid = (
                df_for_corr.select(["SNP", "CHR"])
                .drop_nulls()
                .with_columns(
                    pl.col("SNP").str.replace("rs", "").cast(pl.Int64, strict=False).alias("Target_SNP")
                )
                .drop_nulls(subset=["Target_SNP"])
            )

            #Extract the unique chromosomes
            unique_chrs = df_valid.get_column("CHR").unique().to_list()

            #Build the chromosome dictionary
            chrom_dict = {}
            for c in unique_chrs:
                
                c_array = df_valid.filter(pl.col("CHR") == c).get_column("Target_SNP").unique().to_numpy()
                chrom_dict[str(c)] = c_array
                
            #---------------PARALLEL PROCESSING---------------
            
            del df_valid
            executor = concurrent.futures.ProcessPoolExecutor(max_workers=2)
            future_to_chrom = {}

            try:
                def get_ld_path(chrom):
                    base_dir = r"C:/Users/lenaz/Desktop/LDfiles/"
                    return f"{base_dir}/UNIVERSAL_LD_chr{chrom}.parquet"
            
                #Check for chromosomes in LD files from path
                valid_chroms = [
                    c for c in CHROM_LIST
                    if os.path.exists(get_ld_path(c))
                ]

                if not valid_chroms:
                    print(" No LD files found. Skipping.")
                    executor.shutdown(wait=True)
                else:
                    print(f"Found {len(valid_chroms)} valid LD files. Starting partitioned threads...")

                    future_to_chrom = {
                        executor.submit(TOP_LD_info, chrom_dict.get(chrom, []), chrom,
                                        args.R2_cut_off, args.ld_dir): chrom
                        for chrom in valid_chroms
                    }

                    #Collect results
                    for future in concurrent.futures.as_completed(future_to_chrom):
                        chrom = future_to_chrom[future]
                        try:
                            loser_arr = future.result()
                            if len(loser_arr) > 0:
                                snps_to_exclude.append(loser_arr)
                        except Exception as exc:
                            print(f"  - Chr {chrom} generated an exception: {exc}")

                    #Normal shutdown if everything finishes safely
                    executor.shutdown(wait=True)

            except KeyboardInterrupt:
                
                #Cancel pending tasks if keyboard interrupt
                for future in future_to_chrom:
                    future.cancel()

                executor.shutdown(wait=False, cancel_futures=True)
                os._exit(1)

            gc.collect()

            before = df_for_corr.height

            if snps_to_exclude:
                #Combine all 22 integer arrays into one solid C-array
                flat_losers = np.concatenate(snps_to_exclude)
                
                loser_series = pl.Series("losers", flat_losers)
                loser_strings = pl.lit("rs") + loser_series.cast(pl.Utf8)

                df_for_corr = df_for_corr.filter(~pl.col("SNP").is_in(loser_strings))

            after = df_for_corr.height
        print(f"Excluded {before - after} SNPs in LD (R2 >= {args.R2_cut_off}).")
    else:
        print("Skipping R2-based LD filtering.")


    method_names = {
        'CAUCHY': 'CAUCHY',
        'MINP': 'MINP',
        'HMP': 'HMP',
        'FISHER': 'FISHER',
        'MCM': 'MCM',
        'CMC': 'CMC',
        'LANCASTER': 'LANCASTER',
        'YANG': 'YANG',
        'EMPIRICAL_BROWNS': 'EMPIRICAL_BROWNS',
        'WALD_Z_SCORES': 'WALD_Z_SCORES',
        'WALD_BETAS_SES': 'WALD_BETAS_SES'
    }

    if args.method is None:
        print("No method specified. Default method will be used: CAUCHY")
        choice = 'CAUCHY'
    else:
        choice = args.method

    method_functions = {
        'CAUCHY': cauchy_method,
        'MINP': minP,
        'HMP': HMP,
        'FISHER' : fisher_method,
        'MCM': MCM,
        'CMC': CMC,
        'LANCASTER' : lancaster_method,
        'EMPIRICAL_BROWNS': EmpiricalBrownsMethod,
        'YANG': lambda df_merged: BrownsMethodbyYang(df_merged, df_for_corr=df_for_corr),
        'WALD_Z_SCORES': lambda df_merged: MultivariateWaldTest_Zscores(df_merged, df_for_corr=df_for_corr),
        'WALD_BETAS_SES': lambda df_merged: MultivariateWaldTest_BetasSes(df_merged, df_for_corr=df_for_corr)
    }

    method_func = method_functions[choice]

    df_result = method_func(df_merged)

    p_cols = [col for col in df_result.columns if col.endswith('_p')]
    stat_cols = [col for col in df_result.columns if 'Wald' in col and not col.endswith('_p')]

    #Keep combined_p cols as Float32
    cast_exprs = [
                    pl.col("CHR").cast(pl.Int8),
                    pl.col("N").cast(pl.Int8)
                ] + [pl.col(col).cast(pl.Float32) for col in stat_cols]

    df_result = df_result.with_columns(cast_exprs).drop(beta_cols + se_cols + z_cols)

    #Save results
    output_base = args.output
    method_suffix = method_names[choice]
    if output_base.endswith('.tsv.gz'):
        base_name = output_base[:-7]
    elif output_base.endswith('.gz'):
        base_name = output_base[:-3]
    else:
        base_name = os.path.splitext(output_base)[0]

    output_filename = f"{base_name}_{method_suffix}.tsv.gz"
    df_result.write_csv(output_filename, separator='\t')


#Helper function for LD-pruning
@jit(nopython=True)
def fast_grim_reaper(s1_array, s2_array):
    losers = {np.int64(-1)}
    for i in range(len(s1_array)):
        s1 = s1_array[i]
        s2 = s2_array[i]

        if s1 in losers or s2 in losers:
            continue

        #Always drop the second SNP arbitrarily
        losers.add(s2)

    losers.remove(np.int64(-1))
    out_arr = np.empty(len(losers), dtype=np.int64)

    idx = 0
    for val in losers:
        out_arr[idx] = val
        idx += 1

    return out_arr

#TOP-LD info function
def TOP_LD_info(chrom_target_list, chrom, R2_threshold, base_dir):
    universal_ld_path = os.path.join(base_dir, f"UNIVERSAL_LD_chr{chrom}.parquet")

    if not os.path.exists(universal_ld_path) or len(chrom_target_list) == 0:
        return np.array([], dtype=np.int64)

    scaled_threshold = int(round(R2_threshold * 100))
    
    try:
        #We only need the SNP column
        target_lf = pl.DataFrame({"Target_SNP": chrom_target_list}).lazy()
        matched_df = (
            pl.scan_parquet(universal_ld_path)
            .filter(pl.col("R2") >= scaled_threshold)
            .join(target_lf, left_on="rsID1", right_on="Target_SNP", how="inner")
            .join(target_lf, left_on="rsID2", right_on="Target_SNP", how="inner")
            .select(["rsID1", "rsID2"])
            .collect(streaming=True)
        )

        if matched_df.height > 0:
            s1_arr = matched_df.get_column("rsID1").to_numpy()
            s2_arr = matched_df.get_column("rsID2").to_numpy()
            losers_int = fast_grim_reaper(s1_arr, s2_arr)
            return losers_int

    except Exception as e:
        print(f" [SKIP] Chrom {chrom}: {e}")

    return np.array([], dtype=np.int64)


def cauchy_method(df_merged):
    
    z_columns = [col for col in df_merged.columns if col.startswith('Z')]
    if not z_columns:
        raise ValueError("No z-score columns found.")

    Z = df_merged.select(z_columns).to_numpy().astype(np.float32)
    n_snps = Z.shape[0]

    cauchy_p = np.full(n_snps, np.nan, dtype=np.float64)

    finite_mask = np.isfinite(Z)
    finite_counts = finite_mask.sum(axis=1)
    
    #For SNPs with only 1 valid z-score
    single_study_snps = finite_counts == 1
    if np.any(single_study_snps):
        
        Z_single = Z[single_study_snps]
        mask_single = finite_mask[single_study_snps]

        valid_idx = np.argmax(mask_single, axis=1)

        z_vals = Z_single[np.arange(Z_single.shape[0]), valid_idx]
        cauchy_p_single = z_vals
        cauchy_p[single_study_snps] = cauchy_p_single
    
    #For SNPs with at least 2 valid z-scores
    valid_snps = finite_counts >= 2
    if np.any(valid_snps):
        
        Z_valid = Z[valid_snps]
        finite_mask_valid = finite_mask[valid_snps]
        k_valid_vec = finite_counts[valid_snps]

        #Transform to p-values
        p1, p2 = transform_to_pvalues(Z_valid, missing_value=0.5)
        
        eps = 1e-15
        p1 = np.clip(p1, eps, 1 - eps)
        p2 = np.clip(p2, eps, 1 - eps)

        T1 = np.tan((0.5 - p1) * np.pi)
        T1[~finite_mask_valid] = 0.0
        t1 = np.sum(T1, axis=1) / k_valid_vec
        t1 = np.nan_to_num(t1, nan=0.0, posinf=1e15, neginf=-1e15)
        #Calculate the combined p-value using the Cauchy distribution
        combined_p1 = 0.5 - (np.arctan(t1) / np.pi)

        T2 = np.tan((0.5 - p2) * np.pi)
        T2[~finite_mask_valid] = 0.0
        t2 = np.sum(T2, axis=1) / k_valid_vec
        t2 = np.nan_to_num(t2, nan=0.0, posinf=1e15, neginf=-1e15)
        #Calculate the combined p-value using the Cauchy distribution
        combined_p2 = 0.5 - (np.arctan(t2) / np.pi)

        cauchy_p_valid = 2 * np.minimum(combined_p1, combined_p2)
        cauchy_p_valid = np.nan_to_num(cauchy_p_valid, nan=1.0)
        cauchy_p_valid = np.clip(cauchy_p_valid, 0.0, 1.0)
        cauchy_p[valid_snps] = cauchy_p_valid
        
    return df_merged.with_columns(
        pl.Series(name="Cauchy_p", values=cauchy_p)
    )


def minP(df_merged):
    
    z_columns = [col for col in df_merged.columns if col.startswith('Z')]
    if not z_columns:
        raise ValueError("No z-score columns found.")

    Z = df_merged.select(z_columns).to_numpy().astype(np.float32)
    n_snps = Z.shape[0]

    minp_p = np.full(n_snps, np.nan, dtype=np.float64)

    finite_mask = np.isfinite(Z)
    finite_counts = finite_mask.sum(axis=1)
    
    #For SNPs with only 1 valid z-score
    single_study_snps = finite_counts == 1
    if np.any(single_study_snps):
        
        Z_single = Z[single_study_snps]
        mask_single = finite_mask[single_study_snps]

        valid_idx = np.argmax(mask_single, axis=1)

        z_vals = Z_single[np.arange(Z_single.shape[0]), valid_idx]
        cauchy_p_single = z_vals
        minp_p[single_study_snps] = cauchy_p_single
    
    #For SNPs with at least 2 valid z-scores
    valid_snps = finite_counts >= 2
    if np.any(valid_snps):
        
        Z_valid = Z[valid_snps]
        k_valid_vec = finite_counts[valid_snps]
        
        #Transform to p-values
        p1, p2 = transform_to_pvalues(Z_valid, missing_value=1.0)
        
        eps = 1e-15
        p1 = np.clip(p1, eps, 1 - eps)
        p2 = np.clip(p2, eps, 1 - eps)
        
        #Calculate the final combined p-value as the minimum p-value
        min_p1 = np.min(p1, axis=1)
        min_p2 = np.min(p2, axis=1)
        
        combined_p1 = 1.0 - np.power(1.0 - min_p1, k_valid_vec)
        combined_p2 = 1.0 - np.power(1.0 - min_p2, k_valid_vec)
        minp_p_valid = 2.0 * np.minimum(combined_p1, combined_p2)
        
        minp_p_valid = np.clip(minp_p_valid, 0.0, 1.0)
        minp_p[valid_snps] = minp_p_valid

    return df_merged.with_columns(
        pl.Series(name="MinP_p", values=minp_p)
    )


def HMP(df_merged):
    
    z_columns = [col for col in df_merged.columns if col.startswith('Z')]
    if not z_columns:
        raise ValueError("No z-score columns found.")

    Z = df_merged.select(z_columns).to_numpy().astype(np.float32)
    n_snps = Z.shape[0]

    hmp_p = np.full(n_snps, np.nan, dtype=np.float64)

    finite_mask = np.isfinite(Z)
    finite_counts = finite_mask.sum(axis=1)
    
    #For SNPs with only 1 valid z-score
    single_study_snps = finite_counts == 1
    if np.any(single_study_snps):
        
        Z_single = Z[single_study_snps]
        mask_single = finite_mask[single_study_snps]

        valid_idx = np.argmax(mask_single, axis=1)

        z_vals = Z_single[np.arange(Z_single.shape[0]), valid_idx]
        cauchy_p_single = z_vals
        hmp_p[single_study_snps] = cauchy_p_single
    
    #For SNPs with at least 2 valid z-scores
    valid_snps = finite_counts >= 2
    if np.any(valid_snps):
        
        Z_valid = Z[valid_snps]
        k_valid_vec = finite_counts[valid_snps]
        
        #Transform to p-values
        p1, p2 = transform_to_pvalues(Z_valid, missing_value=np.inf)
        
        eps = 1e-15
        p1 = np.clip(p1, eps, 1 - eps)
        p2 = np.clip(p2, eps, 1 - eps)
        
        #Calculate the combined p-value using the harmonic mean p-value statistic
        hmp1 = k_valid_vec / np.sum(1.0 / p1, axis=1)
        hmp2 = k_valid_vec / np.sum(1.0 / p2, axis=1)
        
        hmp_p_valid = 2 * np.minimum(hmp1, hmp2)
        hmp_p_valid = np.clip(hmp_p_valid, 0.0, 1.0)
        hmp_p[valid_snps] = hmp_p_valid

    return df_merged.with_columns(
        pl.Series(name="HMP_p", values=hmp_p)
    )


def fisher_method(df_merged):
    
    z_columns = [col for col in df_merged.columns if col.startswith('Z')]
    if not z_columns:
        raise ValueError("No z-score columns found.")

    Z = df_merged.select(z_columns).to_numpy().astype(np.float32)
    n_snps = Z.shape[0]

    fisher_p = np.full(n_snps, np.nan, dtype=np.float64)

    finite_mask = np.isfinite(Z)
    finite_counts = finite_mask.sum(axis=1)
    
    #For SNPs with only 1 valid z-score
    single_study_snps = finite_counts == 1
    if np.any(single_study_snps):
        Z_single = Z[single_study_snps]
        mask_single = finite_mask[single_study_snps]

        valid_idx = np.argmax(mask_single, axis=1)

        z_vals = Z_single[np.arange(Z_single.shape[0]), valid_idx]
        fisher_p_single = z_vals
        fisher_p[single_study_snps] = fisher_p_single
    
    #For SNPs with at least 2 valid z-scores
    valid_snps = finite_counts >= 2
    if np.any(valid_snps):
    
        Z_valid = Z[valid_snps]
        k_valid_vec = finite_counts[valid_snps]
        
        #Transform to p-values
        p1, p2 = transform_to_pvalues(Z_valid, missing_value=1.0)
        
        eps = 1e-15
        p1 = np.clip(p1, eps, 1 - eps)
        p2 = np.clip(p2, eps, 1 - eps)
        
        #Degrees of freedom for the chi-squared distribution
        dof = 2 * k_valid_vec
        
        #Compute log-transformed p-value for positive and negative direction
        chi_squared1 = -2 * np.sum(np.log(p1), axis=1)
        combined_p1 = 1 - chi2.cdf(chi_squared1, dof)
        combined_p1 = np.clip(combined_p1, eps, 1 - eps)
        
        chi_squared2 = -2 * np.sum(np.log(p2), axis=1)
        combined_p2 = 1 - chi2.cdf(chi_squared2, dof)
        combined_p2 = np.clip(combined_p2, eps, 1 - eps)

        #Calculation of the final combined p-value
        fisher_p_final = 2 * np.minimum(combined_p1, combined_p2)
        fisher_p_final = np.clip(fisher_p_final, 1e-16, 1.0)
        
        fisher_p[valid_snps] = fisher_p_final
    
    return df_merged.with_columns(
        pl.Series(name="Fisher_p", values=fisher_p)
    )


def MCM(df_merged):
    
    #Compute the Cauchy combined p-values for all SNPs
    df_cauchy = cauchy_method(df_merged)

    #Compute the minP p-values for all SNPs
    df_minp = minP(df_merged)

    cauchy_p = df_cauchy.get_column('Cauchy_p').to_numpy()
    minp_p = df_minp.get_column('MinP_p').to_numpy()

    n_snps = len(cauchy_p)
    
    mcm_p = np.full(n_snps, np.nan)
    
    #If both combined p-values are available
    #Calculation of the final combined MCM p-value using heuristic method
    both_valid = np.isfinite(cauchy_p) & np.isfinite(minp_p)
    mcm_p[both_valid] = 2 * np.minimum(cauchy_p[both_valid], minp_p[both_valid])

    #If only one combined p-value is available, no calculation
    only_cauchy = np.isfinite(cauchy_p) & ~np.isfinite(minp_p)
    mcm_p[only_cauchy] = cauchy_p[only_cauchy]

    only_minp = ~np.isfinite(cauchy_p) & np.isfinite(minp_p)
    mcm_p[only_minp] = minp_p[only_minp]
    
    mcm_p = np.clip(mcm_p, 0.0, 1.0)

    return df_merged.with_columns(pl.Series("MCM_p", mcm_p))


def CMC(df_merged):
    
    #Compute the Cauchy combined p-values for all SNPs
    df_cauchy = cauchy_method(df_merged)

    #Compute the minP p-values for all SNPs
    df_minp = minP(df_merged)
    
    cauchy_p = df_cauchy.get_column('Cauchy_p').to_numpy()
    minp_p = df_minp.get_column('MinP_p').to_numpy()
    
    combined_p_values = np.stack([cauchy_p, minp_p], axis=1)
    
    epsilon = np.finfo(np.float32).tiny
    combined_p_values = np.clip(combined_p_values, epsilon, 1.0 - epsilon)
    
    n_snps = combined_p_values.shape[0]
    
    cmc_p = np.full(n_snps, np.nan)
    
    #If both combined p-values are available
    both_valid = np.isfinite(cauchy_p) & np.isfinite(minp_p)
    eps = 1e-15
    if np.any(both_valid):
        p_pair = np.stack([
            np.clip(cauchy_p[both_valid], eps, 1 - eps),
            np.clip(minp_p[both_valid], eps, 1 - eps)
        ], axis=1)

        #Calculation of the final combined p-value, using the Cauchy statistic
        T = np.tan(np.pi * (0.5 - p_pair))
        t_stat = np.mean(T, axis=1)

        cmc_vals = 0.5 - np.arctan(t_stat) / np.pi
        cmc_vals = np.clip(cmc_vals, 0.0, 1.0)

        cmc_p[both_valid] = cmc_vals

    #If only one combined p-value is available, no calculation
    only_cauchy = np.isfinite(cauchy_p) & ~np.isfinite(minp_p)
    cmc_p[only_cauchy] = cauchy_p[only_cauchy]

    only_minp = ~np.isfinite(cauchy_p) & np.isfinite(minp_p)
    cmc_p[only_minp] = minp_p[only_minp]

    return df_merged.with_columns(pl.Series("CMC_p", cmc_p))


def lancaster_method(df_merged, weight_matrix=None):
    
    z_columns = [col for col in df_merged.columns if col.startswith('Z')]
    if not z_columns:
        raise ValueError("No z-score columns found.")

    Z = df_merged.select(z_columns).to_numpy().astype(np.float32)
    n_snps = Z.shape[0]
    k = Z.shape[1]

    lancaster_p = np.full(n_snps, np.nan, dtype=np.float64)

    finite_mask = np.isfinite(Z)
    finite_counts = finite_mask.sum(axis=1)
    
    #Calculation of the weight matrix
    if weight_matrix is None:
        weights = np.ones(k, dtype=np.float32)
    else:
        weights = np.array(weight_matrix, dtype=np.float32).flatten()
        if len(weights) != k:
            raise ValueError(f"weight_matrix length ({len(weights)}) must match number of studies ({k})")
    
    finite_mask = np.isfinite(Z)
    finite_counts = finite_mask.sum(axis=1)
    
    #For SNPs with only 1 valid z-score
    single_study_snps = finite_counts == 1
    if np.any(single_study_snps):
        Z_single = Z[single_study_snps]
        mask_single = finite_mask[single_study_snps]

        valid_idx = np.argmax(mask_single, axis=1)

        z_vals = Z_single[np.arange(Z_single.shape[0]), valid_idx]
        lancaster_p_single = z_vals
        lancaster_p[single_study_snps] = lancaster_p_single
    
    #For SNPs with at least 2 valid z-scores
    valid_snps = finite_counts >= 2
    if np.any(valid_snps):
    
        Z_valid = Z[valid_snps]
        finite_mask_valid = finite_mask[valid_snps]
        
        #Transform to p-values
        p1, p2 = transform_to_pvalues(Z_valid, missing_value=1.0)
        
        eps = 1e-15
        p1 = np.clip(p1, eps, 1 - eps)
        p2 = np.clip(p2, eps, 1 - eps)
        
        #Lancaster test statistic, for positive and negative direction
        chi_stats1 = chi2.ppf(1.0 - p1, weights)
        chi_stats2 = chi2.ppf(1.0 - p2, weights)
        
        chi_squared1 = np.sum(chi_stats1, axis=1)
        chi_squared2 = np.sum(chi_stats2, axis=1)
        
        #Calculate the degrees of freedom for both directions
        weights_per_snp = np.where(finite_mask_valid, weights, 0.0)
        df_per_snp1 = np.sum(weights_per_snp, axis=1)
        df_per_snp2 = np.sum(weights_per_snp, axis=1)
        
        #Computation of the final combined p-value
        combined_p1 = 1.0 - chi2.cdf(chi_squared1, df_per_snp1)
        combined_p2 = 1.0 - chi2.cdf(chi_squared2, df_per_snp2)
        
        lancaster_p_vals = 2.0 * np.minimum(combined_p1, combined_p2)
        lancaster_p_vals = np.clip(lancaster_p_vals, 0.0, 1.0)

        lancaster_p[valid_snps] = lancaster_p_vals
    
    return df_merged.with_columns(
        pl.Series(name="Lancaster_p", values=lancaster_p)
    )


def EmpiricalBrownsMethod(df_merged):

    z_columns = [col for col in df_merged.columns if col.startswith('Z')]
    if not z_columns:
        raise ValueError("No z-score columns found.")

    Z = df_merged.select(z_columns).to_numpy().astype(np.float32)
    n_snps = Z.shape[0]
    k = Z.shape[1]

    browns_p = np.full(n_snps, np.nan, dtype=np.float64)

    finite_mask = np.isfinite(Z)
    finite_counts = finite_mask.sum(axis=1)

    #For SNPs with only 1 valid z-score
    single_study_snps = finite_counts == 1
    if np.any(single_study_snps):
        
        Z_single = Z[single_study_snps]
        mask_single = finite_mask[single_study_snps]

        valid_idx = np.argmax(mask_single, axis=1)

        z_vals = Z_single[np.arange(Z_single.shape[0]), valid_idx]
        browns_p_single  = z_vals
        browns_p[single_study_snps] = browns_p_single

    #For SNPs with at least 2 valid z-scores
    valid_snps = finite_counts >= 2
    if np.any(valid_snps):
        
        Z_valid = Z[valid_snps]
        
        #Transform to p-values
        p1, p2 = transform_to_pvalues(Z_valid, missing_value=1.0)
        
        eps = 1e-15
        p1 = np.clip(p1, eps, 1 - eps)
        p2 = np.clip(p2, eps, 1 - eps)
        
        n = k
        df_fisher = 2.0 * n
        Expected = 2.0 * n
        
        #Calculate the covariance matrices, for positive and negative direction
        covar_matrix1 = np.cov(p1, rowvar=False)
        covar_matrix2 = np.cov(p2, rowvar=False)
        cov_sum1 = np.sum(covar_matrix1)
        cov_sum2 = np.sum(covar_matrix2)
        
        #Calculate the variances
        Var1 = 4.0 * n + 2.0 * cov_sum1
        Var2 = 4.0 * n + 2.0 * cov_sum2
        
        #Calculate the Brown's correction parameters
        c1 = Var1 / (2.0 * Expected)
        df_brown1 = (2.0 * Expected ** 2) / Var1
        df_brown1 = min(df_brown1, df_fisher)
        c1 = 1.0 if df_brown1 == df_fisher else c1
        
        c2 = Var2 / (2.0 * Expected)
        df_brown2 = (2.0 * Expected **2) / Var2
        df_brown2 = min(df_brown2, df_fisher)
        c2 = 1.0 if df_brown2 == df_fisher else c2
        
        ##Computation of the final combined p-value, using the chi-square statistic
        chi_squared1 = 2.0 * np.sum(-np.log(p1), axis=1)
        combined_p1 = 1 - chi2.cdf(chi_squared1 / c1, df_brown1)
        chi_squared2 = 2.0 * np.sum(-np.log(p2), axis=1)
        combined_p2 = 1 - chi2.cdf(chi_squared2 / c2, df_brown2)
        
        p_brown_final = 2 * np.minimum(combined_p1, combined_p2)
        p_brown_final = np.clip(p_brown_final, 0.0, 1.0)
        
        browns_p[valid_snps] = p_brown_final
    
    return df_merged.with_columns(
        pl.Series(name="Browns_p", values=browns_p)
    )


#Estimate correlation matrix R, using the z-scores
def estimate_correlation(df_merged, method='global'):

    if method != 'global':
        raise ValueError("Only global method is supported.")

    z_cols = [col for col in df_merged.columns if col.startswith('Z')]
    if not z_cols:
        raise ValueError("No z-score columns found.")

    Z_matrix = df_merged.select(z_cols).to_numpy().astype(np.float32)

    #Keep only the SNPs that have all z-scores valid
    finite_mask = np.isfinite(Z_matrix)
    valid_rows = np.all(finite_mask, axis=1)
    Z_valid = Z_matrix[valid_rows].astype(np.float64)

    k = len(z_cols)
    if len(Z_valid) < 2:
        #Not enough complete-case SNPs to compute correlation matrix R, return an Identity matrix
        return np.eye(k, dtype=np.float32)

    R = np.corrcoef(Z_valid, rowvar=False)
    R = np.nan_to_num(R, nan=0.0)
    np.fill_diagonal(R, 1.0)

    return R.astype(np.float32)


#Help function to compute Delta matrix for Yang method
@jit(nopython=True)
def yang_delta_kernel(R_matrix, N_corr_samples, k):
    
    delta_matrix = np.zeros((k, k), dtype=np.float64)
    c1 = 3.9081

    for i in range(k):
        for j in range(i + 1, k):
            r = R_matrix[i, j]

            #Bias correction
            biased_corrected_r = r * (1.0 + ((1.0 - r ** 2) / (2.0 * (N_corr_samples - 3.0))))
            r2 = biased_corrected_r ** 2

            #Polynomial calculation
            f_r = (3.9081 * r2 + 0.0313 * r2 ** 2 + 0.1022 * r2 ** 3 - 0.1378 * r2 ** 4 + 0.0941 * r2 ** 5)

            #Final bias calculation
            bias = (c1 / k) * (1.0 - r2) ** 2

            #Apply to both sides of the symmetric matrix
            delta = f_r - bias
            delta_matrix[i, j] = delta
            delta_matrix[j, i] = delta

    return delta_matrix


def BrownsMethodbyYang(df_merged, df_for_corr=None):

    z_cols = [col for col in df_merged.columns if col.startswith('Z')]
    if not z_cols:
        raise ValueError("No z-scores found.")

    Z = df_merged.select(z_cols).to_numpy().astype(np.float32)
    n_snps, k = Z.shape

    yang_p = np.full(n_snps, np.nan, dtype=np.float64)

    finite_mask = np.isfinite(Z)
    finite_counts = finite_mask.sum(axis=1)

    #For SNPs with only 1 valid z-score
    single_study_snps = finite_counts == 1
    if np.any(single_study_snps):
        
        Z_single = Z[single_study_snps]
        mask_single = finite_mask[single_study_snps]

        valid_idx = np.argmax(mask_single, axis=1)

        z_vals = Z_single[np.arange(Z_single.shape[0]), valid_idx]
        yang_p_single  = z_vals
        yang_p[single_study_snps] = yang_p_single

    #For SNPs with at least 2 valid SNPs
    valid_snps = finite_counts >= 2
    if np.any(valid_snps):
        
        Z_valid = Z[valid_snps]
        finite_mask_valid = finite_mask[valid_snps]
        
        #Transform to one-sided p-values
        p_matrix = np.full(Z_valid.shape, 1.0, dtype=np.float32)
        p_matrix[finite_mask_valid] = 2.0 * ndtr(-np.abs(Z_valid[finite_mask_valid].astype(np.float32)))
        epsilon = np.finfo(np.float32).tiny
        p_matrix = np.clip(p_matrix, epsilon, 1.0)
        
        p_matrix[~finite_mask_valid] = 1.0
        
        #Estimate global correlation matrix R
        R = estimate_correlation(df_for_corr, method='global')
        print(f"Correlation matrix R:\n {R}")
        
        N_corr_samples = len(df_for_corr) if df_for_corr is not None else n_snps

        #Compute delta matrix
        delta_matrix = yang_delta_kernel(R, float(N_corr_samples), k)
        delta_sum = np.sum(delta_matrix)

        #Create gamma distribution for our correlated data
        mean_T = 2.0 * k
        var_T = 4.0 * k + delta_sum
        if var_T <= 0 or mean_T <= 0:
            var_T = 4.0 * k
        v_gamma = 2 * (mean_T**2 / var_T)
        gamma_value = var_T / mean_T
        
        #Computation of T-statistic
        log_p = -np.log(p_matrix)
        T = 2.0 * np.sum(log_p, axis=1)
        T = np.clip(T, 0, 1e6)
        
        #Computation of the final combined p-value
        yang_p_vals = 1 - gamma.sf(T, a = v_gamma / 2, scale = 2 * gamma_value)
        yang_p_vals = np.clip(yang_p_vals, 0.0, 1.0)
        
        yang_p[valid_snps] = yang_p_vals

    return df_merged.with_columns(pl.Series("Yang_p", yang_p))


#Estimate correlation matrix R, using the betas and standard errors
def estimate_correlation_beta_ses(df_merged, method='global'):
    
    if method != 'global':
        raise ValueError("Only method='global' is supported.")

    beta_cols = sorted([col for col in df_merged.columns if col.startswith("BETA")])
    se_cols = sorted([col for col in df_merged.columns if col.startswith("SE")])

    #Find BETA and SE columns and calculate the Z-scores
    if beta_cols and se_cols:
        
        beta_matrix = df_merged.select(beta_cols).to_numpy().astype(np.float32)
        se_matrix = df_merged.select(se_cols).to_numpy().astype(np.float32)

        z_matrix = np.full_like(beta_matrix, np.nan)
        valid_mask = (np.isfinite(beta_matrix) & np.isfinite(se_matrix) & (se_matrix > 0))
        z_matrix[valid_mask] = beta_matrix[valid_mask] / se_matrix[valid_mask]
        
    #Fallback to existing Z columns
    else:
        
        z_cols = sorted([col for col in df_merged.columns if col.startswith('Z')])
        if not z_cols:
            raise ValueError("No BETA/SE or Z columns found.")
        z_matrix = df_merged.select(z_cols).to_numpy().astype(np.float32)

    k = z_matrix.shape[1]

    #Keep only the SNPs that have all z-scores valid
    valid_snps = np.all(np.isfinite(z_matrix), axis=1)
    z_complete = z_matrix[valid_snps]

    if len(z_complete) < 2:
        return np.eye(k, dtype=np.float32)

    R = np.corrcoef(z_complete, rowvar=False)
    R = np.nan_to_num(R, nan=0.0)
    np.fill_diagonal(R, 1.0)

    return R.astype(np.float32)


#Help Function for Wald Method, handles valid z-scores as missingness patterns across SNPs
def wald_worker(chunk_data):
    
    z_chunk, finite_mask_chunk, R = chunk_data
    n_chunk, k = z_chunk.shape

    #Initialize empty arrays for this chunk
    w_out = np.full(n_chunk, np.nan, dtype=np.float32)
    p_out = np.full(n_chunk, np.nan, dtype=np.float64)

    valid_z = finite_mask_chunk.any(axis=1)
    if not np.any(valid_z):
        return w_out, p_out
    
    #Convert each boolean row into a single unique integer pattern ID instantly
    powers = 1 << np.arange(k)
    pattern_ids = finite_mask_chunk.dot(powers)

    #Find all the unique patterns
    unique_patterns = np.unique(pattern_ids)

    #Process each group once based on its unique pattern integer
    for p_val in unique_patterns:
        if p_val == 0:
            continue

        #Get all SNP indices that share the same missingness pattern
        indices = np.nonzero(pattern_ids == p_val)[0]

        obs_tuple = tuple(np.nonzero(finite_mask_chunk[indices[0]])[0])
        n_obs = len(obs_tuple)

        Z_group = z_chunk[np.ix_(indices, obs_tuple)]

        #For SNPs that have only 1 valid z-score
        if n_obs == 1:
            #Compute the Wald Statistic and the combined p-value
            w = Z_group[:, 0] ** 2
            p_val_array = chi2.sf(w, df=1)
        #For SNPs that have at least 2 valid z-scores
        else:
            #Compute R submatrix and inverted R
            R_sub = R[np.ix_(obs_tuple, obs_tuple)]
            try:
                R_inv = np.linalg.inv(R_sub)
            except np.linalg.LinAlgError:
                R_inv = np.linalg.pinv(R_sub)

            Z_Rinv = Z_group @ R_inv
            #Compute the Wald Statistic and the combined p-value
            w = np.einsum('ij,ij->i', Z_Rinv, Z_group)
            p_val_array = chi2.sf(w, df=n_obs)

        w_out[indices] = w.astype(np.float32)
        p_out[indices] = p_val_array

    return w_out, p_out


#Wald Test, using z-scores
def MultivariateWaldTest_Zscores(df_merged, df_for_corr=None):

    z_cols = [col for col in df_merged.columns if col.startswith('Z')]
    if not z_cols:
        raise ValueError("No z-scores found.")

    Z = df_merged.select(z_cols).to_numpy().astype(np.float32)
    n_snps, k = Z.shape

    wald_stats = np.full(n_snps, np.nan, dtype=np.float64)
    wald_p = np.full(n_snps, np.nan, dtype=np.float64)

    #Estimate global correlation matrix R
    R = estimate_correlation(df_for_corr, method='global')
    R = R.astype(np.float32)
    print(f"Correlation matrix R:\n {R}")

    finite_mask = np.isfinite(Z)
    finite_counts = finite_mask.sum(axis=1)

    #For SNPs with at least 1 valid z-score
    valid_snps = finite_counts >= 1
    if np.any(valid_snps):
        
        Z_valid = Z[valid_snps]

        #Define number of CPU cores for parallel processing
        num_cores = max(3, multiprocessing.cpu_count() - 8)

        #Split the arrays into equals for parallel workers
        z_chunks = np.array_split(Z_valid, num_cores)
        mask_chunks = np.array_split(finite_mask[valid_snps], num_cores)

        #Each worker gets a task (Z_chunk, mask_chunk, correlation matrix R)
        tasks = [(z_chunks[i], mask_chunks[i], R) for i in range(num_cores)]

        all_w_out = []
        all_p_out = []

        #Parallel execution using multiple processes
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
            results = executor.map(wald_worker, tasks)

            for w_chunk, p_chunk in results:
                all_w_out.append(w_chunk)
                all_p_out.append(p_chunk)

        #Concatenate all chunks back into one array
        wald_stats_valid = np.concatenate(all_w_out)
        wald_p_valid = np.concatenate(all_p_out)
        
        wald_stats[valid_snps] = wald_stats_valid
        wald_p[valid_snps] = wald_p_valid

    return df_merged.with_columns([
        pl.Series("Wald_Statistic", wald_stats),
        pl.Series("Wald_p", wald_p)
    ])

#Wald test, using betas and standard errors
def MultivariateWaldTest_BetasSes(df_merged, df_for_corr=None):

    beta_cols = [col for col in df_merged.columns if col.startswith('BETA')]
    se_cols = [col for col in df_merged.columns if col.startswith('SE')]

    if not beta_cols or not se_cols:
        raise ValueError("Both BETA and SE columns are required.")

    #Ensure the columns are sorted, so each BETA column matches to its SE column
    beta_cols.sort()
    se_cols.sort()

    beta_matrix = df_merged.select(beta_cols).to_numpy().astype(np.float32)
    se_matrix = df_merged.select(se_cols).to_numpy().astype(np.float32)

    #Estimate global correlation matrix R
    R = estimate_correlation_beta_ses(df_for_corr, method='global')
    R = R.astype(np.float32)
    print(f"Correlation matrix R:\n {R}")

    #Vectorized Z-score calculation: z = BETA / SE
    z_matrix = np.full_like(beta_matrix, np.nan, dtype=np.float32)
    valid_mask = (np.isfinite(beta_matrix) & np.isfinite(se_matrix) & (se_matrix > 0))

    z_matrix[valid_mask] = beta_matrix[valid_mask] / se_matrix[valid_mask]
    
    finite_mask = np.isfinite(z_matrix)
    finite_counts = finite_mask.sum(axis=1)

    #For SNPs with at least 1 valid z-score
    valid_snps = finite_counts >= 1
    if np.any(valid_snps):

        #Define number of CPU cores for parallel processing
        num_cores = max(3, multiprocessing.cpu_count() - 8)

        #Split the arrays into equals for parallel workers
        z_chunks = np.array_split(z_matrix, num_cores)
        mask_chunks = np.array_split(valid_mask, num_cores)

        #Each worker gets a task (Z_chunk, mask_chunk, correlation matrix R)
        tasks = [(z_chunks[i], mask_chunks[i], R) for i in range(num_cores)]

        all_w_out = []
        all_p_out = []
        
        #Parallel execution using multiple processes
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
            results = executor.map(wald_worker, tasks)

            for w_chunk, p_chunk in results:
                all_w_out.append(w_chunk)
                all_p_out.append(p_chunk)

        #Concatenate all chunks back into one array
        wald_stats = np.concatenate(all_w_out)
        wald_p = np.concatenate(all_p_out)

    return df_merged.with_columns([
        pl.Series("Wald_statistic", wald_stats),
        pl.Series("Wald_p", wald_p)
    ])


#Helper function to transform the z-scores to p-values
def transform_to_pvalues(z_scores, missing_value):
    
    nan_mask = ~np.isfinite(z_scores)
    
    #Convert each z-score to positive and negative direction p-values
    p1 = norm.sf(z_scores)
    p2 = norm.sf(-z_scores)
    
    #Studies that have valid z-scores for each SNP will have a true value
    #Studies that don't have a valid z-score will get a neutral value, for each method, that doesn't affect each method's calculations
    p1 = np.where(nan_mask, missing_value, p1)
    p2 = np.where(nan_mask, missing_value, p2)
    
    p1 = np.clip(p1, 1e-15, 1 - 1e-15)
    p2 = np.clip(p2, 1e-15, 1 - 1e-15)
    
    return p1, p2

if __name__ == '__main__':
    main()
