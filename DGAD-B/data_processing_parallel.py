
import os
import re
import json
from enum import Enum
import pandas as pd
from collections import defaultdict
from datetime import datetime
from eval_designs import *
from tqdm import tqdm
from dataclasses import dataclass

import argparse


from eval_designs import *
from sequence_extraction import extract_sequences
from data_ingestor import ExperimentManager

from tqdm import tqdm
import pandas as pd

import concurrent.futures





def calculate_rmsd_for_df(df):
        def calculate_rmsd_for_row(row):
            """Calculate RMSD for a single row of the dataframe"""
            try:
                # Create full paths
                gen_pdb_path = os.path.join(row['result_dir'], row['generated_pdb_rpath'])
                ref_pdb_path = os.path.join(row['result_dir'], row['reference_pdb_rpath'])
                metadata_path = os.path.join(row['result_dir'], row['metadata_rpath'])
                cdr_target = row['cdr_target']
                
                # Get biopython models
                design_gen = get_gen_biopython_model(gen_pdb_path)
                design_ref = get_ref_biopython_model(ref_pdb_path)
                
                # Extract CDR residue range
                residue_first, residue_last = extract_CDR_first_last_residue(metadata_path, cdr_target)
                
                # Extract residue lists
                reslist_gen = extract_reslist(design_gen, residue_first, residue_last)
                reslist_ref = extract_reslist(design_ref, residue_first, residue_last)
                
                # Calculate RMSD
                rmsd = reslist_rmsd(reslist_gen, reslist_ref)
                
                return rmsd
            except Exception as e:
                print(f"Error calculating RMSD for row: {e}")
                return None

        # Add RMSD column to dataframe
        print("Calculating RMSD for all samples...")
        df['rmsd'] = [calculate_rmsd_for_row(row) for row in tqdm(df.to_dict('records'), desc="Calculating RMSD")]

        print(f"RMSD calculation complete. {df['rmsd'].notna().sum()} out of {len(df)} samples calculated successfully.")

        return df


def _extract_sequences_for_row(row: dict):
    """Top-level worker to extract sequences for a single dataframe row (picklable)."""
    reference_pdb_path = os.path.join(row['result_dir'], row['reference_pdb_rpath'])
    sample_id = f"{row['pdb_id']}_{row['heavy_id']}_{row['light_id']}_{row['antigen_id']}"
    reference_sequences = extract_sequences(
        reference_pdb_path,
        heavy_chain_id=row['heavy_id'],
        light_chain_id=row['light_id'],
        antigen_chain_id=row['antigen_id'],
        id=sample_id,
    )
    generated_pdb_path = os.path.join(row['result_dir'], row['generated_pdb_rpath'])
    generated_sequences = extract_sequences(
        generated_pdb_path,
        heavy_chain_id=row['heavy_id'],
        light_chain_id=row['light_id'],
        antigen_chain_id=row['antigen_id'],
        id=sample_id,
    )

    return {
        'reference': reference_sequences,
        'generated': generated_sequences,
    }


def extract_sequences_for_df(df, max_workers=None, use_processes=True):
    """Populate df['sequences'] using parallel execution.

    Args:
        df: pandas DataFrame containing at least result_dir, *_rpath and chain ids.
        max_workers: number of workers. Defaults to all CPUs for processes, or 4x CPUs for threads.
        use_processes: when True, use ProcessPoolExecutor (good for CPU-bound); else threads (good for I/O-bound).
    """
    # Decide default worker count based on execution model
    if max_workers is None:
        cpu_cnt = os.cpu_count() or 1
        max_workers = cpu_cnt if use_processes else min(32, cpu_cnt * 4)

    executor_name = 'processes' if use_processes else 'threads'
    print(f"Extracting sequences for all samples ({executor_name}) with {max_workers} workers...")

    records = df.to_dict('records')
    if not records:
        df['sequences'] = []
        return df

    Executor = concurrent.futures.ProcessPoolExecutor if use_processes else concurrent.futures.ThreadPoolExecutor
    # Choose a reasonable chunksize to reduce scheduling overhead
    chunksize = max(1, len(records) // (max_workers * 4) if max_workers else 1)
    with Executor(max_workers=max_workers) as executor:
        sequences_iter = executor.map(_extract_sequences_for_row, records, chunksize=chunksize)
        sequences_list = list(tqdm(sequences_iter, total=len(records), desc="Extracting sequences"))
    df['sequences'] = sequences_list

    return df
    

def calculate_aar_for_df(df):
    def calculate_aar_for_row(row):
        #average amino acid residue (AAR) calculation , average amino acid recovery
        seq_ref = row['sequences']['reference']['cdrs'][row['sequences']['reference']['cdr_id_to_cdr_idx']['H_CDR3']]['sequence']
        seq_gen = row['sequences']['generated']['cdrs'][row['sequences']['generated']['cdr_id_to_cdr_idx']['H_CDR3']]['sequence']
        
        ar = 0
        aar = 0
        if len(seq_ref) != len(seq_gen):
            print(f"Sequences not same for length for row {row['id']}, setting to 0")
            ar = 0
            aar = 0
        else:
            #calculate average amino acid recovery
            ar =  sum(1 for a, b in zip(seq_ref, seq_gen) if a == b)
            aar = ar / len(seq_ref)

        return pd.Series({'ar': ar, 'aar': aar})
    
    df[['ar', 'aar']] = df.apply(calculate_aar_for_row, axis=1)
    
    return df



if __name__ == "__main__":
    em = ExperimentManager()
    dfs: list[pd.DataFrame] = em.get_experiments(name='ingest__2025_09_07__18_58_45') 
    i = 0
    for df in dfs:
        print("="*50)
        print('Processing experiment:', df['experiment_id'].iloc[0])
        df = calculate_rmsd_for_df(df)
        # Use process-based parallelism to utilize all CPU cores
        df = extract_sequences_for_df(df, max_workers=8, use_processes=True)
        df = calculate_aar_for_df(df)
        
        em.save_experiment(df)
        #i+=1
        #if i > 0:
        #    break  # For testing, process only the first experiment