
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

import torch

from collections import defaultdict
import numpy as np
import torch




model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t36_3B_UR50D")
batch_converter = alphabet.get_batch_converter()
model.eval()
model = model.cuda()  # or .cuda() if using GPU




def calculate_ll_for_df(df, origin='generated', batch_size=16):
    """
    Calculates the log-likelihood for each sequence in a DataFrame using batching
    and adds it as a new column. It ignores 'X' characters in the sequence.

    Args:
        df (pd.DataFrame): DataFrame containing a 'sequences' column.
        origin (str): The origin of the sequences to use (e.g., 'generated', 'reference').
        batch_size (int): The number of sequences to process in a single batch.
    """
    log_likelihoods = [None] * len(df)
    
    print(f"Calculating log-likelihood for {len(df)} sequences with origin '{origin}'...")
    
    # Process in batches
    for i in tqdm(range(0, len(df), batch_size)):
        batch_df = df.iloc[i:i+batch_size]
        
        batch_data = []
        original_indices = []

        for idx, row in batch_df.iterrows():
            sequences = row['sequences']
            try:
                chain_idx = sequences[origin]['chain_type_to_chain_idx']['heavy']
                heavy_chain_sequence = sequences[origin]['chains'][chain_idx]['sequence']
                batch_data.append((f"seq_{idx}", heavy_chain_sequence))
                original_indices.append(idx)
            except (KeyError, TypeError):
                # This will leave the None value at the original index
                continue
        
        if not batch_data:
            continue

        batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
        batch_tokens = batch_tokens.cuda()

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[], return_contacts=False)
            token_probs = torch.log_softmax(results["logits"], dim=-1)

            for j in range(len(batch_strs)):
                sequence_len = len(batch_strs[j])
                seq_tokens = batch_tokens[j, 1:sequence_len + 1]
                
                # --- CHANGE: Create a mask to ignore 'X' (unknown) tokens ---
                mask = seq_tokens != alphabet.unk_idx
                
                # Apply the mask to get tokens and positions for valid amino acids
                valid_tokens = seq_tokens[mask]
                valid_indices = torch.arange(1, sequence_len + 1, device=batch_tokens.device)[mask]
                
                # Sum log-likelihood only for the valid tokens
                ll = token_probs[j, valid_indices, valid_tokens].sum().item()
                
                # Place the result in the correct position using original index
                original_df_index = df.index.get_loc(original_indices[j])
                log_likelihoods[original_df_index] = ll

    # Add the results to the DataFrame
    col_name = 'll' if origin == 'generated' else f"{origin}_ll"
    df[col_name] = log_likelihoods
    
    return df



def calculate_pll_for_df(df, origin='generated', batch_size=4):
    """
    Calculates the pseudo-log-likelihood for the CDRH3 region of each sequence 
    in a DataFrame using batching and adds it as a new column. It ignores 'X' characters.

    Args:
        df (pd.DataFrame): DataFrame containing a 'sequences' column.
        origin (str): The origin of the sequences to use ('generated' or 'reference').
        batch_size (int): The number of DataFrame rows to process in each batch. A smaller
                      batch size is recommended due to the high number of masked
                      variants generated per row.
    """
    pseudo_log_likelihoods = [None] * len(df)
    
    print(f"Calculating pseudo-log-likelihood for CDRH3 of {len(df)} sequences with origin '{origin}'...")

    for i in tqdm(range(0, len(df), batch_size)):
        batch_df = df.iloc[i:i+batch_size]
        
        masked_sequences_for_batch = []
        # Stores info to map results back
        result_mapping_info = []
        
        for df_idx, row in batch_df.iterrows():
            try:
                sequences = row['sequences']
                chain_idx = sequences[origin]['chain_type_to_chain_idx']['heavy']
                heavy_chain_sequence = sequences[origin]['chains'][chain_idx]['sequence']
                
                cdrh3_idx = sequences[origin]['cdr_id_to_cdr_idx']["H_CDR3"]
                H_CDR3_start_idx, H_CDR3_end_idx = tuple(sequences[origin]['cdrs'][cdrh3_idx]['seq_idx_range'])
                H_CDR3_end_idx += 1 # Make range inclusive

                for pos in range(H_CDR3_start_idx, H_CDR3_end_idx):
                    # --- CHANGE: Skip positions with 'X' ---
                    if heavy_chain_sequence[pos] == 'X':
                        continue
                        
                    masked_sequence = list(heavy_chain_sequence)
                    masked_sequence[pos] = '<mask>'
                    masked_sequences_for_batch.append((f"seq_{df_idx}_pos_{pos}", "".join(masked_sequence)))
                    result_mapping_info.append({
                        "df_idx": df_idx,
                        "original_sequence": heavy_chain_sequence,
                        "masked_pos": pos
                    })
            except (KeyError, TypeError):
                continue

        if not masked_sequences_for_batch:
            continue
            
        # Batch process all masked variants for this group of rows
        batch_labels, batch_strs, batch_tokens = batch_converter(masked_sequences_for_batch)
        batch_tokens = batch_tokens.cuda()

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[], return_contacts=False)
            token_probs = torch.log_softmax(results["logits"], dim=-1)

        # Aggregate PLLs for each row in the batch
        batch_pll_aggregator = defaultdict(float)
        
        for j, info in enumerate(result_mapping_info):
            masked_pos = info['masked_pos']
            original_char = info['original_sequence'][masked_pos]
            
            # --- CHANGE: Skip if original character is 'X' (double check) ---
            if original_char == 'X':
                continue

            original_token_idx = alphabet.get_idx(original_char)
            
            # +1 to account for the BOS token
            pll_at_pos = token_probs[j, masked_pos + 1, original_token_idx].item()
            batch_pll_aggregator[info['df_idx']] += pll_at_pos
            
        # Update the main results list
        for df_idx, total_pll in batch_pll_aggregator.items():
            original_df_position = df.index.get_loc(df_idx)
            pseudo_log_likelihoods[original_df_position] = total_pll
        
        
    
    # Add the results to the DataFrame
    col_name = 'pll' if origin == 'generated' else f"{origin}_pll"
    df[col_name] = pseudo_log_likelihoods
    
    return df


def calculate_ll_perplexity(df, origin='generated'):
    """
    Calculates perplexity from the log-likelihood column.
    Assumes 'll' or 'reference_ll' column already exists.

    Args:
        df (pd.DataFrame): DataFrame with a log-likelihood column.
        origin (str): The origin of the sequences ('generated' or 'reference').
    """
    ll_col_name = 'll' if origin == 'generated' else f"{origin}_ll"
    perplexity_col_name = f"{ll_col_name}_perplexity"
    
    if ll_col_name not in df.columns:
        print(f"Error: Column '{ll_col_name}' not found. Please run 'calculate_ll_for_df' first.")
        return df

    perplexities = []
    for _, row in df.iterrows():
        log_likelihood = row.get(ll_col_name)
        
        if pd.isna(log_likelihood):
            perplexities.append(np.nan)
            continue
            
        try:
            sequences = row['sequences']
            chain_idx = sequences[origin]['chain_type_to_chain_idx']['heavy']
            heavy_chain_sequence = sequences[origin]['chains'][chain_idx]['sequence']
            # Perplexity should be normalized by the number of residues that were scored
            sequence_length = len(heavy_chain_sequence.replace('X', ''))
            
            if sequence_length == 0:
                 perplexities.append(np.nan)
                 continue

            perplexity = torch.exp(torch.tensor(-log_likelihood / sequence_length)).item()
            perplexities.append(perplexity)
        except (KeyError, TypeError):
            perplexities.append(np.nan)

    df[perplexity_col_name] = perplexities
    return df

def calculate_pll_perplexity(df, origin='generated'):
    """
    Calculates perplexity from the pseudo-log-likelihood column for the CDRH3 region.
    Assumes 'pll' or 'reference_pll' column already exists.

    Args:
        df (pd.DataFrame): DataFrame with a pseudo-log-likelihood column.
        origin (str): The origin of the sequences ('generated' or 'reference').
    """

    print('binjour')
    pll_col_name = 'pll' if origin == 'generated' else f"{origin}_pll"
    perplexity_col_name = f"{pll_col_name}_perplexity"

    if pll_col_name not in df.columns:
        print(f"Error: Column '{pll_col_name}' not found. Please run 'calculate_pll_for_df' first.")
        return df

    perplexities = []
    for _, row in df.iterrows():
        #print(row)
        pseudo_log_likelihood = row.get(pll_col_name)

        if pd.isna(pseudo_log_likelihood):
            perplexities.append(np.nan)
            continue

        try:
            #print('YAS')
            sequences = row['sequences']
            cdrh3_idx = sequences[origin]['cdr_id_to_cdr_idx']["H_CDR3"]
            seq_range = sequences[origin]['cdrs'][cdrh3_idx]['seq_idx_range']
            #print(seq_range)
            # Also need to account for 'X's in the region length
            heavy_chain_sequence = sequences[origin]['chains'][sequences[origin]['chain_type_to_chain_idx']['heavy']]['sequence']
            cdrh3_sequence = heavy_chain_sequence[seq_range[0]:seq_range[1]+1]
            #print(heavy_chain_sequence, cdrh3_sequence)
            region_length = len(cdrh3_sequence.replace('X', ''))

            if region_length == 0:
                perplexities.append(np.nan)
                continue

            perplexity = torch.exp(torch.tensor(-pseudo_log_likelihood / region_length)).item()
            #print(perplexity)
            perplexities.append(perplexity)
        except (KeyError, TypeError) as e:
            # This will print the specific error, e.g., "KeyError: 'H_CDR3'"
            #print(f"Could not process row due to an error: {e}. Appending NaN.")
            perplexities.append(np.nan)
            
    df[perplexity_col_name] = perplexities


    return df



if __name__ == "__main__":
    em = ExperimentManager()
    dfs: list[pd.DataFrame] = em.get_experiments(name='ingest__2025_09_07__18_58_45') 
    i = 0
    for df in dfs:
        print("="*50)
        print('Processing experiment:', df['experiment_id'].iloc[0])
        
        
        df = calculate_ll_for_df(df, origin='generated', batch_size=256)
        #df = calculate_ll_for_df(df, origin='reference', batch_size=64)
        df = calculate_pll_for_df(df, origin='generated', batch_size=80)
        #df = calculate_pll_for_df(df, origin='reference', batch_size=32)


        df = calculate_ll_perplexity(df, origin='generated')
        df = calculate_pll_perplexity(df, origin='generated')
        
        em.save_experiment(df)
        #i+=1
        #if i > 0:
        #    break  # For testing, process only the first experiment