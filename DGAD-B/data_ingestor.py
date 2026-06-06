
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
from tqdm import tqdm
from easydict import EasyDict




class Experiment:
    def __init__(self, name, experiment_path=None, uid=None, df=None, timestamp=None):
        self.name = name
        self.uid = uid if uid is not None else name
        self.experiment_path = experiment_path
        self.df = df
        self.timestamp = timestamp


    def get_as_experiments_row(self):
        """
        Returns a dictionary representation of the experiment suitable for appending to an experiments catalog DataFrame.
        """
        return {
            'uid': self.uid,
            'name': self.name,
            'experiment_path': self.experiment_path,
            'timestamp': self.timestamp
        }


    def __repr__(self):
        return (f"Experiment(name={self.name}, "
                f"uid={self.uid}, "
                f"experiment_path={self.experiment_path}, "
                f"timestamp={self.timestamp}, "
                f"df={'set' if self.df is not None else 'None'})")




# ----------------------------------------------------------------------------
# UTILS
# ----------------------------------------------------------------------------

def setup_and_parse_args():

    #default name is just a timestamp
    name = datetime.now().strftime("ingest__%Y_%m_%d__%H_%M_%S")
    result_dir_name = 'agg_results_inference_12'
    result_dir = os.path.join('/simonvermeir/DGAD/models/diffab-main-edit-1/results/', result_dir_name)
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=name)
    parser.add_argument('--result_dir', type=str, default=result_dir, help="Path to the results directory containing the diffab samples.")

    args = parser.parse_args()
    return args




# ----------------------------------------------------------------------------
# DATA INGENSTION
# ----------------------------------------------------------------------------


class CDRRegion(Enum):
    H_CDR3 = "H_CDR3"
    L_CDR3 = "L_CDR3"
    H_CDR2 = "H_CDR2"
    L_CDR2 = "L_CDR2"
    H_CDR1 = "H_CDR1"
    L_CDR1 = "L_CDR1"


CDR_FOLDERS = [r.value for r in CDRRegion]

FOLDER_REGEX = re.compile(
    r"^(?P<idx>\d{4})_(?P<pdb_id>\w+?)_(?P<heavy_id>\w+?)_(?P<light_id>\w*?)_(?P<antigen_id>\w+?)_(?P<timestamp>\d{4}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2})$"
)


def parse_generated_diffab_samples(result_dir, experiment_id):

    def parse_folder_name(folder):
        m = FOLDER_REGEX.match(folder)
        if not m:
            return None
        d = m.groupdict()
        
        #sanitize
        if d['light_id'] == '':
            d['light_id'] = None
        return d

    def parse_timestamp(ts):
        return datetime.strptime(ts, "%Y_%m_%d__%H_%M_%S")

    def generate_diffab_uid(data):
        """
        Generate a unique identifier for a diffab sample.
        """
        uid = f"{experiment_id}___{data['pdb_id']}___{data['heavy_id']}_{data['light_id']}_{data['antigen_id']}___{data['sample_id']}_{data.get('sample_step', 'NA')}_{data['cdr_target']}___{data['timestamp']}"
        return uid

    #Step find subdir name that starts with 'codesign_single', this is one of the dir within result_dir
    codesign_single_dir = None
    for folder in os.listdir(result_dir):
        if folder.startswith('codesign_single'):
            codesign_single_dir = folder
            break


    result_dir = os.path.join(result_dir, codesign_single_dir)

    # Step 1: Find all folders and parse names
    all_folders = []
    for folder in os.listdir(result_dir):
        folder_path = os.path.join(result_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        parsed = parse_folder_name(folder)
        if parsed:
            parsed['folder'] = folder
            all_folders.append(parsed)

    # Step 2: Deduplicate by pdb_id, heavy_id, light_id, antigen_id, keep latest timestamp
    key_to_folders = defaultdict(list)
    for d in all_folders:
        key = (d['pdb_id'], d['heavy_id'], d['light_id'], d['antigen_id'])
        key_to_folders[key].append(d)

    deduped_structure_dirs = []
    duplicate_count = 0
    for key, folders in key_to_folders.items():
        if len(folders) > 1:
            # Print all folder names for this key
            for f in folders:
                print(f"{f['folder']}")
            # Print summary line for this group
            base_name = f"{folders[0]['idx']}_{folders[0]['pdb_id']}_{folders[0]['heavy_id']}_{folders[0]['light_id'] or ''}_{folders[0]['antigen_id']}"
            print(f"{len(folders)} folders found for : {base_name}")
            duplicate_count += len(folders) - 1
        # Pick latest timestamp
        folders.sort(key=lambda x: parse_timestamp(x['timestamp']), reverse=True)
        deduped_structure_dirs.append(folders[0])
    if duplicate_count > 0:
        print(f"{duplicate_count} duplicate folder(s) found.")


    # Step 3: For each folder, collect structure_data and sample_data
    sample_columns = [
        'experiment_id', 
        'pdb_id', 
        'heavy_id',
        'light_id', 
        'antigen_id', 
        'sample_id', 
        'sample_step', 
        'cdr_target', 
        'timestamp',
        'reference_pdb_rpath', 
        'metadata_rpath',
        'generated_pdb_rpath', 
        'result_dir'
    ]
    sample_dict = {col: [] for col in sample_columns}

    for d in deduped_structure_dirs:
        folder = d['folder']
        folder_path = os.path.join(result_dir, folder)
        # Check reference.pdb
        reference_pdb = os.path.join(folder_path, "reference.pdb")
        if not os.path.isfile(reference_pdb):
            print(f"Skipping folder (no reference.pdb): {folder}")
            continue
        reference_pdb_rpath = os.path.relpath(reference_pdb, result_dir)
        # Check metadata.json
        metadata_json = os.path.join(folder_path, "metadata.json")
        if not os.path.isfile(metadata_json):
            print(f"Skipping folder (no metadata.json): {folder}")
            continue
        metadata_rpath = os.path.relpath(metadata_json, result_dir)

        structure_data = {
            'pdb_id': d['pdb_id'],
            'heavy_id': d['heavy_id'],
            'light_id': d['light_id'],
            'antigen_id': d['antigen_id'],
            'timestamp': d['timestamp'],
            'reference_pdb_rpath': reference_pdb_rpath,
            'metadata_rpath': metadata_rpath,
            'result_dir': result_dir,
        }

        # Check for CDR folders
        found_any_cdr = False
        for cdr_folder in CDR_FOLDERS:
            cdr_path = os.path.join(folder_path, cdr_folder)
            if not os.path.isdir(cdr_path):
                continue
            pdb_files = sorted([f for f in os.listdir(cdr_path) if f.endswith('.pdb')])
            if not pdb_files:
                continue
            found_any_cdr = True
            for pdb_file in pdb_files:
                base_name = os.path.splitext(pdb_file)[0]
                # Only process files with integer names (e.g., 0000.pdb)
                if not base_name.isdigit():
                    continue
                sample_id = int(base_name)
                generated_pdb_rpath = os.path.relpath(os.path.join(cdr_path, pdb_file), result_dir)
                # Collect sample data
                for col in structure_data:
                    sample_dict[col].append(structure_data[col])
                sample_dict['experiment_id'].append(experiment_id)
                sample_dict['generated_pdb_rpath'].append(generated_pdb_rpath)
                sample_dict['sample_id'].append(sample_id)
                sample_dict['sample_step'].append(100)
                sample_dict['cdr_target'].append(cdr_folder)
        if not found_any_cdr:
            # No CDR folders with samples, skip
            continue

    # Step 4: Create DataFrame
    df = pd.DataFrame(sample_dict)
    print(f"Collected {len(df)} samples.")
    print('bonjour')

    # Step 5: Generate uid for each row
    
    df["uid"] = df.apply(generate_diffab_uid, axis=1)

    print('bonjour')


    return df



class DataIngestor():
    
    def __init__(self):

        pass

    def process_results(self, result_dir, name=None, timestamp=None):
        """
        Process the results directory and return an Experiment object with all samples.
        """
        if not os.path.isdir(result_dir):
            raise ValueError(f"Result directory {result_dir} does not exist.")

        if name is None:
            name = os.path.basename(result_dir)
        df = parse_generated_diffab_samples(result_dir, name)
        
        
        # Optionally set timestamp if available
        experiment = Experiment(
            name=name,
            uid=name,
            df=df,
            timestamp=timestamp
        )
        return experiment

    def get_timestamp(self):
        #format: YYYY_MonthMonth_DayDay__HH_MinuteMinute_SecondSecond
        return datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    

    def process_aggregated_results(self, aggregated_results_dir):
        """
        Process the aggregated results directory and return an Experiment object with all samples.
        """
        if not os.path.isdir(aggregated_results_dir):
            raise ValueError(f"Aggregated results directory {aggregated_results_dir} does not exist.")

        # Collect all subdirectories
        all_subdirs = [os.path.join(aggregated_results_dir, d) for d in os.listdir(aggregated_results_dir) if os.path.isdir(os.path.join(aggregated_results_dir, d))]

        #Check if all subdirs have unique base names otherwise raise an error
        unique_bases = set()
        for subdir in all_subdirs:
            base_name = os.path.basename(subdir)
            if base_name in unique_bases:
                raise ValueError(f"Duplicate subdirectory found: {subdir}")
            unique_bases.add(base_name)

        # Process each subdirectory
        all_experiments= []
        for subdir in all_subdirs:

            #clean up subdir name
            name = os.path.basename(subdir)
            # if name.startswith('results_') remove this part
            if name.startswith('results_'):
                name = name[len('results_'):]
            # if name.startswith('diffab_') remove this part
            if name.startswith('diffab_'):
                name = name[len('diffab_'):]
            if name.startswith('inference_'):
                name = name[len('inference_'):]

            experiment = self.process_results(subdir, name=name, timestamp=self.get_timestamp())
            all_experiments.append(experiment)
        return all_experiments

class ExperimentManager():

    def __init__(self, data_dir='/simonvermeir/DGAD/DGAD_B/data'):
        self.data_dir = data_dir
        self.experiments_register_path = os.path.join(data_dir, 'experiments_register.csv')
        self.experiments_dir = os.path.join(data_dir, 'experiments')


        self.presets_path = os.path.join(data_dir, 'presets.json')
        presets = None
        with open(self.presets_path, 'r') as f:
            presets = json.load(f)
        self.presets = EasyDict(presets)
    

    def save_group(self, name, experiment_uids):
        
        group = {
            'uid': name,
            'experiment_uids': experiment_uids
        }


        self.presets.groups.append(group)
        # Update group_uid_to_idx mapping
        self.presets.group_uid_to_idx[name] = len(self.presets.groups) - 1

        # Save back to presets.json
        with open(self.presets_path, 'w') as f:
            json.dump(self.presets, f, indent=4)
        
        # reload the presets in the class
        with open(self.presets_path, 'r') as f:
            presets = json.load(f)
        self.presets = EasyDict(presets)

        print(f"Group '{name}' saved with {len(experiment_uids)} experiments.")

    #
    # INITIAL ADDING OF EXPERIMENTS
    #


    def add_experiment(self, experiment: Experiment, on_conflict='ask'):
        # Step 1: Read the current csv with list of current experiments
        experiments_catalog_df = pd.read_csv(self.experiments_register_path)
        experiment_path = os.path.join(self.experiments_dir, f"{experiment.name}.csv")
        exists = experiment.name in experiments_catalog_df['name'].values if 'name' in experiments_catalog_df.columns else False
        if exists:
            if on_conflict == 'ask':
                print(f"Experiment '{experiment.name}' already exists.")
                print("Options: 1. continue with another name, 2. overwrite, 3. skip")
                print("Defaulting to skip. Pass on_conflict='overwrite' or 'continue' to change behavior.")
                return
            elif on_conflict == 'overwrite':
                print(f"Overwriting experiment '{experiment.name}'.")
                # Remove old entry from experiments.csv
                experiments_catalog_df = experiments_catalog_df[experiments_catalog_df['name'] != experiment.name]
                # Overwrite the data csv
                experiment.df.to_csv(experiment_path, index=False)
            elif on_conflict == 'continue':
                new_name = experiment.name + "_new"
                print(f"Continuing with new name: {new_name}")
                experiment.name = new_name
                experiment.uid = new_name
                experiment_path = os.path.join(self.experiments_dir, f"{new_name}.csv")
                experiment.df.to_csv(experiment_path, index=False)
            elif on_conflict == 'skip':
                print(f"Skipping experiment '{experiment.name}'.")
                return None
        else:
            experiment.df.to_csv(experiment_path, index=False)

        # Ensure new row matches existing columns
        next_row = experiment.get_as_experiments_row()
        next_row['experiment_rpath'] = os.path.relpath(experiment_path, self.data_dir)
        next_row['dgad_b_data_dir'] = self.data_dir
        
        # Only keep keys that are in the existing DataFrame columns
        next_row = {col: next_row.get(col, "") for col in experiments_catalog_df.columns}
        next_row_df = pd.DataFrame([next_row], columns=experiments_catalog_df.columns)
        experiments_catalog_df = pd.concat([experiments_catalog_df, next_row_df], ignore_index=True)


        experiments_catalog_df.to_csv(self.experiments_register_path, index=False)

        return experiment

    def add_aggregated_experiments(self, name, experiments: list[Experiment], on_conflict='ask'):
        
        saved_uids = []
        for experiment in experiments:
            added_experiment = self.add_experiment(experiment, on_conflict=on_conflict)
            if added_experiment is None:
                continue
            else:
                print(f"Added experiment: {added_experiment.name} with uid: {added_experiment.uid}")
                saved_uids.append(added_experiment.uid)

        # Step 2: Add to group
        self.save_group(name, saved_uids)
    
    def get_experiments(self, name=None, group=True):
        dfs = []
        uids = self.presets.groups[self.presets.group_uid_to_idx[name]]['experiment_uids']

        for uid in uids:
            experiment_path = os.path.join(self.experiments_dir, f"{uid}.csv")
            if not os.path.exists(experiment_path):
                print(f"Experiment file {experiment_path} does not exist, skipping.")
                continue

            df = pd.read_csv(experiment_path)
            
            # Load sequences if available
            sequences_path = os.path.join(self.experiments_dir, f"{uid}_sequences.json")
            if os.path.exists(sequences_path):
                with open(sequences_path, 'r') as f:
                    sequences = json.load(f)
                df['sequences'] = sequences

            dfs.append(df)

        return dfs

    def get_experiment(self, uid):
        experiment_path = os.path.join(self.experiments_dir, f"{uid}.csv")
        if not os.path.exists(experiment_path):
            raise FileNotFoundError(f"Experiment file {experiment_path} does not exist, skipping.")

        df = pd.read_csv(experiment_path)

        # Load sequences if available
        sequences_path = os.path.join(self.experiments_dir, f"{uid}_sequences.json")
        if os.path.exists(sequences_path):
            with open(sequences_path, 'r') as f:
                sequences = json.load(f)
            df['sequences'] = sequences

            
        return df

    def save_experiment(self, df, name=None):
        """
        Save the DataFrame as an experiment.
        If name is None, use experiment_id from the DataFrame.
        """
        print("Saving experiment...")
        if name is None:
            # Save the df
            
            #Remove the sequences column and save it seperatily
            if 'sequences' in df.columns:
                sequences = df['sequences']
                df = df.drop(columns=['sequences'])
                # Save sequences to a separate file
                sequences_path = os.path.join(self.experiments_dir, f"{df['experiment_id'].iloc[0]}_sequences.json")
                with open(sequences_path, 'w') as f:
                    json.dump(sequences.tolist(), f)

            df.to_csv(os.path.join(self.experiments_dir, f"{df['experiment_id'].iloc[0]}.csv"), index=False)
        else:
            raise ValueError("Saving with a specific name is not implemented yet. Please don't use this feature.")

if __name__ == "__main__":
    args = setup_and_parse_args()
    data_ingestor = DataIngestor()
    em = ExperimentManager()

    
    experiments = data_ingestor.process_aggregated_results(args.result_dir)
    em.add_aggregated_experiments(args.name, experiments, on_conflict='ask')


    print(f"Added {len(experiments)} experiments to the register.")
    #Rosetta calulcations
        
    