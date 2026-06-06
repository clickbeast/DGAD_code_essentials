import os
import random
import pandas as pd
import lmdb
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate


import wandb
import random
import wandb
import datetime
import copy
import lmdb
import pickle



#
# UTILS
#

def check_sample_uniqueness(item1, item2, verbose=True):
    """
    Check if two samples have unique characteristics by comparing positions where generate flag is true.
    
    Args:
        item1, item2: Dataset items with 'generate_flag' and 'pos_heavy_atom' keys
        verbose: Whether to print detailed differences
        
    Returns:
        bool: True if samples are unique (have different positions), False otherwise
    """
    import torch
    
    # Get generate flags and positions
    gen_flag1 = item1['generate_flag']  # (264,)
    gen_flag2 = item2['generate_flag']  # (264,)
    pos1 = item1['pos_heavyatom']      # (264, 15, 3)
    pos2 = item2['pos_heavyatom']      # (264, 15, 3)
    
    # Check if generate flags are the same
    if not torch.equal(gen_flag1, gen_flag2):
        if verbose:
            print("Generate flags differ between samples")
        return True
    
    # Get positions where generate flag is true
    gen_positions1 = pos1[gen_flag1]  # (N_gen, 15, 3)
    gen_positions2 = pos2[gen_flag2]  # (N_gen, 15, 3)
    
    if gen_positions1.shape != gen_positions2.shape:
        if verbose:
            print(f"Different number of generated positions: {gen_positions1.shape[0]} vs {gen_positions2.shape[0]}")
        return True
    
    # Compare positions with tolerance for floating point errors
    tolerance = 1e-6
    position_diff = torch.abs(gen_positions1 - gen_positions2)
    different_positions = position_diff > tolerance
    
    # Find which sequence positions and atoms differ
    seq_indices = torch.where(gen_flag1)[0]  # Original sequence indices where generate=True
    different_seq_pos = torch.any(different_positions.view(different_positions.shape[0], -1), dim=1)
    different_seq_indices = seq_indices[different_seq_pos]
    
    if verbose:
        if different_seq_indices.numel() > 0:
            print(f"Samples are unique! Differences found at {different_seq_indices.numel()} sequence positions:")
            for i, seq_pos in enumerate(different_seq_indices[:10]):  # Show first 10
                # Find which atoms differ at this position
                rel_idx = torch.where(different_seq_pos)[0][i]
                atom_diffs = torch.any(different_positions[rel_idx], dim=1)
                different_atoms = torch.where(atom_diffs)[0]
                print(f"  Seq pos {seq_pos.item()}: atoms {different_atoms.tolist()} have different positions")
                if i == 9 and len(different_seq_indices) > 10:
                    print(f"  ... and {len(different_seq_indices) - 10} more positions")
        else:
            print("Samples appear identical (positions match within tolerance)")
    
    return different_seq_indices.numel() > 0


#
#  COLLATION (uses defualt pytorch one but uses class to allow for custumization later if needed)
#

class AuxCollate(object):
    def __call__(self, batch):
        return default_collate(batch)

#
# LOADING AUXILIARY DATASETS
#


class AuxDataset(Dataset):
    """
    Handles loading and accessing synthetic auxiliary datasets.
    """
    
    def __init__(self, dataset_dir, lmdb_path=None, ids_path=None):
        """
        Initialize the auxiliary dataset loader.
        
        Args:
            dataset_dir: Directory containing the dataset.
            lmdb_path: Path to the LMDB file (optional).
            ids_path: Path to the CSV file containing IDs (optional).
        """
        
        self.MAP_SIZE = 256 * 1024 ** 3

        self.dataset_dir = dataset_dir
        self.lmdb_path = lmdb_path or os.path.join(dataset_dir, 'processed', 'data.lmdb')
        self.ids_path = ids_path or os.path.join(dataset_dir, 'ids.csv')
        


        if not os.path.exists(self.lmdb_path):
            raise FileNotFoundError(f"LMDB file not found at {self.lmdb_path}")
        
        if not os.path.exists(self.ids_path):
            raise FileNotFoundError(f"IDs file not found at {self.ids_path}")
        
        # Load IDs
        (ids, cluster_ids) = self._load_ids()

        self.ids_in_split = ids
        self.cluster_ids_in_split = cluster_ids
        
        self.id_to_cluster = {id: cluster_id for id, cluster_id in zip(ids, cluster_ids)}

        #Set up LMDB
        self.db_conn = None
        self.read_txn = None
        self.cursor = None

        
    
    def _load_ids(self):
        print(f"Loading IDs from {self.ids_path}")
        ids = []
        cluster_ids = []

        with open(self.ids_path, 'r') as file:
            # Skip the header row
            header = file.readline().strip()
            print(f"Header: {header}")
            
            # Read each line
            for line in file:
                line = line.strip()
                if line:  # Skip empty lines
                    columns = line.split(',')
                    if len(columns) == 2:
                        ids.append(columns[0])
                        cluster_ids.append(columns[1])
                    else:
                        raise ValueError('id csv contains more than two columns')
            print(f"Loaded {len(ids)} IDs from {self.ids_path}")

        return ids, cluster_ids

    def _connect_db(self):
        if self.db_conn is not None:
            return
        self.db_conn = lmdb.open(
            self.lmdb_path,
            map_size=self.MAP_SIZE,
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.read_txn = self.db_conn.begin(write=False)
        self.cursor = self.read_txn.cursor()

    def get_structure(self, id, sample_idx=0):
        self._connect_db()
        if self.cursor.set_key(id.encode()):
            return pickle.loads(self.cursor.value())[sample_idx]
        else:
            raise KeyError(f"ID {id} not found")

    def get_item_by_idx_and_sample_idx(self, idx, sample_idx):
        return self.get_structure(self.ids_in_split[idx], sample_idx)

    def __len__(self):
        return len(self.ids_in_split)

    def __getitem__(self, index):
        id = self.ids_in_split[index]
        data = self.get_structure(id)
        return data

#
# GENERAL
#
  

if __name__ == '__main__':
    
    # Example usage
    dataset_dir = './data_dgad/dgad_aux_dataset'
    
    aux_dataset = AuxDataset(dataset_dir=dataset_dir)
    
    #grab a single item to test
    import time
    
    # First access - should be slower (loading from disk)
    start_time = time.time()
    item = aux_dataset[0]
    first_access_time = time.time() - start_time
    print(f"First access time: {first_access_time:.6f} seconds")

    # Second access - should be faster (cached in memory)
    start_time = time.time()
    item = aux_dataset[0]
    second_access_time = time.time() - start_time
    print(f"Second access time: {second_access_time:.6f} seconds")
    
    speedup = first_access_time / second_access_time if second_access_time > 0 else float('inf')
    print(f"Speedup factor: {speedup:.2f}x")


    item1 = item
    item2 = aux_dataset.get_item_by_idx_and_sample_idx(0, 1)

    # Check if the different samples are unique by comparing positions where generate flag is true
    print("\nChecking sample uniqueness:")
    are_unique = check_sample_uniqueness(item1, item2, verbose=True)
    print(f"Samples are unique: {are_unique}")

    #Use dataloader to batch items for testing
    print(f"Number of entries in the dataset: {len(aux_dataset)}")
    aux_data_loader = DataLoader(
        aux_dataset, 
        batch_size=4, 
        collate_fn=AuxCollate(), 
        shuffle=False,
        num_workers=0
    )

    for batch in aux_data_loader:
        print(batch)
        break 

    pass


