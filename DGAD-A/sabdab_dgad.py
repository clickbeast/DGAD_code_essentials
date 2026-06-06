from enum import Enum
import os
import random
import logging
import datetime
import pandas as pd
import joblib
import pickle
import lmdb
import subprocess
import torch
from Bio import PDB, SeqRecord, SeqIO, Seq
from Bio.PDB import PDBExceptions
from Bio.PDB import Polypeptide
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from ..utils.protein import parsers, constants
from ._base import register_dataset


import csv

class SplitType(Enum):
        ORIGINAL = 'original'
        BASIC = 'basic'
        HEURISTIC = 'heuristic'
        BASIC_PRESELECT = 'basic_preselect'

class Split(Enum):
        TRAIN = 'train'
        VAL = 'test'
        TEST = 'val'

ALLOWED_AG_TYPES = {
    'protein',
    'protein | protein',
    'protein | protein | protein',
    'protein | protein | protein | protein | protein',
    'protein | protein | protein | protein',
}

RESOLUTION_THRESHOLD = 4.0

TEST_ANTIGENS = [
    'sars-cov-2 receptor binding domain',
    'hiv-1 envelope glycoprotein gp160',
    'mers s',
    'influenza a virus',
    'cd27 antigen',
]


def nan_to_empty_string(val):
    if val != val or not val:
        return ''
    else:
        return val


def nan_to_none(val):
    if val != val or not val:
        return None
    else:
        return val


def split_sabdab_delimited_str(val):
    if not val:
        return []
    else:
        return [s.strip() for s in val.split('|')]


def parse_sabdab_resolution(val):
    if val == 'NOT' or not val or val != val:
        return None
    elif isinstance(val, str) and ',' in val:
        return float(val.split(',')[0].strip())
    else:
        return float(val)


def _aa_tensor_to_sequence(aa):
    return ''.join([Polypeptide.index_to_one(a.item()) for a in aa.flatten()])


def _label_heavy_chain_cdr(data, seq_map, max_cdr3_length=30):
    if data is None or seq_map is None:
        return data, seq_map

    # Add CDR labels
    cdr_flag = torch.zeros_like(data['aa'])
    for position, idx in seq_map.items():
        resseq = position[1]
        cdr_type = constants.ChothiaCDRRange.to_cdr('H', resseq)
        if cdr_type is not None:
            cdr_flag[idx] = cdr_type
    data['cdr_flag'] = cdr_flag

    # Add CDR sequence annotations
    data['H1_seq'] = _aa_tensor_to_sequence( data['aa'][cdr_flag == constants.CDR.H1] )
    data['H2_seq'] = _aa_tensor_to_sequence( data['aa'][cdr_flag == constants.CDR.H2] )
    data['H3_seq'] = _aa_tensor_to_sequence( data['aa'][cdr_flag == constants.CDR.H3] )

    cdr3_length = (cdr_flag == constants.CDR.H3).sum().item()
    # Remove too long CDR3
    if cdr3_length > max_cdr3_length:
        cdr_flag[cdr_flag == constants.CDR.H3] = 0
        logging.warning(f'CDR-H3 too long {cdr3_length}. Removed.')
        return None, None

    # Filter: ensure CDR3 exists
    if cdr3_length == 0:
        logging.warning('No CDR-H3 found in the heavy chain.')
        return None, None

    return data, seq_map


def _label_light_chain_cdr(data, seq_map, max_cdr3_length=30):
    if data is None or seq_map is None:
        return data, seq_map
    cdr_flag = torch.zeros_like(data['aa'])
    for position, idx in seq_map.items():
        resseq = position[1]
        cdr_type = constants.ChothiaCDRRange.to_cdr('L', resseq)
        if cdr_type is not None:
            cdr_flag[idx] = cdr_type
    data['cdr_flag'] = cdr_flag

    data['L1_seq'] = _aa_tensor_to_sequence( data['aa'][cdr_flag == constants.CDR.L1] )
    data['L2_seq'] = _aa_tensor_to_sequence( data['aa'][cdr_flag == constants.CDR.L2] )
    data['L3_seq'] = _aa_tensor_to_sequence( data['aa'][cdr_flag == constants.CDR.L3] )

    cdr3_length = (cdr_flag == constants.CDR.L3).sum().item()
    # Remove too long CDR3
    if cdr3_length > max_cdr3_length:
        cdr_flag[cdr_flag == constants.CDR.L3] = 0
        logging.warning(f'CDR-L3 too long {cdr3_length}. Removed.')
        return None, None

    # Ensure CDR3 exists
    if cdr3_length == 0:
        logging.warning('No CDRs found in the light chain.')
        return None, None

    return data, seq_map


def preprocess_sabdab_structure(task):
    entry = task['entry']
    pdb_path = task['pdb_path']

    parser = PDB.PDBParser(QUIET=True)
    model = parser.get_structure(id, pdb_path)[0]

    parsed = {
        'id': entry['id'],
        'heavy': None,
        'heavy_seqmap': None,
        'light': None,
        'light_seqmap': None,
        'antigen': None,
        'antigen_seqmap': None,
    }
    try:
        if entry['H_chain'] is not None:
            (
                parsed['heavy'], 
                parsed['heavy_seqmap']
            ) = _label_heavy_chain_cdr(*parsers.parse_biopython_structure(
                model[entry['H_chain']],
                max_resseq = 113    # Chothia, end of Heavy chain Fv
            ))
        
        if entry['L_chain'] is not None:
            (
                parsed['light'], 
                parsed['light_seqmap']
            ) = _label_light_chain_cdr(*parsers.parse_biopython_structure(
                model[entry['L_chain']],
                max_resseq = 106    # Chothia, end of Light chain Fv
            ))

        if parsed['heavy'] is None and parsed['light'] is None:
            raise ValueError('Neither valid H-chain or L-chain is found.')
    
        if len(entry['ag_chains']) > 0:
            chains = [model[c] for c in entry['ag_chains']]
            (
                parsed['antigen'], 
                parsed['antigen_seqmap']
            ) = parsers.parse_biopython_structure(chains)

    except (
        PDBExceptions.PDBConstructionException, 
        parsers.ParsingException, 
        KeyError,
        ValueError,
    ) as e:
        logging.warning('[{}] {}: {}'.format(
            task['id'], 
            e.__class__.__name__, 
            str(e)
        ))
        return None

    return parsed



class SAbDabDatasetDgad(Dataset):

    MAP_SIZE = 32*(1024*1024*1024)  # 32GB

    def __init__(
        self, 
        summary_path = './data/sabdab_summary_all.tsv', 
        chothia_dir = './data/all_structures/chothia', 
        processed_dir = './data/processed',
        split = 'train',
        split_seed = 2022,
        split_type = SplitType.ORIGINAL,
        split_ratio = [0.8, 0.1, 0.1],
        transform = None,
        reset = False,
    ):
        super().__init__()
        self.summary_path = summary_path
        self.chothia_dir = chothia_dir
        self.split_type = split_type
        self.split_ratio = split_ratio
        if not os.path.exists(chothia_dir):
            raise FileNotFoundError(
                f"SAbDab structures not found in {chothia_dir}. "
                "Please download them from http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/archive/all/"
            )
        self.processed_dir = processed_dir
        os.makedirs(processed_dir, exist_ok=True)

        self.sabdab_entries = None
        self._load_sabdab_entries()

        self.db_conn = None
        self.db_ids = None
        self._load_structures(reset)

        self.clusters = None
        self.id_to_cluster = None
        self._load_clusters(reset)

        self.ids_in_split = None
        self._load_split(split, split_seed)

        self.transform = transform

    def _load_sabdab_entries(self):
        df = pd.read_csv(self.summary_path, sep='\t')
        entries_all = []
        for i, row in tqdm(
            df.iterrows(), 
            dynamic_ncols=True, 
            desc='Loading entries',
            total=len(df),
        ):
            entry_id = "{pdbcode}_{H}_{L}_{Ag}".format(
                pdbcode = row['pdb'],
                H = nan_to_empty_string(row['Hchain']),
                L = nan_to_empty_string(row['Lchain']),
                Ag = ''.join(split_sabdab_delimited_str(
                    nan_to_empty_string(row['antigen_chain'])
                ))
            )
            ag_chains = split_sabdab_delimited_str(
                nan_to_empty_string(row['antigen_chain'])
            )
            resolution = parse_sabdab_resolution(row['resolution'])
            entry = {
                'id': entry_id,
                'pdbcode': row['pdb'],
                'H_chain': nan_to_none(row['Hchain']),
                'L_chain': nan_to_none(row['Lchain']),
                'ag_chains': ag_chains,
                'ag_type': nan_to_none(row['antigen_type']),
                'ag_name': nan_to_none(row['antigen_name']),
                'date': datetime.datetime.strptime(row['date'], '%m/%d/%y'),
                'resolution': resolution,
                'method': row['method'],
                'scfv': row['scfv'],
            }

            # Filtering
            if (
                (entry['ag_type'] in ALLOWED_AG_TYPES or entry['ag_type'] is None)
                and (entry['resolution'] is not None and entry['resolution'] <= RESOLUTION_THRESHOLD)
            ):
                entries_all.append(entry)
        self.sabdab_entries = entries_all

    def _load_structures(self, reset):
        if not os.path.exists(self._structure_cache_path) or reset:
            if os.path.exists(self._structure_cache_path):
                os.unlink(self._structure_cache_path)
            self._preprocess_structures()

        with open(self._structure_cache_path + '-ids', 'rb') as f:
            self.db_ids = pickle.load(f)
        self.sabdab_entries = list(
            filter(
                lambda e: e['id'] in self.db_ids,
                self.sabdab_entries
            )
        )

    @property
    def _structure_cache_path(self):
        return os.path.join(self.processed_dir, 'structures.lmdb')
        
    def _preprocess_structures(self):
        tasks = []
        for entry in self.sabdab_entries:
            pdb_path = os.path.join(self.chothia_dir, '{}.pdb'.format(entry['pdbcode']))
            if not os.path.exists(pdb_path):
                logging.warning(f"PDB not found: {pdb_path}")
                continue
            tasks.append({
                'id': entry['id'],
                'entry': entry,
                'pdb_path': pdb_path,
            })

        data_list = joblib.Parallel(
            n_jobs = max(joblib.cpu_count() // 2, 1),
        )(
            joblib.delayed(preprocess_sabdab_structure)(task)
            for task in tqdm(tasks, dynamic_ncols=True, desc='Preprocess')
        )

        db_conn = lmdb.open(
            self._structure_cache_path,
            map_size = self.MAP_SIZE,
            create=True,
            subdir=False,
            readonly=False,
        )
        ids = []
        with db_conn.begin(write=True, buffers=True) as txn:
            for data in tqdm(data_list, dynamic_ncols=True, desc='Write to LMDB'):
                if data is None:
                    continue
                ids.append(data['id'])
                txn.put(data['id'].encode('utf-8'), pickle.dumps(data))

        with open(self._structure_cache_path + '-ids', 'wb') as f:
            pickle.dump(ids, f)

    @property
    def _cluster_path(self):
        return os.path.join(self.processed_dir, 'cluster_result_cluster.tsv')

    def _load_clusters(self, reset):
        if not os.path.exists(self._cluster_path) or reset:
            self._create_clusters()

        clusters, id_to_cluster = {}, {}
        with open(self._cluster_path, 'r') as f:
            for line in f.readlines():
                cluster_name, data_id = line.split()
                if cluster_name not in clusters:
                    clusters[cluster_name] = []
                clusters[cluster_name].append(data_id)
                id_to_cluster[data_id] = cluster_name
        self.clusters = clusters
        self.id_to_cluster = id_to_cluster

    def _create_clusters(self):
        cdr_records = []
        for id in self.db_ids:
            structure = self.get_structure(id)
            if structure['heavy'] is not None:
                cdr_records.append(SeqRecord.SeqRecord(
                    Seq.Seq(structure['heavy']['H3_seq']),
                    id = structure['id'],
                    name = '',
                    description = '',
                ))
            elif structure['light'] is not None:
                cdr_records.append(SeqRecord.SeqRecord(
                    Seq.Seq(structure['light']['L3_seq']),
                    id = structure['id'],
                    name = '',
                    description = '',
                ))
        fasta_path = os.path.join(self.processed_dir, 'cdr_sequences.fasta')
        SeqIO.write(cdr_records, fasta_path, 'fasta')

        cmd = ' '.join([
            'mmseqs', 'easy-cluster',
            os.path.realpath(fasta_path),
            'cluster_result', 'cluster_tmp',
            '--min-seq-id', '0.5',
            '-c', '0.8',
            '--cov-mode', '1',
        ])
        subprocess.run(cmd, cwd=self.processed_dir, shell=True, check=True)
    

    def _group_and_sort_clusters_by_size(self, clusters, size_range=None):
        # Group clusters by their size
        size_to_clusters = {}
        for cluster_id, items in clusters.items():
            size = len(items)
            if size not in size_to_clusters:
                size_to_clusters[size] = {}
            size_to_clusters[size][cluster_id] = items

        # Sort the groups by size (descending)
        clusters_grouped_and_sorted = [
            (size, size_to_clusters[size])
            for size in sorted(size_to_clusters.keys(), reverse=True)
        ]

        #Filter out clusters that do not match the size range
        if size_range is not None:
            clusters_grouped_and_sorted_filtered = [
                (size, clusters) for size, clusters in clusters_grouped_and_sorted
                if size_range[0] <= size <= size_range[1]
            ]
            return clusters_grouped_and_sorted_filtered
        

        return clusters_grouped_and_sorted


    def _get_cluster_ids_with_size_range(self, clusters, size_range):
        """
        Get cluster IDs that fall within a specified size range.
        """
        filtered_clusters = {}
        for cluster_id, items in clusters.items():
            size = len(items)
            if size_range[0] <= size <= size_range[1]:
                filtered_clusters[cluster_id] = items
        return filtered_clusters
    

    def _load_split(self, split, split_seed):
        assert split in ('train', 'val', 'test')
        if self.split_type  == SplitType.ORIGINAL:
            self._split_original(split, split_seed)
        elif self.split_type == SplitType.BASIC:
            self._split_basic(split, split_seed)
        elif self.split_type == SplitType.HEURISTIC:
            self._split_heuristic(split, split_seed)
        elif self.split_type == SplitType.BASIC_PRESELECT:
            self._split_basic_preselect(split, split_seed)

    def _split_original(self, split, split_seed):

        ids_test = [
            entry['id']
            for entry in self.sabdab_entries
            if entry['ag_name'] in TEST_ANTIGENS
        ]
        test_relevant_clusters = set([self.id_to_cluster[id] for id in ids_test])

        ids_train_val = [
            entry['id']
            for entry in self.sabdab_entries
            if self.id_to_cluster[entry['id']] not in test_relevant_clusters
        ]
        random.Random(split_seed).shuffle(ids_train_val)

        #Print the ids that are not in ids_test and 
        ids_remain = [
            entry['id']
            for entry in self.sabdab_entries
            if (self.id_to_cluster[entry['id']] in test_relevant_clusters) and (entry['id'] not in ids_test)
        ]
        print('remaining ids')
        print(len(ids_remain))
        print(len(ids_test))

        if split == 'test':
            self.ids_in_split = ids_test
        elif split == 'val':
            self.ids_in_split = ids_train_val[:20]
        else:
            self.ids_in_split = ids_train_val[20:]
        pass

    def _split_basic(self, split, split_seed):
        # Filter out the cluster ids to be in the test
        ids_test = [
            entry['id']
            for entry in self.sabdab_entries
            if entry['ag_name'] in TEST_ANTIGENS
        ]
        test_relevant_clusters = set([self.id_to_cluster[id] for id in ids_test])

        # Get all clusters that are NOT in the test set (based on antigen names)
        available_clusters = [
            cluster_id for cluster_id in self.clusters.keys() 
            if cluster_id not in test_relevant_clusters
        ]
        
        # Randomize the available clusters according to split_seed
        random.Random(split_seed).shuffle(available_clusters)
        
        # Calculate split indices based on ratio
        train_ratio, val_ratio, test_ratio = self.split_ratio
        total_clusters = len(available_clusters)
        
        train_end = int(total_clusters * train_ratio)
        val_end = train_end + int(total_clusters * val_ratio)
        test_end = val_end + int(total_clusters * test_ratio)
        
        # Split available clusters into train/val/additional_test
        train_clusters = available_clusters[:train_end]
        val_clusters = available_clusters[train_end:val_end]
        additional_test_clusters = available_clusters[val_end:test_end]
        
        # Any leftover clusters go to train
        leftover_clusters = available_clusters[test_end:]
        train_clusters.extend(leftover_clusters)
        
        # Test clusters include BOTH predefined test antigens AND additional clusters from ratio
        test_clusters = list(test_relevant_clusters) + additional_test_clusters
        
        # Select the appropriate cluster set based on split
        if split == 'train':
            selected_clusters = train_clusters
        elif split == 'val':
            selected_clusters = val_clusters
        elif split == 'test':
            selected_clusters = test_clusters
        
        # Concatenate all ids from the selected clusters
        self.ids_in_split = []
        for cluster_id in selected_clusters:
            self.ids_in_split.extend(self.clusters[cluster_id])

        # Shuffle the final list of IDs to randomize order within the split
        random.Random(split_seed).shuffle(self.ids_in_split)


        pass

    def _split_basic_preselect(self, split, split_seed):

        # Filter out the cluster ids to be in the test
        ids_test = [
            entry['id']
            for entry in self.sabdab_entries
            if entry['ag_name'] in TEST_ANTIGENS
        ]
        test_relevant_clusters = set([self.id_to_cluster[id] for id in ids_test])

        # Get all clusters that are NOT in the test set (based on antigen names)
        available_clusters = [
            cluster_id for cluster_id in self.clusters.keys() 
            if cluster_id not in test_relevant_clusters
        ]

        #Filter out big clusters, we will do a preselection manually here for test and val. The remaining go to train.
        leave_out_cluster_with_size_range = [18, 169]
        test_cluster_sizes = [105,78, 29, 18]
        val_cluster_sizes = [40, 20]
        
        preselected_train_clusters = []
        preselected_val_clusters = []
        preselected_test_clusters = []  

        
        preselected_cluster_groups = self._group_and_sort_clusters_by_size(
            self.clusters, 
            leave_out_cluster_with_size_range
        )
        
        for (size, clusters) in preselected_cluster_groups:
            if size in test_cluster_sizes:
                cluster_ids = clusters.keys()
                #Pick one random id from cluster_ids
                random.Random(split_seed).shuffle(list(cluster_ids))
                preselected_test_clusters.append(list(cluster_ids)[0])
                #Add the remaining ids to the train clusters
                if len(cluster_ids) > 1:
                    preselected_train_clusters.extend(list(cluster_ids)[1:])
            elif size in val_cluster_sizes:
                cluster_ids = clusters.keys()
                #Pick one random id from cluster_ids
                random.Random(split_seed).shuffle(list(cluster_ids))
                preselected_val_clusters.append(list(cluster_ids)[0])
                #Add the remaining ids to the train clusters
                if len(cluster_ids) > 1:
                    preselected_train_clusters.extend(list(cluster_ids)[1:])
            else:

                preselected_train_clusters.extend(clusters.keys())
        

        # Filter out the preselected clusters from the available clusters
        available_clusters = [
            cluster_id for cluster_id in available_clusters 
            if cluster_id not in preselected_train_clusters and 
               cluster_id not in preselected_val_clusters and 
               cluster_id not in preselected_test_clusters
        ]

        # Randomize the available clusters according to split_seed
        random.Random(split_seed).shuffle(available_clusters)
        
        # Calculate split indices based on ratio
        train_ratio, val_ratio, test_ratio = self.split_ratio
        total_clusters = len(available_clusters)
        
        train_end = int(total_clusters * train_ratio)
        val_end = train_end + int(total_clusters * val_ratio)
        test_end = val_end + int(total_clusters * test_ratio)
        
        # Split available clusters into train/val/additional_test
        train_clusters = available_clusters[:train_end] + preselected_train_clusters
        val_clusters = available_clusters[train_end:val_end] + preselected_val_clusters
        additional_test_clusters = available_clusters[val_end:test_end] + preselected_test_clusters
        
        # Any leftover clusters go to train
        leftover_clusters = available_clusters[test_end:]
        train_clusters.extend(leftover_clusters)
        
        # Test clusters include BOTH predefined test antigens AND additional clusters from ratio
        test_clusters = list(test_relevant_clusters) + additional_test_clusters
        
        # Select the appropriate cluster set based on split
        if split == 'train':
            selected_clusters = train_clusters
        elif split == 'val':
            selected_clusters = val_clusters
        elif split == 'test':
            selected_clusters = test_clusters
        
        # Concatenate all ids from the selected clusters
        self.ids_in_split = []
        for cluster_id in selected_clusters:
            self.ids_in_split.extend(self.clusters[cluster_id])

        # Shuffle the final list of IDs to randomize order within the split
        random.Random(split_seed).shuffle(self.ids_in_split)


        pass

    def _split_heuristic(self, split, split_seed):
        
        #Todo maybe change heuristic so larger groups are more likely to be in the train set.    

        # Filter out the cluster ids to be in the test
        ids_test = [
            entry['id']
            for entry in self.sabdab_entries
            if entry['ag_name'] in TEST_ANTIGENS
        ]
        test_relevant_clusters = set([self.id_to_cluster[id] for id in ids_test])

        # Get all clusters that are NOT in the test set (based on antigen names)
        available_cluster_ids = [
            cluster_id for cluster_id in self.clusters.keys() 
            if cluster_id not in test_relevant_clusters
        ]
        
        available_clusters = {}
        for cluster_id in available_cluster_ids:
            available_clusters[cluster_id] = self.clusters[cluster_id]


        clusters_grouped_and_sorted = self._group_and_sort_clusters_by_size(available_clusters)

        self.ids_in_split = []

        for cluster_group in clusters_grouped_and_sorted:
            size, clusters_in_group  = cluster_group

            print(f'handling cluster group {size}')
            

            # Convert dictionary to list of cluster IDs and randomize
            cluster_ids_in_group = list(clusters_in_group.keys())
            random.Random(split_seed).shuffle(cluster_ids_in_group)
            
            # Calculate split indices based on ratio
            train_ratio, val_ratio, test_ratio = self.split_ratio
            total_clusters = len(cluster_ids_in_group)
            
            train_end = int(total_clusters * train_ratio)
            val_end = train_end + int(total_clusters * val_ratio)
            test_end = val_end + int(total_clusters * test_ratio)
            
            # Split available clusters into train/val/additional_test
            print(train_end)
            train_clusters = cluster_ids_in_group[:train_end]
            val_clusters = cluster_ids_in_group[train_end:val_end]
            additional_test_clusters = cluster_ids_in_group[val_end:test_end]
            
            # Any leftover clusters go to train
            leftover_clusters = cluster_ids_in_group[test_end:]
            train_clusters.extend(leftover_clusters)
            
            # Test clusters include BOTH predefined test antigens AND additional clusters from ratio
            test_clusters = additional_test_clusters
            
            # Select the appropriate cluster set based on split
            if split == 'train':
                selected_clusters = train_clusters
            elif split == 'val':
                selected_clusters = val_clusters
            elif split == 'test':
                selected_clusters = test_clusters
            
            # Concatenate all ids from the selected clusters
            ids_in_split_group = []
            for cluster_id in selected_clusters:
                ids_in_split_group.extend(self.clusters[cluster_id])
            
            #Concatenate across groups
            self.ids_in_split.extend(ids_in_split_group)
            pass

        
        if split == 'test':
            for cluster_id in test_relevant_clusters:
                self.ids_in_split.extend(self.clusters[cluster_id])

        # Shuffle the final list of IDs to randomize order within the split
        random.Random(split_seed).shuffle(self.ids_in_split)

            
        pass

    def _connect_db(self):
        if self.db_conn is not None:
            return
        self.db_conn = lmdb.open(
            self._structure_cache_path,
            map_size=self.MAP_SIZE,
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

    def get_structure(self, id):
        self._connect_db()
        with self.db_conn.begin() as txn:
            return pickle.loads(txn.get(id.encode()))

    def __len__(self):
        return len(self.ids_in_split)

    def __getitem__(self, index):
        id = self.ids_in_split[index]
        data = self.get_structure(id)
        if self.transform is not None:
            data = self.transform(data)
        return data
    


@register_dataset('sabdab_dgad')
def get_sabdab_dataset(cfg, transform):
    return SAbDabDatasetDgad(
        summary_path = cfg.summary_path,
        chothia_dir = cfg.chothia_dir,
        processed_dir = cfg.processed_dir,
        split_seed = 70,
        split = cfg.split,
        split_type = SplitType.BASIC_PRESELECT,
        split_ratio = [0.9, 0.05, 0.05],
        transform = transform,
    )

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--processed_dir', type=str, default='./data/processed')
    parser.add_argument('--reset', action='store_true', default=False)
    args = parser.parse_args()
    if args.reset:
        sure = input('Sure to reset? (y/n): ')
        if sure != 'y':
            exit()

    seeds = list(range(71,101))
    seeds = [70]

    for seed_name in seeds:
        print("\n" + "-"*50)
        print(f'generating dataset for seed: {seed_name}')
        for  split_name in ['train', 'test', 'val']:
            split=split_name 
            split_seed=seed_name
            split_type = SplitType.BASIC_PRESELECT
            split_ratio=[0.9, 0.05, 0.05]


            split_ratio_train = split_ratio[0]
            split_ratio_val =  split_ratio[1]
            split_ratio_test = split_ratio[2]


            dataset = SAbDabDatasetDgad(
                processed_dir=args.processed_dir,
                split_seed=split_seed,
                split=split,
                split_type = split_type,
                split_ratio=split_ratio,
                reset=args.reset
            )


        #print(dataset[0])
            #print(len(dataset), len(dataset.clusters))
            base_dir = '/Users/simonvermeir/Documents/School/Burgerlijk-Ingenieur/Master-Thesis/DGAD_Alpha/DGAD-A/datasets_lists'
            # create csv out of
            # Open a CSV file and write the header
            file_name = f'ids_dgad_{split_type.value}_{split}_{str(split_ratio)}_{split_seed}.csv'
            with open(os.path.join(base_dir, file_name), 'w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(['id', 'cluster_id', 'split_type', 'split', 'split_ratio', 'split_ratio_train', 'split_ratio_val', 'split_ratio_test', 'split_seed'])

                # Iterate over each element in the dataset and write the id
                
                if False:
                    for element in tqdm(dataset, desc='Writing split'):
                        writer.writerow([element['id'], dataset.id_to_cluster[element['id']], split_type.value , split, split_ratio])
                    

                # Iterate over each id
                if True:
                    for pdb_id in tqdm(dataset.ids_in_split, desc='Writing split'):
                        writer.writerow([pdb_id, dataset.id_to_cluster[pdb_id], split_type.value , split, split_ratio, split_ratio_train, split_ratio_val, split_ratio_test, split_seed])
                        pass