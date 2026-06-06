import pandas as pd
from Bio import PDB
from Bio.PDB import PDBParser, Selection
import numpy as np
import json

def get_gen_biopython_model(path):
    parser = PDB.PDBParser(QUIET=False)
    return parser.get_structure(path, path)[0]

def get_ref_biopython_model(path):
    parser = PDB.PDBParser(QUIET=False)
    return parser.get_structure(path, path)[0]


def extract_reslist(model, residue_first, residue_last):
    #check if they are both part of the same chain
    assert residue_first[0] == residue_last[0]
    residue_first, residue_last = tuple(residue_first), tuple(residue_last)

    chain_id = residue_first[0]

    pos_first, pos_last = residue_first[1:], residue_last[1:]
    chain = model[chain_id]

    reslist = []



    #Give me all residues one by one
    for res in Selection.unfold_entities(chain, 'R'):

        pos_current = (res.id[1], res.id[2])
        if pos_first <= pos_current <= pos_last:
            reslist.append(res)


    return reslist



def reslist_rmsd(res_list1, res_list2):
    res_short, res_long = (res_list1, res_list2) if len(res_list1) < len(res_list2) else (res_list2, res_list1)
    M, N = len(res_short), len(res_long)

    def d(i, j):
        coord_i = np.array(res_short[i]['CA'].get_coord())
        coord_j = np.array(res_long[j]['CA'].get_coord())
        return ((coord_i - coord_j) ** 2).sum()

    SD = np.full([M, N], np.inf)
    for i in range(M):
        j = N - (M - i)
        SD[i, j] = sum([d(i + k, j + k) for k in range(N - j)])

    for j in range(N):
        SD[M - 1, j] = d(M - 1, j)

    for i in range(M - 2, -1, -1):
        for j in range((N - (M - i)) - 1, -1, -1):
            SD[i, j] = min(
                d(i, j) + SD[i + 1, j + 1],
                SD[i, j + 1]
            )

    min_SD = SD[0, :N - M + 1].min()
    best_RMSD = np.sqrt(min_SD / M)
    return best_RMSD


def extract_CDR_first_last_residue(metadata_path, cdr_target):
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    for item in metadata['items']:
        if item['tag'] == cdr_target:
            residue_first = item['residue_first']
            residue_last = item['residue_last']
            return residue_first, residue_last

if __name__ == '__main__':

    #Collect all the paths
    #open the run csv
    RUN_PATH = '/Users/simonvermeir/Documents/School/Burgerlijk-Ingenieur/Master-Thesis/GAD_Inference_Experiments/data_processing/run_0.csv'
    OUT_PATH = '/Users/simonvermeir/Documents/School/Burgerlijk-Ingenieur/Master-Thesis/GAD_Inference_Experiments/data_processing/run_0_0.csv'
    run_df = pd.read_csv(RUN_PATH, index_col=0)
    rmsd_samples = []
    for index, row in run_df.iterrows():
        run_id = row['run_id']
        pdb_id = row['pdb_id']
        heavy_id = row['heavy_id']
        light_id = row['light_id']
        antigen_id = row['antigen_id']
        sample_id = row['sample_id']
        timestamp = row['timestamp']
        cdr_target = row['cdr_target']
        gen_pdb_path = row['gen_pdb_path']
        ref_pdb_path = row['ref_pdb_path']
        metadata_path = row['metadata_path']

        model_gen = get_gen_biopython_model(gen_pdb_path)
        model_ref = get_ref_biopython_model(ref_pdb_path)

        residue_first, residue_last = extract_CDR_first_last_residue(metadata_path, cdr_target)

        #figure out the residue first and last
        reslist_gen = extract_reslist(model_gen, residue_first, residue_last)
        reslist_ref = extract_reslist(model_ref, residue_first, residue_last)

        rmsd = reslist_rmsd(reslist_gen, reslist_ref)
        rmsd_samples.append(rmsd)
        print(f'RMSD: {rmsd}')

    run_df['rmsd'] = rmsd_samples
    run_df.to_csv(OUT_PATH, index=True)