from tqdm import tqdm
import os
import string
import pickle
import sys
import pandas as pd
import multiprocessing as mp
from itertools import permutations, combinations
import random
import glob
import numpy as np
import csv

from protein_embedding import find_sequence, ProteinEmbedding

import psico.fullinit
from pymol import cmd


def init(dev_num, c_type, region):
    global chain_type, prot_region
    chain_type = c_type
    prot_region = region

    global data_path, id_pairs_path
    data_path = '/data/cb/rsingh/work/antibody/ci_data/raw/sabdab_pure_042522/sabdab_dataset'
    id_pairs_path = '/data/cb/rsingh/work/antibody/ci_data/processed/sabdab_pure_042522/all_pairs_set1.p'


def work_parallel(idx_pair):
    idx, pdb_id1, pdb_id2 = idx_pair

    cmd.delete('all')

    p1_path = os.path.join(data_path, pdb_id1, 'structure', '{}.pdb'.format(pdb_id1))
    p2_path = os.path.join(data_path, pdb_id2, 'structure', '{}.pdb'.format(pdb_id2))

    cmd.load(p1_path, "999C")
    cmd.load(p2_path, "999D")

    cmd.split_chains('999C')
    cmd.split_chains('999D')


    p1_seq_H, p1_seq_L = find_sequence(dataset = 'sabdab_pure', pdb_id=pdb_id1)
    p2_seq_H, p2_seq_L = find_sequence(dataset = 'sabdab_pure', pdb_id=pdb_id2)

    if chain_type == 'H':
        p1_seq, p2_seq = p1_seq_H, p2_seq_H
    else:
        p1_seq, p2_seq = p1_seq_L, p2_seq_L

    fold = prot_region if chain_type == 'H' else str(5 + int(prot_region))
    start1, end1, start2, end2 = 0, 0, 0, 0
    if prot_region != 'whole':
        try:
            p1 = ProteinEmbedding(p1_seq, chain_type=chain_type, dev=idx, fold=fold)
            p1.create_cdr_mask()
            p2 = ProteinEmbedding(p2_seq, chain_type=chain_type, dev=idx, fold=fold)
            p2.create_cdr_mask()
        except:
            return (idx, -1, -1)

        p1_mask = p1.cdr_mask.tolist()
        a = list(map(int, p1_mask))
        t = list(map(str, a))
        p1_mask_str = "".join(t)

        p2_mask = p2.cdr_mask.tolist()
        a = list(map(int, p2_mask))
        t = list(map(str, a))
        p2_mask_str = "".join(t)

        if prot_region not in p1_mask_str or prot_region not in p2_mask_str:
            return (idx, -1, -1)

        start1, end1 = p1_mask_str.index(prot_region)+1, p1_mask_str.rindex(prot_region)+1
        start2, end2 = p2_mask_str.index(prot_region)+1, p2_mask_str.rindex(prot_region)+1

        del p1, p2, p1_mask, p2_mask, p1_mask_str, p2_mask_str # prevents multiprocessing from slowing down!

        # try:
        #     start1, end1 = p1_mask_str.index(prot_region)+1, p1_mask_str.rindex(prot_region)+1
        # except:
        #     print("p1 mask:", p1_mask_str)
        #     raise ValueError
        # try:
        #     start2, end2 = p2_mask_str.index(prot_region)+1, p2_mask_str.rindex(prot_region)+1
        # except:
        #     print("p2 mask:", p2_mask_str)
        #     raise ValueError

        # print("Indices for Chain {} Region {}, Against {} :".format(chain_type, prot_region, against), start1, end1, start2, end2)


    # find the chain letters
    chain_letters = []
    for pdb_id in [pdb_id1, pdb_id2]:
        h = glob.glob(os.path.join(data_path, pdb_id, 'sequence', '{}_*_V{}.fa'.format(pdb_id, chain_type)))
        h1 = [a for a in h if "{}_V{}.fa".format(chain_type, chain_type) in a]
        if len(h1)> 0:
           vhfile = h1[0]
        else:
           vhfile = h[0]
        chain_letter = vhfile[-7]
        chain_letters.append(chain_letter)

    prot1 = '999C_{}'.format(chain_letters[0])
    prot2 = '999D_{}'.format(chain_letters[1])


    if prot_region == 'whole':
        try:
            rmsd = cmd.super(prot1+'////CA', prot2+'////CA')[0]
        except:
            print("{},{}-{} RMSD for region {} didn't work...".format(pdb_id1, pdb_id2, chain_type, prot_region))
            rmsd = -1
        try:
            tm_score = cmd.tmalign(prot1+'////CA', prot2+'////CA')
        except:
            print("{},{}-{} TM Score for region {} didn't work...".format(pdb_id1, pdb_id2, chain_type, prot_region))
            tm_score = -1

    else:
        try:
            rmsd = cmd.super(prot1+'///{}-{}/CA'.format(start1, end1), prot2+'///{}-{}/CA'.format(start2, end2))[0]
        except:
            print("{},{}-{} RMSD for region {} didn't work...".format(pdb_id1, pdb_id2, chain_type, prot_region))
            rmsd = -1
        try:
            tm_score = cmd.tmalign(prot1+'///{}-{}/CA'.format(start1, end1), prot2+'///{}-{}/CA'.format(start2, end2))
        except:
            print("{},{}-{} TM Score for region {} didn't work...".format(pdb_id1, pdb_id2, chain_type, prot_region))
            tm_score = -1

    return (idx, tm_score, rmsd)


def generate_id_pairs(args):
    valid_ids_path = '/data/cb/rsingh/work/antibody/ci_data/processed/sabdab_pure_042522/valid_ids_set2.txt'

    # saving perms
    with open(valid_ids_path, 'r') as f:
        valid_ids = f.read().splitlines()
    perms = list(combinations(valid_ids, 2))
    random.shuffle(perms)
    
    with open('/data/cb/rsingh/work/antibody/ci_data/processed/sabdab_pure_042522/all_pairs_set2.p', 'wb') as f:
        pickle.dump(perms, f)

    print('DONE!')
    return



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="which code path to run. see main(..) for details", type=str, choices = ["1_generate_id_pairs", "2_calculate_pair_scores"])
    parser.add_argument("--device_num", help="GPU device for computation.", type=int, default=0)
    parser.add_argument("--num_processes", help="number of processes for multiprocessing", type=int, default=1)
    parser.add_argument("--chunk_size", help="chunk size for each processing step", type=int, default=1)
    parser.add_argument("--region", help="region of the protein sequence that we're processing", type=str)
    parser.add_argument("--chain_type", help="Which chain to process: H or L", type=str, choices = ["H", "L"])
    parser.add_argument("--extra", help="put this as the LAST option and arbitrary space-separated key=val pairs after that", type=str, nargs='*')

    args = parser.parse_args()
    assert args.mode is not None
    
    extra_args = dict([a.split("=") for a in args.extra]) if args.extra else {}
    args.extra = extra_args
    
    pd.set_option('use_inf_as_na', True)

    
    if args.mode == '1_generate_id_pairs':
        generate_id_pairs(args)
    elif args.mode == '2_calculate_pair_scores':
        sample_n = 110000

        # load the pairs:
        all_pairs_path = '/data/cb/rsingh/work/antibody/ci_data/processed/sabdab_pure_042522/all_pairs_set1.p'
        with open(all_pairs_path, 'rb') as f:
            all_pairs = pickle.load(f)
        n_pairs = all_pairs[:sample_n]
        n_pairs = [(i, id1, id2) for i, (id1, id2) in enumerate(n_pairs)]

        # perform multiprocessing
        import multiprocessing as mp
        from multiprocessing import get_context

        tm_scores = set()
        with get_context('spawn').Pool(processes=args.num_processes, initializer=init, initargs=[args.device_num, args.chain_type, args.region]) as p:
            for result in tqdm(p.imap_unordered(work_parallel, n_pairs, chunksize=args.chunk_size), total=len(n_pairs)):
                tm_scores.add(result)

                if len(tm_scores)%10000 == 0:
                    with open('../results/CDR{}{}_pairs_{}.p'.format(args.chain_type, args.region, len(tm_scores)), 'wb') as f:
                        pickle.dump(tm_scores, f)
                # tm_scores.append(result)

        tm_scores = list(tm_scores)
        tm_scores.sort(key=lambda x: x[0])

        # save the results into a csv file
        # save_count, num_pairs = 0, 100000
        with open('../results/set1_CDR{}_{}_pair_scores_100k.csv'.format(args.region, args.chain_type), 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(['pdb_id1', 'pdb_id2', 'tm_score', 'rmsd'])
            for i in range(len(tm_scores)):
                if tm_scores[i][1] > 0 and tm_scores[i][2] >= 0:
                    writer.writerow([n_pairs[i][1], n_pairs[i][2], tm_scores[i][1], tm_scores[i][2]])
                    # save_count += 1

                    # if save_count == num_pairs:
                    #     break

    else:
        assert False