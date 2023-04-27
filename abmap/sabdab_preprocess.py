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

from abmap.abmap_augment import ProteinEmbedding
from abmap.utils import find_sequence
from abmap.plm_embed import reload_models_to_device

import psico.fullinit
from pymol import cmd

def init(dev_num, c_type, region, which_set):
    global chain_type, prot_region
    chain_type = c_type
    prot_region = region

    global data_path, id_pairs_path
    data_path = '../data/raw/sabdab/sabdab_dataset'
    id_pairs_path = '../data/processed/sabdab/all_pairs_{}.p'.format(which_set)


def work_parallel(idx_pair):
    idx, pdb_id1, pdb_id2 = idx_pair

    cmd.delete('all')

    p1_path = os.path.join(data_path, pdb_id1, 'structure', '{}.pdb'.format(pdb_id1))
    p2_path = os.path.join(data_path, pdb_id2, 'structure', '{}.pdb'.format(pdb_id2))

    cmd.load(p1_path, "999C")
    cmd.load(p2_path, "999D")

    cmd.split_chains('999C')
    cmd.split_chains('999D')


    p1_seq_H, p1_seq_L = find_sequence(pdb_id=pdb_id1)
    p2_seq_H, p2_seq_L = find_sequence(pdb_id=pdb_id2)

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


def get_valid_ids():
    with open('../data/processed/sabdab/valid_ids_all.txt', 'w') as f:
        valid_list = []
        path = '../data/raw/sabdab/sabdab_dataset'
        pdb_ids = os.listdir(path)
        for pdb_id in tqdm(pdb_ids):
            try:
                vhs, vls = find_sequence(pdb_id = pdb_id)
                prot_embed_h = ProteinEmbedding(vhs, 'H')
                prot_embed_l = ProteinEmbedding(vls, 'L')
                prot_embed_h.create_cdr_mask()
                prot_embed_l.create_cdr_mask()
                valid_list.append(pdb_id)
            except:
                pass
        f.write('\n'.join(valid_list))


def generate_id_pairs(args):
    valid_ids_path = '../data/processed/sabdab/valid_ids_{}.txt'.format(args.set)

    # saving perms
    with open(valid_ids_path, 'r') as f:
        valid_ids = f.read().splitlines()
    perms = list(combinations(valid_ids, 2))
    random.shuffle(perms)
    
    with open('../data/processed/sabdab/all_pairs_{}.p'.format(args.set), 'wb') as f:
        pickle.dump(perms, f)

    print('DONE!')
    return


def save_sabdab_cdrs(args):

    pdb_ids_path = "../data/processed/sabdab/valid_ids_all.txt"
    with open(pdb_ids_path, 'r') as f:
        pdb_ids = f.read().splitlines()

    results_dict_H, results_dict_L = dict(), dict()
    for pdb_id in tqdm(pdb_ids):
        seq_h, seq_l = find_sequence(pdb_id=pdb_id)

        prot_h = ProteinEmbedding(seq_h, 'H')
        prot_h.create_cdr_mask()
        prot_l = ProteinEmbedding(seq_l, 'L')
        prot_l.create_cdr_mask()

        cdrs_H = ["", "", ""]
        for i in range(len(prot_h.cdr_mask)):
            for q in (1, 2, 3):
                if prot_h.cdr_mask[i] == q:
                    cdrs_H[q-1] += seq_h[i]

        cdrs_L = ["", "", ""]
        for i in range(len(prot_l.cdr_mask)):
            for q in (1, 2, 3):
                if prot_l.cdr_mask[i] == q:
                    cdrs_L[q-1] += seq_l[i]

        results_dict_H[pdb_id] = cdrs_H
        results_dict_L[pdb_id] = cdrs_L

    with open('../data/processed/sabdab/sabdab_cdrH_strs.p', 'wb') as p:
        pickle.dump(results_dict_H, p)
    with open('../data/processed/sabdab/sabdab_cdrL_strs.p', 'wb') as p:
        pickle.dump(results_dict_L, p)


def main_sabdab(args, orig_embed = False):

    pdb_ids = os.listdir('../data/raw/sabdab/sabdab_dataset')[:10]

    print(len(pdb_ids))

    out_folder = "../data/processed/sabdab/cdrembed_maskaug4"
    # out_folder = "/net/scratch3/scratch3-3/chihoim/ablm/data/processed/sabdab/cdrembed_maskaug4"
    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)

    embed_type = 'beplerberger'
    # embed_type = 'protbert'
    # embed_type = 'esm1b'
    # embed_type = 'tape'

    reload_models_to_device(args.device_num)

    k=100
    for c_type in ['H', 'L']:
        for pdb_id in tqdm(pdb_ids):

            file_name = '{}_{}_{}_k{}.p'.format(pdb_id, 'cat2', c_type, k)
            if os.path.exists(os.path.join(out_folder, embed_type, file_name)):
                continue


            seq_h, seq_l = find_sequence(pdb_id = pdb_id)
            seq = seq_h if c_type == 'H' else seq_l
            prot_embed = ProteinEmbedding(seq, c_type, fold='x')

            out_path = os.path.join(out_folder, embed_type)
            if not os.path.isdir(out_path):
                os.mkdir(out_path)


            try:
                cdr_embed = prot_embed.create_cdr_specific_embedding(embed_type, k=k, seperator=False, mask=True)
            except:
                print("processing pdb id {} didn't work...".format(pdb_id))
                # continue

            file_name = '{}_{}_{}_k{}.p'.format(pdb_id, 'cat2', c_type, k)
            with open(os.path.join(out_path, file_name), 'wb') as fh:
                print("Saving", pdb_id)
                pickle.dump(cdr_embed, fh)



def make_sabdab_features(args):
    reload_models_to_device(args.device_num)

    device = torch.device(base_config.device if torch.cuda.is_available() else "cpu")
    
    chain_type = args.chain_type
    out_dir = "../data/processed/sabdab/{}_features".format(args.set)
    embed_path = "../data/processed/sabdab/cdrembed_maskaug4/beplerberger"
    pdb_ids_path = "..data/processed/sabdab/valid_ids_{}.txt".format(args.set)
    with open(pdb_ids_path, 'r') as f:
        pdb_ids = f.read().splitlines()

    # Load our pretrained AbNet Model
    from model import AntibodyNetMultiAttn
    pretrained = AntibodyNetMultiAttn(embed_dim=2200, mid_dim2=1024, mid_dim3=512,
                                      proj_dim=252, num_enc_layers=1, num_heads=16).to(device)
    pretrained_path = "../model_ckpts/091522_bb_newnum_{}/beplerberger_epoch50.pt".format(chain_type)
    checkpoint = torch.load(pretrained_path, map_location=device)
    pretrained.load_state_dict(checkpoint['model_state_dict'])
    pretrained.eval()
    print("Loaded the pre-trained model!")

    k = 100
    out_path = os.path.join(out_dir, 'beplerberger', chain_type)
    for pdb_id in tqdm(pdb_ids):
        file_name = 'sabdab_{}_{}_{}_k{}.p'.format(pdb_id, 'cat2', chain_type, k)
        with open(os.path.join(embed_path, file_name), 'rb') as f:
            prot_emb = pickle.load(f)

        cdr_embed = torch.unsqueeze(prot_emb, dim=0) # create the batch dimension (singleton)
        cdr_embed = cdr_embed.to(device)

        # create and save STRUCTURE specific feature:
        with torch.no_grad():
            feat_task, _ = pretrained(cdr_embed, cdr_embed, None, None, 0, task_specific=True)
        assert feat_task.shape[-1] == 512
        with open(os.path.join(out_path, "{}_struc_{}.p".format(pdb_id, chain_type)), 'wb') as p:
            pickle.dump(torch.squeeze(feat_task), p)

        # create and save FUNCTION specific feature:
        with torch.no_grad():
            feat_task, _ = pretrained(cdr_embed, cdr_embed, None, None, 1, task_specific=True)
        assert feat_task.shape[-1] == 512
        with open(os.path.join(out_path, "{}_func_{}.p".format(pdb_id, chain_type)), 'wb') as p:
            pickle.dump(torch.squeeze(feat_task), p)

        # create and save INTERMEDIATE n' x k feature, before Transformer:
        with torch.no_grad():
            feat_interm, _ = pretrained(cdr_embed, cdr_embed, None, None, None, return2=True)
        assert feat_interm.shape[-1] == 256
        with open(os.path.join(out_path, "{}_interm_{}.p".format(pdb_id, chain_type)), 'wb') as p:
            pickle.dump(torch.squeeze(feat_interm), p)

    return




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="which code path to run. see main(..) for details", type=str)
    parser.add_argument("--device_num", help="GPU device for computation.", type=int, default=0)
    parser.add_argument("--num_processes", help="number of processes for multiprocessing", type=int, default=1)
    parser.add_argument("--chunk_size", help="chunk size for each processing step", type=int, default=1)
    parser.add_argument("--region", help="region of the protein sequence that we're processing", type=str)
    parser.add_argument("--chain_type", help="Which chain to process: H or L", type=str, choices = ["H", "L"])
    parser.add_argument("--set", help="Creating set1 or set2 dataset", type=str, choices = ["set1", "set2"])
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
        all_pairs_path = '../data/processed/sabdab/all_pairs_{}.p'.format(args.set)
        with open(all_pairs_path, 'rb') as f:
            all_pairs = pickle.load(f)
        n_pairs = all_pairs[:sample_n]
        n_pairs = [(i, id1, id2) for i, (id1, id2) in enumerate(n_pairs)]

        # perform multiprocessing
        import multiprocessing as mp
        from multiprocessing import get_context

        tm_scores = set()
        with get_context('spawn').Pool(processes=args.num_processes, initializer=init, 
                                       initargs=[args.device_num, args.chain_type, args.region, args.set]) as p:
            for result in tqdm(p.imap_unordered(work_parallel, n_pairs, chunksize=args.chunk_size), total=len(n_pairs)):
                tm_scores.add(result)

                # to save intermediate results. Comment out as necessary
                if len(tm_scores)%10000 == 0:
                    with open('../data/processed/sabdab/CDR{}{}_pairs_{}.p'.format(args.chain_type, args.region, len(tm_scores)), 'wb') as f:
                        pickle.dump(tm_scores, f)


        tm_scores = list(tm_scores)
        tm_scores.sort(key=lambda x: x[0])

        # save the results into a csv file
        # save_count, num_pairs = 0, 100000
        with open('../data/processed/sabdab/{}_CDR{}_{}_pair_scores_100k.csv'.format(args.set, args.region, args.chain_type), 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(['pdb_id1', 'pdb_id2', 'tm_score', 'rmsd'])
            for i in range(len(tm_scores)):
                if tm_scores[i][1] > 0 and tm_scores[i][2] >= 0:
                    writer.writerow([n_pairs[i][1], n_pairs[i][2], tm_scores[i][1], tm_scores[i][2]])
                    # save_count += 1

                    # if save_count == num_pairs:
                    #     break

    elif args.mode == '3_generate_plm_embeddings':
        main_sabdab(args)

    else:
        assert False