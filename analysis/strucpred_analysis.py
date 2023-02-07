from tqdm import tqdm, trange
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
import torch
import time
import torch.nn.functional as F

import psico.fullinit

sys.path.append('..')
sys.path.append('/net/scratch3.mit.edu/scratch3-3/chihoim/DeepAb')
import base_config
from protein_embedding import find_sequence, ProteinEmbedding

from Bio import pairwise2


def find_sequence(dataset = 'sabdab_all', pdb_id = ''):
    """
    given a dataset, find the heavy and light chain sequences
    """

    if dataset == 'sabdab_all':
        path = '/data/cb/rsingh/work/antibody/ci_data/raw/sabdab_all/sequences'
    elif dataset == 'sabdab_pure':
        path = '/data/cb/rsingh/work/antibody/ci_data/raw/sabdab_pure_042522/sabdab_dataset'
    else:
        assert False

    prot_path = os.path.join(path, pdb_id, "sequence")

    # reading the heavy chain sequence
    h = glob.glob(prot_path+"/"+pdb_id+"_*_VH.fa")
    h1 = [a for a in h if "H_VH.fa" in a]
    if len(h1)> 0:
        vhfile = h1[0]
    else:
        vhfile = h[0]

    with open(vhfile, 'r') as f:
        vh_seq = f.read().splitlines()[1]

    # reading the light chain sequence
    l = glob.glob(prot_path+"/"+pdb_id+"_*_VL.fa")
    l1 = [a for a in l if "L_VL.fa" in a]
    if len(l1)> 0:
        vlfile = l1[0]
    else:
        vlfile = l[0]

    with open(vlfile, 'r') as g:
        vl_seq = g.read().splitlines()[1]

    return (vh_seq, vl_seq)

def extract_seqs(args, chain_type):
    # reload_models_to_device(args.device_num)

    set1_path = "/data/cb/rsingh/work/antibody/ci_data/processed/sabdab_pure_042522/valid_ids_set1.txt"
    with open(set1_path, 'r') as f:
        pdb_ids_set1 = f.read().splitlines()
    print(len(pdb_ids_set1))

    set2_path = "/data/cb/rsingh/work/antibody/ci_data/processed/sabdab_pure_042522/valid_ids_set2.txt"
    with open(set2_path, 'r') as f:
        pdb_ids_set2 = f.read().splitlines()
    print(len(pdb_ids_set2))

    pdb_ids_path = "/data/cb/rsingh/work/antibody/ci_data/processed/sabdab_pure_042522/valid_ids_all.txt"
    with open(pdb_ids_path, 'r') as f:
        pdb_ids = f.read().splitlines()
    print(len(pdb_ids))

    out_folder = "/data/cb/rsingh/work/antibody/ci_data/processed/sabdab_pure_042522"

    fasta_content = []

    for pdb_id in tqdm(pdb_ids):
        seq_h, seq_l = find_sequence(dataset='sabdab_pure', pdb_id = pdb_id)
        seq_ = seq_h if chain_type == 'H' else seq_l

        which_set = 'set1' if pdb_id in pdb_ids_set1 else 'set2'

        fasta_content.append(">{}_{}_{}\n".format(pdb_id, chain_type, which_set))
        fasta_content.append(seq_+"\n")

    with open(os.path.join(out_folder, 'sabdab_all_seqs_{}.fa'.format(chain_type)), 'w') as f:
        f.writelines(fasta_content)

    print("DONE!")
    return


def create_pdb_clus_dict(args, chain_type):
    root_dir = '/data/cb/rsingh/work/antibody/ci_data/processed/sabdab_pure_042522/seq_identity0.5'
    pdb_clus_file = os.path.join(root_dir, 'sabdab_all_seqs_{}.clstr'.format(chain_type))
    with open(pdb_clus_file, 'r') as f:
        pdb_clus_lines = f.read().splitlines()

    cluster_dict = dict()
    for clus_line in pdb_clus_lines:
        if clus_line[0] == '>':
            cluster_num = clus_line[1:]
            cluster_dict[cluster_num] = []
            continue

        # extract the pdb id information:
        start, end = clus_line.index('>')+1, clus_line.rindex('...')
        cluster_dict[cluster_num].append(clus_line[start:end])

    similar_ids_dict = dict()

    for clus_id in tqdm(cluster_dict.keys()):
        orig_list = cluster_dict[clus_id]

        for pdb_id in orig_list:
            similar_ids = orig_list.copy()
            similar_ids.remove(pdb_id)
            similar_ids_dict[pdb_id] = similar_ids

    # print([i for i in similar_ids_dict.items()][:5])
    with open(os.path.join(root_dir, 'similar_ids_map_{}.p'.format(chain_type)), 'wb') as p:
        pickle.dump(similar_ids_dict, p)




def create_omegafold_preds(args):
    pdb_ids_path = "/data/cb/rsingh/work/antibody/ci_data/processed/omegafold_preds/valid_ids_set2.txt"
    with open(pdb_ids_path, 'r') as f:
        pdb_ids = f.read().splitlines()
    print(len(pdb_ids))

    seqs_fasta = "/data/cb/rsingh/work/antibody/ci_data/processed/omegafold_preds/set2_seqs.fa"
    preds_folder = '/data/cb/rsingh/work/antibody/ci_data/processed/omegafold_preds/pdb_preds'

    os.system('omegafold --device cuda:2 {} {}'.format(seqs_fasta, preds_folder))


def calc_topk_similar(args, chain_type, k):
    from model import AbMAPAttn

    #model, ckpt, df, covabdab_ids = f_initialize(base_config.device)
    threshold = 0.7

    similar_ids_dict_path = '/data/cb/rsingh/work/antibody/ci_data/processed/sabdab_pure_042522/seq_identity{}/similar_ids_map_{}.p'.format(threshold, chain_type)
    with open(similar_ids_dict_path, 'rb') as p:
        similar_ids_dict = pickle.load(p)

    pdb_ids_path_set1 = "/data/cb/rsingh/work/antibody/ci_data/processed/sabdab_pure_042522/valid_ids_set1.txt"
    pdb_ids_path_set2 = "/data/cb/rsingh/work/antibody/ci_data/processed/sabdab_pure_042522/valid_ids_set2.txt"
    embeddings_path = "/data/cb/rsingh/work/antibody/ci_data/processed/sabdab_pure_042522/cdrembed_maskaug4/beplerberger"
    cuda_num = 'cuda:{}'.format(args.device_num)
    device = torch.device(cuda_num if torch.cuda.is_available() else "cpu")

    def medoid(top_prots):
        total_scores = []
        for i, (p1_id, _) in enumerate(top_prots):
            total_score = 0
            for j, (p2_id, _) in enumerate(top_prots):
                with open(os.path.join(embeddings_path, "sabdab_{}_cat2_{}_k100.p".format(p1_id, chain_type)), 'rb') as p:
                    p1_embed = pickle.load(p).to(device)
                with open(os.path.join(embeddings_path, "sabdab_{}_cat2_{}_k100.p".format(p2_id, chain_type)), 'rb') as p:
                    p2_embed = pickle.load(p).to(device)
                if i != j:
                    with torch.no_grad():
                        e1, e2 = model(torch.unsqueeze(p1_embed, dim=0), torch.unsqueeze(p2_embed, dim=0), x1_mask=None, x2_mask=None, task=0, task_specific=True)
                        score = F.cosine_similarity(e1, e2)
                    total_score += score.item()
            total_scores.append(total_score)

        argmax = total_scores.index(max(total_scores))

        return top_prots[argmax][0]


    # load the list of set1 and set2 Sabdab proteins here:
    with open(pdb_ids_path_set1, 'r') as f1:
        pdb_ids_set1 = f1.read().splitlines()
    with open(pdb_ids_path_set2, 'r') as f2:
        pdb_ids_set2 = f2.read().splitlines()

    # load the pre-trained model (Bi-LSTM):
    pretrained_path = "../../model_ckpts/111622_abnet_bb_CDR{}{}/beplerberger_epoch50.pt".format(chain_type, args.region)
    # pretrained_path = "../../model_ckpts/best_pretrained_cdr{}{}/beplerberger_epoch50.pt".format(chain_type, args.region)
    ckpt = torch.load(pretrained_path, map_location=device)
    model = AbMAPAttn(embed_dim=2200, mid_dim2=1024, mid_dim3=512, 
                                 proj_dim=252, num_enc_layers=1, num_heads=16).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print("Pre-trained model loaded!")

    results = dict()
    for pdb_id2 in tqdm(pdb_ids_set2):
        with open(os.path.join(embeddings_path, "sabdab_{}_cat2_{}_k100.p".format(pdb_id2, chain_type)), 'rb') as p:
            prot1_embed = pickle.load(p).to(device)
        tm_scores = []
        for pdb_id1 in pdb_ids_set1:

            # remove from consideration if it is similar to the query seq!
            if '{}_{}_set1'.format(pdb_id1, chain_type) in similar_ids_dict['{}_{}_set2'.format(pdb_id2, chain_type)]:
                print("Skipped {} in set1 for querying {} from set2".format(pdb_id1, pdb_id2))
                continue


            with open(os.path.join(embeddings_path, "sabdab_{}_cat2_{}_k100.p".format(pdb_id1, chain_type)), 'rb') as p:
                prot2_embed = pickle.load(p).to(device)
            with torch.no_grad():
                e1, e2 = model(torch.unsqueeze(prot1_embed, dim=0), torch.unsqueeze(prot2_embed, dim=0), x1_mask=None, x2_mask=None, task=0, task_specific=True)
                tms = F.cosine_similarity(e1, e2)
            tm_scores.append((pdb_id1, tms.item()))
        topk = sorted(tm_scores, key = lambda x: x[1], reverse = True)[:k]
        medoid_result = medoid(topk)
        results[pdb_id2] = (topk, medoid_result)

    # with open("/data/cb/rsingh/work/antibody/ci_data/processed/eval_struc/111422_sabdab_set2_{}_topk_seqiden_{}.p".format(chain_type, threshold), "wb") as f:
        # pickle.dump(results, f)

    with open("/data/cb/rsingh/work/antibody/ci_data/processed/eval_struc/111622_sabdab_set2_CDR{}{}_topk_oursbb_seqiden_{}.p".format(chain_type, args.region, threshold), "wb") as f:
        pickle.dump(results, f)

    print("DONE!")


def calc_topk_similar_our_protbert(args, chain_type, k):
    from model import AbMAPAttn

    threshold = 0.7

    similar_ids_dict_path = '/data/cb/rsingh/work/antibody/ci_data/processed/sabdab_pure_042522/seq_identity{}/similar_ids_map_{}.p'.format(threshold, chain_type)
    with open(similar_ids_dict_path, 'rb') as p:
        similar_ids_dict = pickle.load(p)

    pdb_ids_path_set1 = "/data/cb/rsingh/work/antibody/ci_data/processed/sabdab_pure_042522/valid_ids_set1.txt"
    pdb_ids_path_set2 = "/data/cb/rsingh/work/antibody/ci_data/processed/sabdab_pure_042522/valid_ids_set2.txt"
    embeddings_path = "/data/cb/rsingh/work/antibody/ci_data/processed/sabdab_pure_042522/cdrembed_maskaug4/protbert"
    cuda_num = 'cuda:{}'.format(args.device_num)
    device = torch.device(cuda_num if torch.cuda.is_available() else "cpu")

    def medoid(top_prots):
        total_scores = []
        for i, (p1_id, _) in enumerate(top_prots):
            total_score = 0
            for j, (p2_id, _) in enumerate(top_prots):
                with open(os.path.join(embeddings_path, "sabdab_{}_cat2_{}_k100.p".format(p1_id, chain_type)), 'rb') as p:
                    p1_embed = pickle.load(p).to(device)
                with open(os.path.join(embeddings_path, "sabdab_{}_cat2_{}_k100.p".format(p2_id, chain_type)), 'rb') as p:
                    p2_embed = pickle.load(p).to(device)
                if i != j:
                    with torch.no_grad():
                        e1, e2 = model(torch.unsqueeze(p1_embed, dim=0), torch.unsqueeze(p2_embed, dim=0), x1_mask=None, x2_mask=None, task=0, task_specific=True)
                        score = F.cosine_similarity(e1, e2)
                    total_score += score.item()
            total_scores.append(total_score)

        argmax = total_scores.index(max(total_scores))

        return top_prots[argmax][0]


    # load the list of set1 and set2 Sabdab proteins here:
    with open(pdb_ids_path_set1, 'r') as f1:
        pdb_ids_set1 = f1.read().splitlines()
    with open(pdb_ids_path_set2, 'r') as f2:
        pdb_ids_set2 = f2.read().splitlines()

    # load the pre-trained model:
    pretrained_path = "../../model_ckpts/111422_abnet_protbert_CDR{}{}/protbert_epoch50.pt".format(chain_type, args.region)
    # pretrained_path = "../../model_ckpts/101722_abnet_protbert_{}_{}/protbert_epoch50.pt".format(chain_type, args.region)
    ckpt = torch.load(pretrained_path, map_location=device)
    model = AbMAPAttn(embed_dim=1024, mid_dim2=512, mid_dim3=256, 
                                 proj_dim=252, num_enc_layers=1, num_heads=16).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print("Pre-trained model loaded!")

    results = dict()
    for pdb_id2 in tqdm(pdb_ids_set2):
        with open(os.path.join(embeddings_path, "sabdab_{}_cat2_{}_k100.p".format(pdb_id2, chain_type)), 'rb') as p:
            prot1_embed = pickle.load(p).to(device)
        tm_scores = []
        for pdb_id1 in pdb_ids_set1:

            # remove from consideration if it is similar to the query seq!
            if '{}_{}_set1'.format(pdb_id1, chain_type) in similar_ids_dict['{}_{}_set2'.format(pdb_id2, chain_type)]:
                print("Skipped {} in set1 for querying {} from set2".format(pdb_id1, pdb_id2))
                continue


            with open(os.path.join(embeddings_path, "sabdab_{}_cat2_{}_k100.p".format(pdb_id1, chain_type)), 'rb') as p:
                prot2_embed = pickle.load(p).to(device)
            with torch.no_grad():
                e1, e2 = model(torch.unsqueeze(prot1_embed, dim=0), torch.unsqueeze(prot2_embed, dim=0), x1_mask=None, x2_mask=None, task=0, task_specific=True)
                tms = F.cosine_similarity(e1, e2)
            tm_scores.append((pdb_id1, tms.item()))
        topk = sorted(tm_scores, key = lambda x: x[1], reverse = True)[:k]
        medoid_result = medoid(topk)
        results[pdb_id2] = (topk, medoid_result)

    with open("/data/cb/rsingh/work/antibody/ci_data/processed/eval_struc/111522_sabdab_set2_CDR{}{}_topk_oursprot_seqiden_{}.p".format(chain_type, args.region, threshold), "wb") as f:
        pickle.dump(results, f)


    print("DONE!")

def calc_topk_similar_our_esm1b(args, chain_type, k):
    from model import AbMAPAttn

    threshold = 0.7

    similar_ids_dict_path = '/data/cb/rsingh/work/antibody/ci_data/processed/sabdab_pure_042522/seq_identity{}/similar_ids_map_{}.p'.format(threshold, chain_type)
    with open(similar_ids_dict_path, 'rb') as p:
        similar_ids_dict = pickle.load(p)

    pdb_ids_path_set1 = "/data/cb/rsingh/work/antibody/ci_data/processed/sabdab_pure_042522/valid_ids_set1.txt"
    pdb_ids_path_set2 = "/data/cb/rsingh/work/antibody/ci_data/processed/sabdab_pure_042522/valid_ids_set2.txt"
    embeddings_path = "/data/cb/rsingh/work/antibody/ci_data/processed/sabdab_pure_042522/cdrembed_maskaug4/esm1b"
    cuda_num = 'cuda:{}'.format(args.device_num)
    device = torch.device(cuda_num if torch.cuda.is_available() else "cpu")

    def medoid(top_prots):
        total_scores = []
        for i, (p1_id, _) in enumerate(top_prots):
            total_score = 0
            for j, (p2_id, _) in enumerate(top_prots):
                with open(os.path.join(embeddings_path, "sabdab_{}_cat2_{}_k100.p".format(p1_id, chain_type)), 'rb') as p:
                    p1_embed = pickle.load(p).to(device)
                with open(os.path.join(embeddings_path, "sabdab_{}_cat2_{}_k100.p".format(p2_id, chain_type)), 'rb') as p:
                    p2_embed = pickle.load(p).to(device)
                if i != j:
                    with torch.no_grad():
                        e1, e2 = model(torch.unsqueeze(p1_embed, dim=0), torch.unsqueeze(p2_embed, dim=0), x1_mask=None, x2_mask=None, task=0, task_specific=True)
                        score = F.cosine_similarity(e1, e2)
                    total_score += score.item()
            total_scores.append(total_score)

        argmax = total_scores.index(max(total_scores))

        return top_prots[argmax][0]


    # load the list of set1 and set2 Sabdab proteins here:
    with open(pdb_ids_path_set1, 'r') as f1:
        pdb_ids_set1 = f1.read().splitlines()
    with open(pdb_ids_path_set2, 'r') as f2:
        pdb_ids_set2 = f2.read().splitlines()

    # load the pre-trained model:
    pretrained_path = "../../model_ckpts/103122_abnet_esm1b_CDR{}{}/esm1b_epoch5.pt".format(chain_type, args.region)
    # pretrained_path = "../../model_ckpts/103122_abnet_esm1b_{}whole/esm1b_epoch50.pt".format(chain_type)
    ckpt = torch.load(pretrained_path, map_location=device)
    model = AbMAPAttn(embed_dim=1280, mid_dim2=512, mid_dim3=256, 
                                 proj_dim=252, num_enc_layers=1, num_heads=16).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print("Pre-trained model loaded!")

    results = dict()
    for pdb_id2 in tqdm(pdb_ids_set2):
        with open(os.path.join(embeddings_path, "sabdab_{}_cat2_{}_k100.p".format(pdb_id2, chain_type)), 'rb') as p:
            prot1_embed = pickle.load(p).to(device)
        tm_scores = []
        for pdb_id1 in pdb_ids_set1:

            # remove from consideration if it is similar to the query seq!
            if '{}_{}_set1'.format(pdb_id1, chain_type) in similar_ids_dict['{}_{}_set2'.format(pdb_id2, chain_type)]:
                print("Skipped {} in set1 for querying {} from set2".format(pdb_id1, pdb_id2))
                continue


            with open(os.path.join(embeddings_path, "sabdab_{}_cat2_{}_k100.p".format(pdb_id1, chain_type)), 'rb') as p:
                prot2_embed = pickle.load(p).to(device)
            with torch.no_grad():
                e1, e2 = model(torch.unsqueeze(prot1_embed, dim=0), torch.unsqueeze(prot2_embed, dim=0), x1_mask=None, x2_mask=None, task=0, task_specific=True)
                tms = F.cosine_similarity(e1, e2)
            tm_scores.append((pdb_id1, tms.item()))
        topk = sorted(tm_scores, key = lambda x: x[1], reverse = True)[:k]
        medoid_result = medoid(topk)
        results[pdb_id2] = (topk, medoid_result)

    with open("/data/cb/rsingh/work/antibody/ci_data/processed/eval_struc/111622_sabdab_set2_CDR{}{}_topk_oursesm1b_seqiden_{}_earlystopping.p".format(chain_type, args.region, threshold), "wb") as f:
        pickle.dump(results, f)


    print("DONE!")


def calc_topk_similar_seqalign(args, chain_type):
    from model import AbMAPAttn

    pdb_ids_path_set1 = "/data/cb/rsingh/work/antibody/ci_data/processed/sabdab_pure_042522/valid_ids_set1.txt"
    pdb_ids_path_set2 = "/data/cb/rsingh/work/antibody/ci_data/processed/sabdab_pure_042522/valid_ids_set2.txt"
    threshold = 0.5

    # load the list of set1 and set2 Sabdab proteins here:
    with open(pdb_ids_path_set1, 'r') as f1:
        pdb_ids_set1 = f1.read().splitlines()
    with open(pdb_ids_path_set2, 'r') as f2:
        pdb_ids_set2 = f2.read().splitlines()

    similar_ids_dict_path = '/data/cb/rsingh/work/antibody/ci_data/processed/sabdab_pure_042522/seq_identity{}/similar_ids_map_{}.p'.format(threshold, chain_type)
    with open(similar_ids_dict_path, 'rb') as p:
        similar_ids_dict = pickle.load(p)

    results = dict()
    for pdb_id2 in tqdm(pdb_ids_set2):
        seq_h, seq_l = find_sequence(dataset='sabdab_pure', pdb_id=pdb_id2)
        seq_2 = seq_h if chain_type == 'H' else seq_l

        align_scores = []
        for pdb_id1 in pdb_ids_set1:

            # remove from consideration if it is similar to the query seq!
            if '{}_{}_set1'.format(pdb_id1, chain_type) in similar_ids_dict['{}_{}_set2'.format(pdb_id2, chain_type)]:
                print("Skipped {} in set1 for querying {} from set2".format(pdb_id1, pdb_id2))
                continue


            seq_h, seq_l = find_sequence(dataset='sabdab_pure', pdb_id=pdb_id1)
            seq_1 = seq_h if chain_type == 'H' else seq_l

            align_score = pairwise2.align.globalxx(seq_1, seq_2, score_only=True)
            align_scores.append((align_score, pdb_id1))

        _, best_id = max(align_scores)
        results[pdb_id2] = best_id

    with open("/data/cb/rsingh/work/antibody/ci_data/processed/eval_struc/100322_sabdab_set2_{}_seqalign_seqiden_{}.p".format(chain_type, threshold), "wb") as f:
        pickle.dump(results, f)

    print("DONE!")


def calc_topk_similar_rawembed(args, chain_type, k):

    threshold = 0.7

    similar_ids_dict_path = '/data/cb/rsingh/work/antibody/ci_data/processed/sabdab_pure_042522/seq_identity{}/similar_ids_map_{}.p'.format(threshold, chain_type)
    with open(similar_ids_dict_path, 'rb') as p:
        similar_ids_dict = pickle.load(p)

    pdb_ids_path_set1 = "/data/cb/rsingh/work/antibody/ci_data/processed/sabdab_pure_042522/valid_ids_set1.txt"
    pdb_ids_path_set2 = "/data/cb/rsingh/work/antibody/ci_data/processed/sabdab_pure_042522/valid_ids_set2.txt"
    embeddings_path = f"/data/cb/rsingh/work/antibody/ci_data/processed/sabdab_pure_042522/original_embeddings/{args.embed_type}"
    cuda_num = 'cuda:{}'.format(args.device_num)
    device = torch.device(cuda_num if torch.cuda.is_available() else "cpu")

    def medoid(top_prots):
        total_scores = []
        for i, (p1_id, _) in enumerate(top_prots):
            total_score = 0
            for j, (p2_id, _) in enumerate(top_prots):
                with open(os.path.join(embeddings_path, "sabdab_{}_{}_orig.p".format(p1_id, chain_type)), 'rb') as p:
                    p1_embed = pickle.load(p).to(device)
                with open(os.path.join(embeddings_path, "sabdab_{}_{}_orig.p".format(p2_id, chain_type)), 'rb') as p:
                    p2_embed = pickle.load(p).to(device)
                if i != j:
                    p1_vec = torch.mean(p1_embed, dim=0)
                    p2_vec = torch.mean(p2_embed, dim=0)
                    score = torch.dot(p1_vec, p2_vec)

                    total_score += score.item()
            total_scores.append(total_score)


        argmax = total_scores.index(max(total_scores))

        return top_prots[argmax][0]


    # load the list of set1 and set2 Sabdab proteins here:
    with open(pdb_ids_path_set1, 'r') as f1:
        pdb_ids_set1 = f1.read().splitlines()
    with open(pdb_ids_path_set2, 'r') as f2:
        pdb_ids_set2 = f2.read().splitlines()

    results = dict()
    for pdb_id2 in tqdm(pdb_ids_set2):
        with open(os.path.join(embeddings_path, "sabdab_{}_{}_orig.p".format(pdb_id2, chain_type)), 'rb') as p:
            raw_embed1 = pickle.load(p).to(device)
        tm_scores = []
        for pdb_id1 in pdb_ids_set1:

            # remove from consideration if it is similar to the query seq!
            if '{}_{}_set1'.format(pdb_id1, chain_type) in similar_ids_dict['{}_{}_set2'.format(pdb_id2, chain_type)]:
                print("Skipped {} in set1 for querying {} from set2".format(pdb_id1, pdb_id2))
                continue

            with open(os.path.join(embeddings_path, "sabdab_{}_{}_orig.p".format(pdb_id1, chain_type)), 'rb') as p:
                raw_embed2 = pickle.load(p).to(device)

            # PROTBERT ONLY:
            p1_vec = torch.mean(raw_embed1, dim=0)
            p2_vec = torch.mean(raw_embed2, dim=0)
            tms = torch.dot(p1_vec, p2_vec)

            tm_scores.append((pdb_id1, tms.item()))
        topk = sorted(tm_scores, key = lambda x: x[1], reverse = True)[:k]
        medoid_result = medoid(topk)
        results[pdb_id2] = (topk, medoid_result)

    with open(f"/data/cb/rsingh/work/antibody/ci_data/processed/eval_struc/110222_sabdab_set2_{chain_type}_topk_iden{threshold}_{args.embed_type}.p", "wb") as f:
        pickle.dump(results, f)

    print("DONE!")


def init_predict_struc_deepab(c_type):
    global chain_type
    chain_type = c_type

    os.environ['MKL_THREADING_LAYER'] = 'GNU'

def predict_struc_deepab(pdb_id):
    # find the H or L chain sequence of that pdb_id:
    seq_h, seq_l = find_sequence(dataset='sabdab_pure', pdb_id=pdb_id)
    if chain_type == 'H':
        sequence = seq_h
    else:
        sequence = seq_l

    # check if it was already made:
    if pdb_id in os.listdir('/data/cb/rsingh/work/antibody/ci_data/processed/eval_struc/pred_struc_{}'.format(chain_type)):
        return pdb_id

    # create a fasta file of a sequence to predict structure from:
    temp_loc = '/net/scratch3.mit.edu/scratch3-3/chihoim/DeepAb/temp/{}_seq.fasta'.format(pdb_id)
    with open(temp_loc, 'w') as f:
        content = ['>:H\n', sequence]
        f.writelines(content)

    # predict the structure!
    pred_dir = '/data/cb/rsingh/work/antibody/ci_data/processed/eval_struc/pred_struc_{}_2/{}'.format(chain_type, pdb_id)
    os.system('mkdir {}'.format(pred_dir))
    os.system('python ../../DeepAb/predict.py {} --use_gpu --target {} --decoys 5 --single_chain --model_dir {} --pred_dir {}'.format(
                temp_loc, pdb_id, '../../DeepAb/trained_models/ensemble_abresnet', pred_dir))

    return pdb_id


def calc_rmsd(struc1, struc2, chain_type, against = "ours", region = 'whole', fold = '0'):
    from pymol import cmd

    cmd.delete('all')

    cmd.load(struc1, "999C")
    cmd.load(struc2, "999D")

    cmd.split_chains('999C')
    cmd.split_chains('999D')

    if against is 'ours':
        pdb_id1, pdb_id2 = struc1[-8:-4], struc2[-8:-4]
    else:
        pdb_id1, pdb_id2 = struc1[-8:-4], struc2[-15:-11]
        # pdb_id1, pdb_id2 = struc1[-8:-4], struc2[-10:-6]

    p1_seq_H, p1_seq_L = find_sequence(dataset = 'sabdab_pure', pdb_id=pdb_id1)
    p2_seq_H, p2_seq_L = find_sequence(dataset = 'sabdab_pure', pdb_id=pdb_id2)

    if chain_type == 'H':
        p1_seq, p2_seq = p1_seq_H, p2_seq_H
    else:
        p1_seq, p2_seq = p1_seq_L, p2_seq_L

    start1, end1, start2, end2 = 0, 0, 0, 0
    if region != 'whole':
        p1 = ProteinEmbedding(p1_seq, chain_type=chain_type, fold=fold)
        p1.create_cdr_mask()
        p2 = ProteinEmbedding(p2_seq, chain_type=chain_type, fold=fold)
        p2.create_cdr_mask()

        p1_mask = p1.cdr_mask.tolist()
        a = list(map(int, p1_mask))
        t = list(map(str, a))
        p1_mask_str = "".join(t)

        p2_mask = p2.cdr_mask.tolist()
        a = list(map(int, p2_mask))
        t = list(map(str, a))
        p2_mask_str = "".join(t)

        start1, end1 = p1_mask_str.index(region)+1, p1_mask_str.rindex(region)+1
        start2, end2 = p2_mask_str.index(region)+1, p2_mask_str.rindex(region)+1

        print("Indices for Chain {} Region {}, Against {} :".format(chain_type, region, against), start1, end1, start2, end2)


    # find the chain letters
    chain_letters = []
    data_path = '/data/cb/rsingh/work/antibody/ci_data/raw/sabdab_pure_042522/sabdab_dataset'
    for pdb_id in [pdb_id1, pdb_id2]:
        h = glob.glob(os.path.join(data_path, pdb_id, 'sequence', '{}_*_V{}.fa'.format(pdb_id, chain_type)))
        h1 = [a for a in h if "{}_V{}.fa".format(chain_type, chain_type) in a]
        if len(h1) > 0 :
            vhfile = h1[0]
        else:
            vhfile = h[0]
        chain_letter = vhfile[-7]
        chain_letters.append(chain_letter)

    # if chains in predicted structure not in GT structure
    if chain_letters[1] not in cmd.get_chains('999D'):
        chain_letters[1] = cmd.get_chains('999D')[0]

    assert chain_letters[0] in cmd.get_chains('999C')
    assert chain_letters[1] in cmd.get_chains('999D')

    prot1 = '999C_{}'.format(chain_letters[0])
    prot2 = '999D_{}'.format(chain_letters[1])


    if region == 'whole':
        try:
            rmsd = cmd.super(prot1+'////CA', prot2+'////CA')[0]
            # rmsd = cmd.align(prot1+'////CA', prot2+'////CA')[0]
            print(f"RMSD: {rmsd}")
        except:
            print("{},{}-{} RMSD for region {} didn't work...".format(pdb_id1, pdb_id2, chain_type, region))
            rmsd = -1
        try:
            tm_score = cmd.tmalign(prot1+'////CA', prot2+'////CA')
        except:
            print("{},{}-{} TM Score for region {} didn't work...".format(pdb_id1, pdb_id2, chain_type, region))
            tm_score = -1

    else:
        try:
            rmsd = cmd.super(prot1+'///{}-{}/CA'.format(start1, end1), prot2+'///{}-{}/CA'.format(start2, end2))[0]
            # rmsd = cmd.align(prot1+'///{}-{}/CA'.format(start1, end1), prot2+'///{}-{}/CA'.format(start2, end2))[0]
            print(f"RMSD: {rmsd}")
        except:
            print("{},{}-{} RMSD for region {} didn't work...".format(pdb_id1, pdb_id2, chain_type, region))
            rmsd = -1
        try:
            tm_score = cmd.tmalign(prot1+'///{}-{}/CA'.format(start1, end1), prot2+'///{}-{}/CA'.format(start2, end2))
        except:
            print("{},{}-{} TM Score for region {} didn't work...".format(pdb_id1, pdb_id2, chain_type, region))
            tm_score = -1


    return (rmsd, tm_score)

def evaluate_pred(args, chain_type, region):
    if args.log_name == "":
        print("Need a log name!")
        assert False

    eval_prots = "/data/cb/rsingh/work/antibody/ci_data/processed/sabdab_pure_042522/valid_ids_set2.txt"
    dataset_path = '/data/cb/rsingh/work/antibody/ci_data/raw/sabdab_pure_042522/sabdab_dataset'
    deepab_path = "/data/cb/rsingh/work/antibody/ci_data/processed/eval_struc/pred_struc_{}".format(chain_type)
    omega_path = '/data/cb/rsingh/work/antibody/ci_data/processed/omegafold_preds/pdb_preds'
    alpha_path = '/data/cb/rsingh/work/antibody/ci_data/processed/alphafold_preds/preds_collated'

    # get the list of set2 pdb ids for evaluation
    with open(eval_prots, 'r') as f:
        eval_prots_list = f.read().splitlines()

    # our model's predictions on Set2
    root_dir = '/data/cb/rsingh/work/antibody/ci_data/processed/eval_struc'
    our_model_pred_path = os.path.join(root_dir, '111622_sabdab_set2_CDR{}{}_topk_oursbb_seqiden_0.7.p'.format(chain_type, region))


    with open(our_model_pred_path, 'rb') as f:
        our_model_pred = pickle.load(f)

    ours_wrong, deepab_wrong = 0, 0
    total_ours_tms, total_ours_rmsd, total_deepab_tms, total_deepab_rmsd = [], [], [], []
    for pt_id in tqdm(eval_prots_list):
        gt_path = os.path.join(dataset_path, pt_id, "structure", "{}.pdb".format(pt_id))

        # check if DeepAb path is correct
        deepab_struc = os.path.join(deepab_path, pt_id, "{}.deepab.pdb".format(pt_id))
        if not os.path.isfile(deepab_struc):
            deepab_wrong += 1
            continue

        # check if OmegaFold path is correct
        # deepab_struc = os.path.join(omega_path, "{}_{}.pdb".format(pt_id, chain_type))
        # if not os.path.isfile(deepab_struc):
        #     deepab_wrong += 1
        #     continue

        # check if AlphaFold path is correct
        # deepab_struc = os.path.join(alpha_path, "{}_{}.pdb".format(pt_id, chain_type))
        # if not os.path.isfile(deepab_struc):
        #     deepab_wrong += 1
        #     continue

        # calculate GT vs DeepAb:
        # rmsd_deepab, tms_deepab = calc_rmsd(gt_path, deepab_struc, chain_type, against="deepab", region=region, fold=args.device_num)
        rmsd_deepab, tms_deepab = 1, 1

        # calculate GT vs our model:
        # for our_pred_id in [our_model_pred[pt_id]]: # ONLY FOR SEQ ALIGNMENT!
        # our_pred_id_ = our_model_pred[pt_id][-1] if region == 'whole' else our_model_pred[pt_id][int(region)-1]
        # for our_pred_id in [our_pred_id_]:

        # print("Candidates:", our_model_pred[pt_id][0]) # RAW EMBED
        # for our_pred_id, _ in our_model_pred[pt_id][0]:
        for our_pred_id in [our_model_pred[pt_id][1]]: # MEDOID METHOD

            our_model_struc = os.path.join(dataset_path, our_pred_id, "structure", "{}.pdb".format(our_pred_id))
            rmsd_our, tms_our = calc_rmsd(gt_path, our_model_struc, chain_type, region=region, fold=args.device_num)

            if rmsd_our > 0 and tms_our > 0:
                break

        if rmsd_our < 0 or tms_our <= 0:
            print("WRONG!!")
            print("predict for", pt_id, "our prediction", our_pred_id)
            print("OUR RMSD: {}, OUR TMS: {}".format(rmsd_our, tms_our))
            ours_wrong += 1
            continue
        if rmsd_deepab < 0 or tms_deepab <= 0:
            deepab_wrong += 1
            continue

        total_ours_tms.append(tms_our)
        total_ours_rmsd.append(rmsd_our)
        total_deepab_tms.append(tms_deepab)
        total_deepab_rmsd.append(rmsd_deepab)


    # SAVE the data
    results_dict = dict()
    results_dict['ours_tms'] = np.array(total_ours_tms)
    results_dict['ours_rmsd'] = np.array(total_ours_rmsd)
    results_dict['deepab_tms'] = np.array(total_deepab_tms)
    results_dict['deepab_rmsd'] = np.array(total_deepab_rmsd)

    print("Total number of Abs evaluated on: {}, Missing {} (Ours), {} (DeepAb)".format(
                                                    len(total_ours_tms), ours_wrong, deepab_wrong))

    save_path = '/data/cb/rsingh/work/antibody/ci_data/processed/eval_struc/{}'.format(args.log_name)

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    with open(os.path.join(save_path, "{}_{}.p".format(region, chain_type)), "wb") as f:
        pickle.dump(results_dict, f)

    print("Chain: {}, Region: {}".format(chain_type, region))
    for name, data in [("Ours", total_ours_tms), ("DeepAb", total_deepab_tms)]:
        print("***********************************")
        print("TMS")
        print("***********************************")
        print("{} Model Statistics:".format(name))
        print("Mean: {}".format(np.mean(data)))
        print("Std: {}".format(np.std(data)))
        print("StdErr: {}".format(np.std(data)/np.sqrt(len(total_ours_tms))))
        print("Max: {}".format(np.max(data)))
        print("Min: {}".format(np.min(data)))
        print("Median: {}".format(np.median(data)))
    print("***********************************")

    for name, data in [("Ours", total_ours_rmsd), ("DeepAb", total_deepab_rmsd)]:
        print("***********************************")
        print("RMSD")
        print("***********************************")
        print("{} Model Statistics:".format(name))
        print("Mean: {}".format(np.mean(data)))
        print("Std: {}".format(np.std(data)))
        print("StdErr: {}".format(np.std(data)/np.sqrt(len(total_ours_tms))))
        print("Max: {}".format(np.max(data)))
        print("Min: {}".format(np.min(data)))
        print("Median: {}".format(np.median(data)))
    print("***********************************")




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="which code path to run. see main(..) for details", type=str)
    parser.add_argument("--device_num", help="GPU device for computation.", type=int, default=0)
    parser.add_argument("--log_name", help="log name for saving", type=str)
    parser.add_argument("--chain_type", help="chain type.", type=str)
    parser.add_argument("--embed_type", help="which embedding to use", type=str)
    parser.add_argument("--region", help="Region of the Ab to calculate scores", type=str)
    parser.add_argument("--num_processes", help="number of processes for multiprocessing", type=int, default=1)
    parser.add_argument("--chunk_size", help="chunk size for each processing step", type=int, default=1)
    parser.add_argument("--extra", help="put this as the LAST option and arbitrary space-separated key=val pairs after that", type=str, nargs='*')

    args = parser.parse_args()
    assert args.mode is not None
    
    extra_args = dict([a.split("=") for a in args.extra]) if args.extra else {}
    args.extra = extra_args
    
    pd.set_option('use_inf_as_na', True)


    if args.mode == '1_calc_topk_similar':
        calc_topk_similar(args, chain_type=args.chain_type, k=10)

    elif args.mode == '1_calc_topk_similar_rawembed':
        calc_topk_similar_rawembed(args, chain_type=args.chain_type, k=10)

    elif args.mode == '1_calc_topk_similar_our_protbert':
        calc_topk_similar_our_protbert(args, chain_type=args.chain_type, k=10)

    elif args.mode == '1_calc_topk_similar_our_esm1b':
        calc_topk_similar_our_esm1b(args, chain_type=args.chain_type, k=10)

    elif args.mode == '2_predict_struc_deepab':

        import multiprocessing as mp
        from multiprocessing import Pool, set_start_method
        set_start_method('spawn')

        # get the list of set 2 pdb ids
        pdb_ids_path_set2 = "/data/cb/rsingh/work/antibody/ci_data/processed/sabdab_pure_042522/valid_ids_set2.txt"
        with open(pdb_ids_path_set2, 'r') as f2:
            pdb_ids_set2 = f2.read().splitlines()

        results = []
        c_type = args.chain_type
        with Pool(processes=12, initializer=init_predict_struc_deepab, initargs=[c_type]) as p:

            for result in tqdm(p.imap_unordered(predict_struc_deepab, pdb_ids_set2), total=len(pdb_ids_set2)):
                results.append(result)

        print("DONE!")

    elif args.mode == '3_evaluate_pred':

        evaluate_pred(args, chain_type=args.chain_type, region=args.region)

        # for c in ['H', 'L']:
        #     for r in ['1', '2', '3']:
        #         evaluate_pred(args, chain_type=c, region=r)

        # for c in ['H', 'L']:
        #     for r in ['whole', '1', '2', '3']:
        #         evaluate_pred(args, chain_type=c, region=r)

    elif args.mode == '4_extract_seqs':

        extract_seqs(args, args.chain_type)

    elif args.mode == '5_create_omegafold_preds':

        create_omegafold_preds(args)

    elif args.mode == '6_calc_topk_similar_seqalign':

        calc_topk_similar_seqalign(args, chain_type=args.chain_type)

    elif args.mode == '7_calc_topk_similar_seqalign_cdr':

        calc_topk_similar_seqalign_cdr(args, chain_type=args.chain_type)

    elif args.mode == '8_create_pdb_clus_dict':
        create_pdb_clus_dict(args, chain_type=args.chain_type)

    else:
        assert False