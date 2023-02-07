from itertools import combinations
import os, gc
import pickle
import pandas as pd
from tqdm import tqdm
import argparse
import random
import pickle
import numpy as np
from matplotlib import pyplot as plt
# import torch
import multiprocessing as mp
import time

import sklearn.datasets
import umap
from sklearn_extra.cluster import KMedoids

from utils import evaluate_spearman, scatter_plot, get_free_gpu

import base_config
from protein_embedding import ProteinEmbedding, find_sequence
from model import AntibodyNetMulti
import embed
from mutate import generate_mutation_cov
import copy


# device = None
# pretrained_path = None
# ckpt = None
# model = None
# df = None
# covabdab_ids = None

def f_initialize(device_str):
    # load pre-trained AntibodyNet Model
    device = torch.device(device_str)
    pretrained_path = "../model_ckpts/best_arch_Set1/020322_beplerberger_epoch40.pt"
    ckpt = torch.load(pretrained_path)
    model = AntibodyNetMulti(embed_dim=2200).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print("Pre-trained model loaded!")

    pdb_ids_path = "/data/cb/rsingh/work/antibody/ci_data/raw/covabdab/neutralize_sarscov2_cset2.csv"
    df = pd.read_csv(pdb_ids_path)
    covabdab_ids = df["Name"].tolist()
    return model, ckpt, df, covabdab_ids


def make_cdr_emb(p_emb, kmut_matrix, chain_type = "h"):
    """
    given a (num_muts x n x d) k-mutations matrix, compute the CDR embedding 
    by averaging over the k mutation matrices and then concatenating the CDR regions 
    of the sequence --> (n* x d)
    """

    assert chain_type in ['h', 'l']

    if chain_type == 'h':
        diff_matrix = p_emb.embedding[0] - kmut_matrix
    else:
        diff_matrix = p_emb.embedding[1] - kmut_matrix
    
    avg_kmut = torch.mean(diff_matrix, dim=0)

    # assert avg_kmut.shape[0] == len(mask)

    return avg_kmut


def generate_embeddings(**kwargs):
    data_path = '/data/cb/rsingh/work/antibody/ci_data/raw/pyigclassify/cdr_data.txt'
    embed_path = '/data/cb/rsingh/work/antibody/ci_data/processed/sabdab_all/cdr_embeddings/beplerberger'
    out_path = "/data/cb/rsingh/work/antibody/ci_data/processed/cdrtxt/full_H"
    df = pd.read_csv(data_path, sep='\t')
    print(df.columns)

    h1s = df.loc[df['CDR'] == 'H1']

    print(h1s)

    # pdb_ids = df['PDB'].unique()
    # print(len(pdb_ids))

    raise ValueError

    # get the embeddings
    for ind in tqdm(h1s.index):
        prot_id = h1s['PDB'][ind].lower()
        # seq_cdr1 = h1s['seq'][ind]
        # seq_cdr2 = h1s['seq'][ind+1]
        # seq_cdr3 = h1s['seq'][ind+2]
        try:
            cdr_embed = pickle.load(open(os.path.join(embed_path, 'sabdab_{}_cat2_H_k100.p'.format(prot_id))))
        except:
            print("couldn't find the file!")
            raise ValueError


        cluster = h1s['cluster'][ind]

        emb_type = 'beplerberger'
        prot_embed = ProteinEmbedding()
        prot_embed.vh_seq = sequence
        prot_embed.vl_seq = sequence
        prot_embed.embed_seq(emb_type)
        prot_embed.cdr_masks = (torch.ones(len(sequence)), torch.ones(len(sequence)))

        k=100
        mut_type = 'cat2'
        kmut_vh, kmut_vl = prot_embed.create_kmut_matrix(num_muts = k,
                                                     mut_type = mut_type, embed_type = emb_type)

        cdr_embedding_h = make_cdr_emb(prot_embed, kmut_vh)

        fh = open(os.path.join(out_path, 'cdrtxt_{}_{}_H_k{}.p'.format(prot_id, mut_type, k)), 'wb')

        temp_dic = {'pdb_id': prot_id, 'embedding': cdr_embedding_h, 'cluster': cluster}

        pickle.dump(temp_dic, fh)

        fh.close()


def topk_similar(chain_type, k = 10, **kwargs):
    #model, ckpt, df, covabdab_ids = f_initialize(base_config.device)

    pdb_ids_path_set1 = "/data/cb/rsingh/work/antibody/ci_data/raw/sabdab_all/valid_ids_Set1.txt"
    pdb_ids_path_set2 = "/data/cb/rsingh/work/antibody/ci_data/raw/sabdab_all/valid_ids_Set2.txt"
    embeddings_path = "/data/cb/rsingh/work/antibody/ci_data/processed/sabdab_all/cdrembed_maskaug4/beplerberger"
    device = torch.device(base_config.device if torch.cuda.is_available() else "cpu")

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
                    score = model(torch.unsqueeze(p1_embed, dim=0), torch.unsqueeze(p2_embed, dim=0), task=0)
                    total_score += score.item()
            total_scores.append(total_score)

        argmax = total_scores.index(max(total_scores))

        return top_prots[argmax][0]


    # load the list of set1 and set2 Sabdab proteins here:
    with open(pdb_ids_path_set1, 'r') as f1:
        pdb_ids_set1 = f1.read().splitlines()
    with open(pdb_ids_path_set2, 'r') as f2:
        pdb_ids_set2 = f2.read().splitlines()

    # load the trained bi-modal model (Bi-LSTM):
    pretrained_path = "../model_ckpts/best_arch_Set1_LC/best_lc_beplerberger_epoch40.pt"
    ckpt = torch.load(pretrained_path)
    model = AntibodyNetMulti(embed_dim=2200).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print("Pre-trained model loaded!")

    results = dict()
    for pdb_id2 in tqdm(pdb_ids_set2):
        with open(os.path.join(embeddings_path, "sabdab_{}_cat2_{}_k100.p".format(pdb_id2, chain_type)), 'rb') as p:
            prot1_embed = pickle.load(p).to(device)
        tm_scores = []
        for pdb_id1 in pdb_ids_set1:
            with open(os.path.join(embeddings_path, "sabdab_{}_cat2_{}_k100.p".format(pdb_id1, chain_type)), 'rb') as p:
                prot2_embed = pickle.load(p).to(device)
            tms = model(torch.unsqueeze(prot1_embed, dim=0), torch.unsqueeze(prot2_embed, dim=0), task=0)
            tm_scores.append((pdb_id1, tms.item()))
        topk = sorted(tm_scores, key = lambda x: x[1], reverse = True)[:k]
        medoid_result = medoid(topk)
        results[pdb_id2] = (topk, medoid_result)

    with open("../results/sabdab_set2_{}_topk.p".format(chain_type), "wb") as f:
        pickle.dump(results, f)

    print("DONE!")


def validate_mutants():
    pdb_ids_path = "/data/cb/rsingh/work/antibody/ci_data/raw/sabdab_all/valid_ids_Set1.txt"
    with open(pdb_ids_path, 'r') as f:
        pdb_ids = f.read().splitlines()

    for pdb_id in pdb_ids:
        for embed_type in ['beplerberger', 'esm1b']:
            seq_h, seq_l = find_sequence(pdb_id = pdb_id)

            prot_embed_h = ProteinEmbedding(sequence = seq_h, chain_type='H')

            prob_h = embed.compute_prob(seq_h, embed_type = embed_type)



def init(shared_val):
    global start_time
    start_time = shared_val

def work_parallel(cov_id):

    # locking & threading
    with start_time.get_lock():
        wait_time = max(0, start_time.value - time.time())
        time.sleep(wait_time)
 
        # find free gpu
        free_gpu = -1
        while free_gpu < 0:
            free_gpu = get_free_gpu()

        thread_device_str = f'cuda:{free_gpu}'

        # debug
        global torch
        import torch
        torch.cuda.set_device(thread_device_str)
        
        device = torch.device(thread_device_str)
        model, _, df, _ = f_initialize(thread_device_str)
        # print("Flag 348.30 ", df.shape, device, type(model))

        thread_bb_model = copy.deepcopy(embed.bb_model.cpu()).cuda(device)
        thread_esm_model = copy.deepcopy(embed.esm_model.cpu()).cuda(device)

        start_time.value = time.time() + 5.0


    seq_h = df.loc[df["Name"]==cov_id]['VH or VHH'].item()

    prot_embed_h = ProteinEmbedding(sequence = seq_h, chain_type='H', 
                                    embed_device=thread_device_str, embed_model=thread_bb_model)
    prot_embed_h.create_cdr_mask()
    mut_seqs = []
    num_muts, lowest = 50000, 2500
    k = 100
    # print("Flag 348.45 ", time.time())

    print("{}: Computing the logits of {} randomly mutated seqs ...".format(cov_id, num_muts))
    for i in range(num_muts):
        mut_seq = generate_mutation_cov(prot_embed_h.seq, prot_embed_h.cdr_mask, p=0.5, mut_type='cat2')
        perp_score = embed.compute_prob(mut_seq, embed_type='esm1b',
                                embed_device=thread_device_str, embed_model=thread_esm_model).item()
        mut_seqs.append((perp_score, mut_seq))
        # if i%10==1: print("Flag 348.46 ", i)

    Y = sorted(mut_seqs)[:lowest]

    # print("Flag 348.50 ", time.time())

    del thread_esm_model
    # for each of the top 5000 mutated seqs, pass through the AntibodyNet Model:
    # get the structure representation
    Y_strucs = []
    print("{}: Generating structure features of the top {} sequences...".format(cov_id, lowest))
    for _, sequence in Y:
        # generate the CDR embedding
        prot_embed = ProteinEmbedding(sequence, 'H', embed_device=thread_device_str, embed_model=thread_bb_model)
        prot_embed.create_cdr_mask()
        prot_embed.embed_seq(embed_type='beplerberger')
        kmut_matr_h = prot_embed.create_kmut_matrix(num_muts=k, embed_type='beplerberger')
        cdr_embed = prot_embed.create_cdr_embedding(kmut_matr_h, sep=False, mask=True)
        kmut_matr_h = kmut_matr_h.detach()
        cdr_embed = cdr_embed.to(device)
        cdr_embed = torch.unsqueeze(cdr_embed, dim=0)
        
        # feed to pre-trained network
        x, x_pos = cdr_embed[:,:,-2204:-4], cdr_embed[:,:,-4:]
        cdr_embed = cdr_embed.detach()
        x = model.project(x)
        x = torch.cat([x, x_pos], dim=-1)
        feat = torch.squeeze(model.recurrent(x, 0)).cpu().detach()
        Y_strucs.append(feat)

        #x, feat = x.detach(), feat.detach()

        # delete variables
        del feat, x, x_pos, cdr_embed, prot_embed, kmut_matr_h
    
    # print("Flag 348.90 ", time.time())
    del thread_bb_model, model, Y
    gc.collect()
    torch.cuda.empty_cache()

    # Apply CovAbDab neutralization cross-validation-learned prediction model to each in Y_strucs
    # identify the top 300
    # TO-DO (RS)
    # Let neut_pred be the top 300 list

    # # TEST (MUST CHANGE):
    # neut_pred = Y_strucs[:50]

    # neut_pred = neut_pred.detach().cpu().numpy()
    # kmedoids = KMedoids(n_clusters=5).fit(neut_pred)
    # # k-medoids:
    # km_centers = kmedoids.cluster_centers_



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='/data/cb/rsingh/work/antibody/ci_data/processed/sabdab_all',
                        help='Directory for location of training data')

    parser.add_argument('--num_epochs', type=int, default=1,
                        help='Number of epochs for training.')

    parser.add_argument('--batch_size', type=int, default=8,
                        help='Input batch size for the model.')


    args = parser.parse_args()

    # device = torch.device(base_config.device if torch.cuda.is_available() else "cpu")
    # pretrained_path = "../model_ckpts/best_arch_Set1/020322_beplerberger_epoch40.pt"
    # ckpt = torch.load(pretrained_path)
    # model = AntibodyNetMulti(embed_dim=2200).to(device)
    # model.load_state_dict(ckpt['model_state_dict'])
    # model.eval()
    # print("Pre-trained model loaded!")

    pdb_ids_path = "/data/cb/rsingh/work/antibody/ci_data/raw/covabdab/neutralize_sarscov2_cset2.csv"
    df = pd.read_csv(pdb_ids_path)
    covabdab_ids = df["Name"].tolist()[:50]


    # if torch.cuda.is_available():
    #     print("Using the GPU!")
    # else:
    #     print("WARNING: Could not find GPU! Using CPU only")

    # main(args)
    import torch
    torch.cuda.set_device('cuda:3')
    from torch.multiprocessing import set_start_method
    set_start_method('spawn')
    p = mp.Pool(processes=32, initializer=init, initargs=[mp.Value('d')])
    # p.map(work_parallel, covabdab_ids)
    for _ in tqdm(p.imap_unordered(work_parallel, covabdab_ids), total=len(covabdab_ids)):
        pass