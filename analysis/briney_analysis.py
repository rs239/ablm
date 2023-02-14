import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import os, sys
import h5py
import pickle
import random
import glob

from torch.nn.utils.rnn import pad_sequence
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch import nn
from torch import optim as opt
from torch.nn.utils.rnn import pad_sequence

sys.path.append("..")
from protein_embedding import ProteinEmbedding
from embed import reload_models_to_device
from utils import evaluate_spearman

from Bio import pairwise2



def generate_seq_count(args):
    seq_count_dict = dict()
    raw_file_num = args.patient_id

    raw_files_dir = '/data/cb/rsingh/work/antibody/ci_data/raw/briney_nature_2019/{}_consensus-cdr3nt-90_minimal'.format(raw_file_num)
    raw_files = os.listdir(raw_files_dir)

    for raw_file in raw_files:
        file_path = os.path.join(raw_files_dir, raw_file)
        df = pd.read_csv(file_path)
        df = df.loc[~df['vj_aa'].str.contains('*', regex=False)]
        df.reset_index(drop=True, inplace=True)

        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
            seq = row['vj_aa']
            if seq in seq_count_dict.keys():
                seq_count_dict[seq] += 1
                continue
            else:
                seq_count_dict[seq] = 1


    # create seq count dict
    save_dir = '/data/cb/rsingh/work/antibody/ci_data/processed/briney_nature_2019/{}_consensus-cdr3nt-90_minimal'.format(raw_file_num)

    with open(os.path.join(save_dir, '{}_seqcount.p'.format(raw_file_num)), 'wb') as d:
        pickle.dump(seq_count_dict, d)


def init(dev_num):
    global device_n
    device_n = dev_num

    device = torch.device("cuda:{}".format(device_n) if torch.cuda.is_available() else "cpu")

    reload_models_to_device(device_n)


def work_parallel(pair):
    dev, seq = pair
    prot = ProteinEmbedding(seq, 'H', embed_device="cuda:{}".format(device_n), dev=dev, fold=device_n)

    try:
        prot.create_cdr_mask()
    except:
        return (None, None)

    k=50

    cdr_embed = prot.create_cdr_specific_embedding(embed_type='beplerberger', k=k)

    return (cdr_embed.detach().cpu().numpy(), seq)


def init_align(seq_samples):
    global seqs_list
    seqs_list = seq_samples

def parallel_align(pair_idxs):
    idx1, idx2 = pair_idxs
    seq1, seq2 = seqs_list[idx1][0], seqs_list[idx2][0]
    align_score = pairwise2.align.globalxx(seq1, seq2, score_only=True)
    return idx1, idx2, align_score

def unpad_embedding(embedding):
    # embedding should be in the dimension of (1 x n' x d)
    
    v = np.abs(embedding).sum(axis=2)
    v2 = 1*(v < 1e-8)
    v3 = np.argmax(v2, axis=1)[0]
    
    return v3




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="which code path to run. see main(..) for details", type=str)
    parser.add_argument("--patient_id", help="ID of the patient to process", type=int)
    parser.add_argument("--device_num", help="GPU device for computation.", type=int, default=0)
    parser.add_argument("--num_processes", help="number of processes for multiprocessing", type=int)
    parser.add_argument("--chunk_size", help="chunk size for mapping", type=int)
    parser.add_argument("--sample", help="Are we sampling from the total dataset.", type=int, default=-1)
    parser.add_argument("--extra", help="put this as the LAST option and arbitrary space-separated key=val pairs after that", type=str, nargs='*')

    args = parser.parse_args()
    assert args.mode is not None
    
    extra_args = dict([a.split("=") for a in args.extra]) if args.extra else {}
    args.extra = extra_args
    
    pd.set_option('use_inf_as_na', True)

    if args.mode == '1_generate_seqcount':
        generate_seq_count(args)


    elif args.mode == '2_generate_features':
        # WRITE PARALLEL PROCESSING CODE HERE:
        patient_id = args.patient_id

        data_dir = "/data/cb/rsingh/work/antibody/ci_data/processed/briney_nature_2019/{}".format(patient_id)

        with open(os.path.join(data_dir, '{}_seqcount.p'.format(patient_id)), 'rb') as d:
            seq_count_dict = pickle.load(d)

        print("Total Number of Seqs:", len(seq_count_dict.keys()))

        if args.sample < 0:
            n='all'
            sampled_seqs_ = seq_count_dict.keys()
        else:
            # sample 100k of the seqs:
            n = args.sample
            sampled_seqs_ = random.sample(seq_count_dict.keys(), n)

        sampled_seqs = [(i, s) for i, s in enumerate(sampled_seqs_)]        

        import multiprocessing as mp
        from multiprocessing import get_context

        feats_result = []

        with get_context('spawn').Pool(processes=args.num_processes, initializer=init, initargs=[args.device_num]) as p:

            for result in tqdm(p.imap_unordered(work_parallel, sampled_seqs, chunksize=args.chunk_size), total=len(sampled_seqs)):
                feats_result.append(result)

        feats_final, labels_final, seqs_final = [], [], []
        for i, (feat, seq) in enumerate(feats_result):
            if feat is None:
                continue
            feats_final.append(torch.FloatTensor(feat))
            labels_final.append(seq_count_dict[seq])
            seqs_final.append(seq)

        # feats_final = np.stack(feats_final)
        feats_final = pad_sequence(feats_final, batch_first=True)
        labels_final = np.array(labels_final)

        assert feats_final.shape[0] == len(labels_final) == len(seqs_final)

        n_ = feats_final.shape[0]

        file_name = '{}_n{}_data.h5'.format(patient_id, n_)
        if file_name in os.listdir(data_dir):
            os.system('rm -f {}'.format(os.path.join(data_dir, file_name)))

        with h5py.File(os.path.join(data_dir, file_name), "w") as f:
            dset = f.create_dataset('feats', feats_final.shape)
            dset[:,:,:] = feats_final

            dset_lab = f.create_dataset('labels', labels_final.shape)
            dset_lab[:] = labels_final

        with open(os.path.join(data_dir, '{}_n{}_seqs.p'.format(patient_id, n_)), 'wb') as g:
            pickle.dump(seqs_final, g)

        print("Final data shape:", feats_final.shape)

        print("DONE!")

    elif args.mode == '3_create_align_matrix':
        data_path = '/data/cb/rsingh/work/antibody/ci_data/processed/briney_nature_2019/subjects_by_id'
        subject_ids = os.listdir(data_path)
        # subject_ids = ['326797', '326713', '326737']
        subject_ids = ['327059', '326651', '326797', '326713', '326737', '316188']
        subject_ids = ['326651', '326780', '326713', '326797', '327059', '326650', '316188', '326737', '326907']
        print(subject_ids)

        # ----------------------------------------------------------
        # SAMPLE K SEQUENCES FROM EACH SUBJECT IN subject_ids
        seq_samples = []
        unique_seqs = set()
        k = 5000

        for subject in tqdm(subject_ids):
            h = glob.glob(os.path.join(data_path, subject, '{}_seqs.p'.format(subject)))
            if len(h) != 1:
                assert False
            with open(h[0], 'rb') as f:
                subject_seqs = pickle.load(f)

            h = glob.glob(os.path.join(data_path, subject, '{}_seqcount.p'.format(subject)))
            if len(h) != 1:
                assert False
            with open(h[0], 'rb') as f:
                subject_seqcounts = pickle.load(f)

            seq_idxs = list(range(len(subject_seqs)))
            random.shuffle(seq_idxs)
            
            count = 0
            for i in seq_idxs:
                seq = subject_seqs[i]
                if subject_seqcounts[seq] == 1 and seq not in unique_seqs:
                    seq_samples.append((seq, subject))
                    unique_seqs.add(seq)
                    count += 1
                    if count == k:
                        break

        print(len(seq_samples))
        print(seq_samples[:5])
        # ----------------------------------------------------------

        scores_matrix = np.zeros((len(seq_samples), len(seq_samples)))
        from itertools import permutations

        perm = list(permutations(list(range(len(seq_samples))), 2))

        import multiprocessing as mp
        from multiprocessing import get_context

        with get_context('spawn').Pool(processes=args.num_processes, initializer=init_align, initargs=[seq_samples]) as p:
            for result in tqdm(p.imap_unordered(parallel_align, perm, chunksize=args.chunk_size), total=len(perm)):
                idx1, idx2, score = result
                scores_matrix[idx1, idx2] = score


        # SAVE THE MATRIX AND SEQUENCE LIST:
        with h5py.File(os.path.join('../../results/briney_similarity', 
               'align_score_matrix_k{}_{}subjs.h5'.format(k, len(subject_ids))), 'w') as h:
            dset = h.create_dataset('scores', scores_matrix.shape)
            dset[:,:] = scores_matrix

        with open(os.path.join('../../results/briney_similarity', 
               'align_seqs_list_k{}_{}subjs.p'.format(k, len(subject_ids))), 'wb') as f:
            pickle.dump(seq_samples, f)

        print("Saved the files!")

    elif args.mode == '4_create_cossim_matrix':
        data_path = '/data/cb/rsingh/work/antibody/ci_data/processed/briney_nature_2019/subjects_by_id'
        feats_path = '/data/cb/rsingh/work/antibody/ci_data/processed/briney_nature_2019/H_struc_feats_070222.p'
        subject_ids = os.listdir(data_path)
        # subject_ids = ['326797', '326713', '326737']
        # subject_ids = ['327059', '326651', '326797', '326713', '326737', '316188']
        subject_ids = ['326651', '326780', '326713', '326797', '327059', '326650', '316188', '326737', '326907']
        # subject_idxs = [3, 2, 7]
        # subject_idxs = [4, 0, 3, 2, 7, 6]
        subject_idxs = list(range(9))
        print(subject_ids)

        # ----------------------------------------------------------
        # SAMPLE K SEQUENCES FROM EACH SUBJECT IN subject_ids
        seq_samples = []
        unique_seqs = set()
        struc_feats = []
        k = 5000

        for q, subject in tqdm(enumerate(subject_ids), total=len(subject_ids)):
            # load the subject embeddings:
            with open(feats_path, 'rb') as f:
                briney_all_feats, briney_all_counts, briney_all_subj, briney_filenames = pickle.load(f)

            h = glob.glob(os.path.join(data_path, subject, '{}_seqs.p'.format(subject)))
            if len(h) != 1:
                assert False
            with open(h[0], 'rb') as f:
                subject_seqs = pickle.load(f)

            h = glob.glob(os.path.join(data_path, subject, '{}_seqcount.p'.format(subject)))
            if len(h) != 1:
                assert False
            with open(h[0], 'rb') as f:
                subject_seqcounts = pickle.load(f)

            seq_idxs = list(range(len(subject_seqs)))
            random.shuffle(seq_idxs)
            
            count = 0
            for i in seq_idxs:
                seq = subject_seqs[i]
                if subject_seqcounts[seq] == 1 and seq not in unique_seqs:
                    seq_samples.append((seq, subject))
                    unique_seqs.add(seq)

                    struc_feats.append(briney_all_feats[subject_idxs[q]][i])

                    count += 1
                    if count == k:
                        break

        print(len(seq_samples))
        print(seq_samples[:5])
        # ----------------------------------------------------------

        struc_feats = np.stack(struc_feats)
        cossim_matrix = np.matmul(struc_feats, np.transpose(struc_feats))

        for i in range(len(cossim_matrix)):
            cossim_matrix[i, i] = 0

        # SAVE THE MATRIX AND SEQUENCE LIST:
        with h5py.File(os.path.join('../../results/briney_similarity', 
               'cossim_score_matrix_k{}_{}subjs.h5'.format(k, len(subject_ids))), 'w') as h:
            dset = h.create_dataset('scores', cossim_matrix.shape)
            dset[:,:] = cossim_matrix

        with open(os.path.join('../../results/briney_similarity', 
               'cossim_seqs_list_k{}_{}subjs.p'.format(k, len(subject_ids))), 'wb') as f:
            pickle.dump(seq_samples, f)

        print("Saved the files!")

    elif args.mode == '5_evaluate_matrix':

        # subject_ids = ['326797', '326713', '326737']
        # subject_ids = ['327059', '326651', '326797', '326713', '326737', '316188']
        subject_ids = ['326651', '326780', '326713', '326797', '327059', '326650', '316188', '326737', '326907']
        counts_per_subj = []
        k = 5000

        # matrix_name = 'align_score_matrix_k{}_{}subjs.h5'.format(k, len(subject_ids))
        matrix_name = 'cossim_score_matrix_k{}_{}subjs.h5'.format(k, len(subject_ids))

        with h5py.File(os.path.join('../../results/briney_similarity', matrix_name), 'r') as h:
            matrix = h['scores']
            matrix_best_idxs = np.argmax(matrix, axis=1)

            for i in range(0, len(matrix), k):
                count_inside_subj = 0
                for j in range(i, i+k):
                    if i <= matrix_best_idxs[j] < i+k:
                        count_inside_subj += 1
                counts_per_subj.append(count_inside_subj/k)

        print("Evaluate for matrix", matrix_name)
        for i in range(len(subject_ids)):
            print("Ratio of best hit inside subject {}: {}".format(subject_ids[i], counts_per_subj[i]))

    else:
        assert False