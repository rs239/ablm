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
    
    # global pretrained
    # chain_type = 'H'
    # pretrained = AbMAP(embed_dim=2200, lstm_layers=1).to(device)
    # best_arch_path = '/net/scratch3.mit.edu/scratch3-3/chihoim/model_ckpts/best_pretrained'
    # best_path = os.path.join(best_arch_path, 'multi_{}_best.pt'.format(chain_type))
    # checkpoint = torch.load(best_path, map_location=device)
    # pretrained.load_state_dict(checkpoint['model_state_dict'])
    # pretrained.eval()

    reload_models_to_device(device_n)

def work_parallel(pair):
    dev, seq = pair
    prot = ProteinEmbedding(seq, 'H', embed_device="cuda:{}".format(device_n), dev=dev, fold=device_n)

    try:
        prot.create_cdr_mask()
    except:
        return (None, None)

    k=50
    prot.embed_seq('beplerberger')
    kmut_matr_h = prot.create_kmut_matrix(num_muts=k, embed_type='beplerberger')
    cdr_embed = prot.create_cdr_embedding(kmut_matr_h, sep = False, mask = True)

    return (cdr_embed.detach().cpu().numpy(), seq)

    # cdr_embed_batch = torch.unsqueeze(cdr_embed, dim=0).to(device_n)
    # with torch.no_grad():
    #     x, x_pos = cdr_embed_batch[:,:,-2204:-4], cdr_embed_batch[:,:,-4:]
    #     x = pretrained.project(x)
    #     x = torch.cat([x, x_pos], dim=-1)
    #     feat = torch.squeeze(pretrained.recurrent(x, 0)).detach().cpu().numpy()

    # return (feat, seq)


def init_align(seq_samples):
    global seqs_list
    seqs_list = seq_samples

def parallel_align(pair_idxs):
    idx1, idx2 = pair_idxs
    seq1, seq2 = seqs_list[idx1][0], seqs_list[idx2][0]
    # print("seq1", seq1)
    # print("seq2", seq2)
    align_score = pairwise2.align.globalxx(seq1, seq2, score_only=True)
    return idx1, idx2, align_score

def unpad_embedding(embedding):
    # embedding should be in the dimension of (1 x n' x d)
    
    v = np.abs(embedding).sum(axis=2)
    v2 = 1*(v < 1e-8)
    v3 = np.argmax(v2, axis=1)[0]
    
    return v3

def predict_count_attn(args):

    from model import AbMAPAttn, BrineyPredictorAttn
    from sklearn import utils

    device = torch.device("cuda:{}".format(args.device_num) if torch.cuda.is_available() else "cpu")

    # Load our pretrained AbNet Model
    pretrained = AbMAPAttn(embed_dim=2200, mid_dim2=1024, mid_dim3=512,
                                      proj_dim=252, num_enc_layers=1, num_heads=16).to(device)
    chain_type = 'H'
    best_arch_path = '/net/scratch3.mit.edu/scratch3-3/chihoim/model_ckpts/best_pretrained'
    best_path = os.path.join(best_arch_path, 'bb_feat256_{}_best_newnum.pt'.format(chain_type))
    checkpoint = torch.load(best_path, map_location=device)
    pretrained.load_state_dict(checkpoint['model_state_dict'])
    pretrained.eval()
    print("Loaded the pre-trained model!")

    # HYPERPARAMETERS:
    num_epochs = 200
    batch_size = 32
    lr = 5e-4
    save_every = 50

    # Build Data Loader
    # briney_root = "/data/cb/rsingh/work/antibody/ci_data/processed/briney_nature_2019/subjects_by_id"
    # labels_total = []
    # ids_list = os.listdir(briney_root)
    # print(ids_list)
    # idx_count = 0
    # with h5py.File("/data/cb/rsingh/work/antibody/ci_data/processed/briney_nature_2019/all_subjs.h5", "w") as f:
    #     dset_f = f.create_dataset('feats', (885422, 60, 256))
    #     dset_l = f.create_dataset('labels', (885422,))

    #     for id_ in ids_list:
    #         print("Processing subject {} ...".format(id_))
    #         g = glob.glob(os.path.join(briney_root, id_, '{}_*data.h5'.format(id_)))
    #         if len(g) != 1:
    #             assert False
    #         data_path = g[0]
    #         with h5py.File(data_path, 'r') as h:
    #             labels = h['labels'][:]
    #             labels_total.append(labels)

    #             # for i in tqdm(range(1000)):
    #             for i in tqdm(range(h['feats'].shape[0])):
    #                 feature_np = h['feats'][i:i+1]
    #                 unpad_idx = unpad_embedding(feature_np)
    #                 if unpad_idx == 0:
    #                     feature_tensor = torch.Tensor(feature_np).to(device)
    #                 else:
    #                     feature_tensor = torch.Tensor(feature_np[:, :unpad_idx, :]).to(device)

    #                 with torch.no_grad():
    #                     feature_tensor, _ = pretrained(feature_tensor, feature_tensor, None, None, None, return2=True)
    #                 feature_tensor = torch.squeeze(feature_tensor, dim=0)

    #                 feature_np = feature_tensor.detach().cpu().numpy()
    #                 dset_f[idx_count, :unpad_idx, :] = feature_np

    #                 idx_count += 1


    #     labels_total = np.concatenate(labels_total, axis=0)
    #     dset_l[:] = labels_total

    # assert idx_count == len(labels_total)
    # print("Total of {} data points saved".format(idx_count))

    # return

    # data_path = '/data/cb/rsingh/work/antibody/ci_data/processed/briney_nature_2019/subjects_by_id/326651/326651_n99623_data.h5'
    data_path = '/data/cb/rsingh/work/antibody/ci_data/processed/briney_nature_2019/all_subjs.h5'
    with h5py.File(data_path, 'r') as h:
        feats, labels = h['feats'][:], h['labels'][:]
        labels = np.log(labels)
        labels = (labels > 1e-5).astype(float) # making a binary label
        feats, labels = utils.shuffle(feats, labels)
        print("converted h5py to numpy!")

    train_idx, val_idx = 700000, 790000
    train_loader_feat = DataLoader(feats[:train_idx], batch_size=batch_size)
    train_loader_label = DataLoader(labels[:train_idx], batch_size=batch_size)
    val_loader_feat = DataLoader(feats[train_idx:val_idx], batch_size=batch_size)
    val_loader_label = DataLoader(labels[train_idx:val_idx], batch_size=batch_size)
    test_loader_feat = DataLoader(feats[val_idx:], batch_size=batch_size)
    test_loader_label = DataLoader(labels[val_idx:], batch_size=batch_size)
    print("Created Dataloaders!")

    # Construct Briney Predictor, Loss Func, Optimizer
    model = BrineyPredictorAttn(input_dim = 256, mid_dim1 = 64, mid_dim2 = 16, mid_dim3 = 4).to(device)
    # loss_fn = nn.MSELoss()
    loss_fn = nn.BCELoss()
    optimizer = opt.AdamW(model.parameters(), lr=lr)

    for i in range(num_epochs):
        # TRAINING!
        model.train()
        total_loss = 0
        for batch_feats, batch_label in tqdm(zip(train_loader_feat, train_loader_label), total=len(train_loader_label)):
            batch_feats = batch_feats.float().to(device)
            batch_label = batch_label.float().to(device)

            # create mask for transformer
            batch_feats_mask0 = torch.zeros(batch_feats.shape[:2]).to(device)
            batch_feats_mask1 = torch.ones(batch_feats.shape[:2]).to(device)
            batch_feats_mask = torch.where(torch.sum(batch_feats, dim=-1) == 0, batch_feats_mask1, batch_feats_mask0)

            # with torch.no_grad():
            #     batch_feats, _ = pretrained(batch_feats, batch_feats, batch_feats_mask, batch_feats_mask, 0, return3=True)

            batch_pred = model(batch_feats, batch_feats_mask)

            # update the parameters (loss, backprop, etc.)
            loss = loss_fn(batch_pred, batch_label)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print("Epoch {} Average Training Loss: {}".format(i+1, total_loss/len(train_loader_feat)))

        # VALIDATION!
        model.eval()
        total_loss = 0
        for batch_feats, batch_label in tqdm(zip(val_loader_feat, val_loader_label), total=len(val_loader_label)):
            batch_feats = batch_feats.float().to(device)
            batch_label = batch_label.float().to(device)

            # create mask for transformer
            batch_feats_mask0 = torch.zeros(batch_feats.shape[:2]).to(device)
            batch_feats_mask1 = torch.ones(batch_feats.shape[:2]).to(device)
            batch_feats_mask = torch.where(torch.sum(batch_feats, dim=-1) == 0, batch_feats_mask1, batch_feats_mask0)

            with torch.no_grad():
                # batch_feats, _ = pretrained(batch_feats, batch_feats, batch_feats_mask, batch_feats_mask, 0, return3=True)
                batch_pred = model(batch_feats, batch_feats_mask)

            loss = loss_fn(batch_pred, batch_label)
            total_loss += loss.item()
        print("Epoch {} Average Validation Loss: {}".format(i+1, total_loss/len(val_loader_feat)))
        print('-'*30)


        # saving models:
        if (i+1)%save_every == 0:
            model_savepath = "/net/scratch3.mit.edu/scratch3-3/chihoim/model_ckpts/081322_briney_pred_clas"
            raw_model = model.module if hasattr(model, "module") else model
            model_sum = {'model_state_dict': raw_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}
            savepath = os.path.join(model_savepath, 'briney_predictor_{}epochs.pt'.format(i+1))
            torch.save(model_sum, savepath)

    print('='*30)
    total_loss = 0
    for batch_feats, batch_label in tqdm(zip(test_loader_feat, test_loader_label), total=len(test_loader_label)):
        batch_feats = batch_feats.float().to(device)
        batch_label = batch_label.float().to(device)

        # create mask for transformer
        batch_feats_mask0 = torch.zeros(batch_feats.shape[:2]).to(device)
        batch_feats_mask1 = torch.ones(batch_feats.shape[:2]).to(device)
        batch_feats_mask = torch.where(torch.sum(batch_feats, dim=-1) == 0, batch_feats_mask1, batch_feats_mask0)

        with torch.no_grad():
            # batch_feats, _ = pretrained(batch_feats, batch_feats, batch_feats_mask, batch_feats_mask, 0, return3=True)
            batch_pred = model(batch_feats, batch_feats_mask)

        loss = loss_fn(batch_pred, batch_label)
        total_loss += loss.item()
    print("Average Test Loss: {}".format(total_loss/len(test_loader_feat)))


def predict_count(args, is_regression):

    from model import AbMAPAttn, BrineyPredictorAttn
    device = torch.device("cuda:{}".format(args.device_num) if torch.cuda.is_available() else "cpu")

    # Load all Briney subjects into one Tensor
    print("packing Briney subjects data into single dataset...")
    briney_root = '/data/cb/rsingh/work/antibody/ci_data/processed/briney_nature_2019'
    with open(os.path.join(briney_root, 'H_interm_feats_081622.p'), 'rb') as f:
        briney_all_feats, briney_all_counts, briney_all_subj, briney_filenames = pickle.load(f)
    briney_all_feats_np = np.concatenate(briney_all_feats, axis=0)
    briney_all_counts_np = np.concatenate(briney_all_counts, axis=0)


    # briney_all_counts_np = np.exp(briney_all_counts_np)

    # for CLASSIFICATION ONLY
    # briney_all_counts_np = (briney_all_counts_np > 1e-5).astype(float)
    # print("Number of counts > 1:", np.sum(briney_all_counts_np))


    dataset_all = np.concatenate([briney_all_feats_np, briney_all_counts_np[:, np.newaxis]], axis=-1)
    np.random.shuffle(dataset_all)
    print(briney_all_feats_np.shape, briney_all_counts_np.shape, dataset_all.shape)
    train_size, val_size = int(0.7*len(briney_all_counts_np)), int(0.15*len(briney_all_counts_np))
    test_size = len(briney_all_counts_np) - train_size - val_size
    

    if is_regression == True:
        # build regression model
        from sklearn.linear_model import LinearRegression, LogisticRegression, PoissonRegressor, GammaRegressor
        reg_train, reg_test = dataset_all[:train_size+val_size], dataset_all[train_size+val_size:]
        reg_train_data, reg_train_label = reg_train[:,:-1], reg_train[:,-1]
        reg_test_data, reg_test_label = reg_test[:,:-1], reg_test[:,-1]
        print(reg_train_data.shape, reg_test_data.shape)

        # linreg = LinearRegression().fit(reg_train_data, reg_train_label)
        linreg = PoissonRegressor().fit(reg_train_data, reg_train_label)
        reg_test_pred = linreg.predict(reg_test_data)
        spear = evaluate_spearman(reg_test_pred, reg_test_label)
        reg_score = linreg.score(reg_test_data, reg_test_label)
        print("Spearman Rank Score for Regression Test Set:", spear)
        print("Coefficient of Determination for Regression Test Set:", reg_score)

        # logreg = LogisticRegression(class_weight='balanced', C=0.5).fit(reg_train_data, reg_train_label)
        # reg_score = logreg.score(reg_test_data, reg_test_label)
        # print("Mean accuracy on the Test set:", reg_score)

        return


    # HYPERPARAMETERS:
    num_epochs = 500
    batch_size = 32
    lr = 5e-4
    save_every = 100

    # build Briney Predictor, Loss, Optimizer:
    model = BrineyPredictor(input_dim = 252, mid_dim1 = 64, mid_dim2 = 16, mid_dim3 = 4).to(device)
    loss_fn = nn.MSELoss()
    # loss_fn = nn.BCELoss()
    optimizer = opt.AdamW(model.parameters(), lr=lr)

    print("creating dataloaders...")
    # create train/test dataloders
    train_set, valid_set, test_set = data.random_split(dataset_all, [train_size, val_size, test_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    for i in range(num_epochs):
        # TRAINING!
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader):
            batch = batch.float().to(device)
            batch_feats, batch_label = batch[:, :-1], batch[:, -1]
            # print(type(batch), type(batch_feats), type(batch_label))
            batch_pred = model(batch_feats)

            # update the parameters (loss, backprop, etc.)
            loss = loss_fn(batch_pred, batch_label)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print("Epoch {} Average Training Loss: {}".format(i+1, total_loss/len(train_loader)))

        # VALIDATION!
        model.eval()
        total_loss = 0
        for batch in tqdm(valid_loader):
            batch = batch.float().to(device)
            batch_feats, batch_label = batch[:, :-1], batch[:, -1]
            with torch.no_grad():
                batch_pred = model(batch_feats)

            loss = loss_fn(batch_pred, batch_label)
            total_loss += loss.item()
        print("Epoch {} Average Validation Loss: {}".format(i+1, total_loss/len(valid_loader)))
        print('-'*30)


        # saving models:
        if (i+1)%save_every == 0:
            model_savepath = "/net/scratch3.mit.edu/scratch3-3/chihoim/model_ckpts/081122_briney_pred"
            # model_savepath = "/net/scratch3.mit.edu/scratch3-3/chihoim/model_ckpts/081322_briney_pred_clas"
            raw_model = model.module if hasattr(model, "module") else model
            model_sum = {'model_state_dict': raw_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}
            savepath = os.path.join(model_savepath, 'briney_predictor_{}epochs.pt'.format(i+1))
            torch.save(model_sum, savepath)

    print('='*30)
    total_loss = 0
    for batch in tqdm(test_loader):
        batch = batch.float().to(device)
        batch_feats, batch_label = batch[:, :-1], batch[:, -1]
        with torch.no_grad():
            batch_pred = model(batch_feats)

        loss = loss_fn(batch_pred, batch_label)
        total_loss += loss.item()
    print("Average Test Loss: {}".format(total_loss/len(test_loader)))



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
        # from multiprocessing import set_start_method
        # set_start_method('spawn')
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


    elif args.mode == '6_predict_count':
        # predict_count(args, is_regression=False)
        predict_count_attn(args)


    else:
        assert False