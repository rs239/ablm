import os
import argparse
from tqdm import tqdm
import glob
import time
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import pandas as pd

import sys
sys.path.append('..')

import base_config
from utils import get_boolean_mask, find_sequence
from mutate import generate_mutation
from embed import embed_sequence, reload_models_to_device #, load_model
from model import MultiTaskLossWrapper

import torch



class ProteinEmbedding:

    def __init__(self, sequence, chain_type, embed_device=None, embed_model=None, dev=0, fold=0):
        self.seq = sequence
        self.chain_type = chain_type
        self.embedding = None
        self.cdr_mask = None
        self.cdr_embedding = None
        self.embed_device = embed_device
        self.embed_model = embed_model
        self.dev = dev
        self.fold = fold

        assert self.chain_type in ['H', 'L']

    def embed_seq(self, embed_type = "beplerberger"):
        self.embedding = embed_sequence(self.seq, embed_type = embed_type, 
                                        embed_device=self.embed_device, 
                                        embed_model=self.embed_model)

    def create_cdr_mask(self, scheme='chothia', buffer_region = False):
        self.cdr_mask = get_boolean_mask(self.seq, self.chain_type, scheme, buffer_region, self.dev, self.fold)

    def mutate_seq(self, p=0.5, mut_type='cat2'):
        if self.cdr_mask != None:
            return generate_mutation(self.seq, self.cdr_mask, p, mut_type)
        else:
            print("please first create the CDR masks!")
            raise ValueError

    def create_embeds_noaug(self, num_muts = 50, mut_type = 'cat1', embed_type = 'beplerberger'):
        """
        given a sequence of length n, amd embedding type with dimension d,
        return a average mutated embeeding matrix of same shape (n x d)
        """

        mutation_embeds = []

        # print("performing {} mutagenesis of {} mutations...".format(mut_type, num_muts))
        for i in range(num_muts):
            mut_seq = self.mutate_seq(p = 0.1, mut_type = mut_type)
            mut_embed = embed_sequence(mut_seq, embed_type, 
                                        embed_device=self.embed_device, 
                                        embed_model=self.embed_model)
            mutation_embeds.append(mut_embed)

        mutation_embeds = torch.stack(mutation_embeds)
        output = torch.mean(mutation_embeds, dim=0)

        return output

    def create_kmut_matrix(self, num_muts = 50, mut_type = 'cat2', embed_type = 'beplerberger'):
        """
        given a sequence of length n, and embedding type with dimension d,
        return a mutated CDR embedding difference matrix of size (num_muts x n x d)
        """

        mutation_embeds = []

        # print("performing {} mutagenesis of {} mutations...".format(mut_type, num_muts))
        for i in range(num_muts):
            mut_seq = self.mutate_seq(mut_type = mut_type)
            mut_embed = embed_sequence(mut_seq, embed_type,
                                        embed_device=self.embed_device, 
                                        embed_model=self.embed_model)
            mutation_embeds.append(mut_embed)

        mutation_embeds = torch.stack(mutation_embeds)

        return mutation_embeds

    def create_embedding_whole(self, kmut_matrix, verbose=False):
        diff_matrix = self.embedding - kmut_matrix
        avg_kmut = torch.mean(diff_matrix, dim=0)

        cdr_label_all = (self.cdr_mask > 0).float()
        cdr_label_1 = (self.cdr_mask == 1).float()
        cdr_label_2 = (self.cdr_mask == 2).float()
        cdr_label_3 = (self.cdr_mask == 3).float()
        cdr_label = torch.stack([cdr_label_1, cdr_label_2, cdr_label_3, cdr_label_all], dim=-1)

        if verbose:
            return torch.cat((self.embedding.cpu(), avg_kmut.cpu(), cdr_label), dim=-1)
        else:
            out = torch.cat((avg_kmut.cpu(), cdr_label), dim=-1)
            assert out.shape[-1] == 6169
            return out

    def create_cdr_embedding(self, kmut_matrix, sep = False, mask = False):
        """
        given a (k x n x d) k-mutations matrix, compute the CDR embedding 
        by averaging over the k mutation matrices and then concatenating the CDR regions 
        of the sequence --> (n* x d)
        """

        diff_matrix = self.embedding - kmut_matrix
        avg_kmut = torch.mean(diff_matrix, dim=0)

        assert avg_kmut.shape[0] == len(self.cdr_mask)

        cdr1, cdr2, cdr3 = [], [], []
        for i, cdr in enumerate(self.cdr_mask):
            if cdr == 1:
                cdr1.append(i)
            elif cdr == 2:
                cdr2.append(i)
            elif cdr == 3:
                cdr3.append(i)

        out = None
        cdr_idxs = cdr1 + cdr2 + cdr3
        if sep:
            emb_dim = avg_kmut.shape[-1]
            out = torch.cat((avg_kmut[cdr1, :].cuda(), torch.zeros(1, emb_dim).cuda(),
                              avg_kmut[cdr2, :].cuda(), torch.zeros(1, emb_dim).cuda(),
                              avg_kmut[cdr3, :].cuda()), dim=0)
        else:
            out = avg_kmut[cdr_idxs, :]

        if mask:
            if sep:
                print("Cannot use mask augmentions when there are separator tokens.")
                raise ValueError
            cdr_mask_loc = self.cdr_mask[cdr_idxs]
            cdr_label_all = (cdr_mask_loc > 0).float()
            cdr_label_1 = (cdr_mask_loc == 1).float()
            cdr_label_2 = (cdr_mask_loc == 2).float()
            cdr_label_3 = (cdr_mask_loc == 3).float()
            cdr_label = torch.stack([cdr_label_1, cdr_label_2, cdr_label_3, cdr_label_all], dim=-1)

            out = torch.cat((out.cpu(), cdr_label), dim=-1)

        self.cdr_embedding = out

        return out

    def create_cdr_embedding_nomut(self, sep = False):
        """
        splice the CDR regions of the embedding together with no mutation adjustments
        if sep is True, tensor of 0's are inserted as pad's between each CDR region
        """

        result_matr = self.embedding.detach().clone()

        assert result_matr.shape[0] == len(self.cdr_mask)

        cdr1, cdr2, cdr3 = [], [], []
        for i, cdr in enumerate(self.cdr_mask):
            if cdr == 1:
                cdr1.append(i)
            elif cdr == 2:
                cdr2.append(i)
            elif cdr == 3:
                cdr3.append(i)

        if sep:
            emb_dim = result_matr.shape[-1]
            return torch.cat((result_matr[cdr1, :].cuda(), torch.zeros(1, emb_dim).cuda(),
                              result_matr[cdr2, :].cuda(), torch.zeros(1, emb_dim).cuda(),
                              result_matr[cdr3, :].cuda()), dim=0)
        else:
            cdr_idxs = cdr1 + cdr2 + cdr3
            return result_matr[cdr_idxs, :]


    def create_cdr_specific_embedding(self, embed_type, k, seperator = False, mask = True):
        """
        Create a CDR-specific embedding directly from sequence
        embed_type : embed with which general purpose PLM? (i.e. beplerberger)
        k : number of mutant sequences you'd wish to generate
        seperator : separate the CDR regions with 0-tensor tokens? (default: False)
        mask : Appending a mask that indicates which CDR a residue belongs to (1, 2, 3) (default: True)
        """

        self.embed_seq(embed_type = embed_type)

        self.create_cdr_mask()

        kmut_matr_h = self.create_kmut_matrix(num_muts=k, embed_type=embed_type)
        cdr_embed = self.create_cdr_embedding(kmut_matr_h, sep = seperator, mask = mask)

        return cdr_embed



def get_valid_ids():
    f = open('/data/cb/rsingh/work/antibody/ci_data/raw/sabdab_all/valid_ids.txt', 'w')
    valid_list = []
    path = '/data/cb/rsingh/work/antibody/ci_data/raw/sabdab_all/sequences'
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
    f.close()
            

def parallelize(pdb_id, c_type, emb_type='beplerberger'):
    out_path = "/data/cb/rsingh/work/antibody/ci_data/processed/sabdab_all/cdr_embeddings_new"
    out_folder = os.path.join(out_path, emb_type)

    k = 100

    seq_h, seq_l = find_sequence(pdb_id = pdb_id)

    seq = seq_h if c_type == 'H' else seq_l

    prot_embed = ProteinEmbedding(seq, c_type)
    prot_embed.embed_seq(embed_type = emb_type)
    prot_embed.create_cdr_mask()

    kmut_matr = prot_embed.create_kmut_matrix(num_muts=k, embed_type = emb_type)
    cdr_embed = prot_embed.create_cdr_embedding(kmut_matr, sep=True)
    f = open(os.path.join(out_folder, 'sabdab_{}_{}_{}_k{}.p'.format(pdb_id, 'cat2', prot_embed.chain_type, k)), 'wb')
    pickle.dump(cdr_embed, f)
    f.close()

    del prot_embed, cdr_embed, kmut_matr




def main_libra(args, orig_embed=False):
    reload_models_to_device(args.device_num)

    # seqs_path = "/data/cb/rsingh/work/antibody/ci_data/raw/libraseq/libraseq_standardized.csv"
    seqs_path = "/data/cb/rsingh/work/antibody/ci_data/raw/libraseq/libraseq_standardized_Set1.csv"
    out_folder = '/data/cb/rsingh/work/antibody/ci_data/processed/libraseq/cdrembed_maskaug4'
    # out_folder = '/data/cb/rsingh/work/antibody/ci_data/processed/libraseq/original_embeddings'

    df = pd.read_csv(seqs_path, keep_default_na=False)
    ids_to_drop = []

    k = 100
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        # if index == 10:
        #     break
        seq_h, seq_l = row['vh_seq'], row['vl_seq']

        for seq, chain_type in [(seq_h, 'H'), (seq_l, 'L')]:
            prot_embed = ProteinEmbedding(seq, chain_type, dev=args.device_num, fold='y')
            p_id = row['id']+'_cat2_{}_k{}.p'.format(chain_type, k)

            for model_typ in ['esm1b']:
                out_path = os.path.join(out_folder, model_typ)
                if not os.path.isdir(out_path):
                    os.mkdir(out_path)

                # check if file exists:
                # if os.path.exists(os.path.join(out_path, p_id)):
                #     continue

                try:
                    prot_embed.embed_seq(embed_type = model_typ)

                    if orig_embed is True:
                        out_folder = "/data/cb/rsingh/work/antibody/ci_data/processed/libraseq/original_embeddings"
                        out_path = os.path.join(out_folder, model_typ)

                        file_name = '{}_{}_orig.p'.format(row['id'], prot_embed.chain_type)
                        with open(os.path.join(out_path, file_name), 'wb') as fh:
                            print("Saving", row['id'])
                            pickle.dump(prot_embed.embedding, fh)
                        continue

                    # cdr_embed = prot_embed.embedding     

                    prot_embed.create_cdr_mask()

                    kmut_matr = prot_embed.create_kmut_matrix(num_muts=k, embed_type=model_typ)
                    cdr_embed = prot_embed.create_cdr_embedding(kmut_matr, sep = False, mask = True)

                    # --------------------------------------------------
                    # TRYING TO CONCATENATE NOMUT! (PROTBERT ONLY)
                    # cdr_embed_nomut = prot_embed.create_cdr_embedding_nomut().cuda(args.device_num)
                    # cdr_embed = torch.cat((cdr_embed_nomut, cdr_embed.cuda(args.device_num)), dim=-1)
                    # --------------------------------------------------


                    with open(os.path.join(out_path, p_id), 'wb') as f:
                        pickle.dump(cdr_embed, f)

                except:
                    ids_to_drop.append(index)


    ids_to_drop = list(set(ids_to_drop))
    print("Seqs from these indices did not work...")
    print(ids_to_drop)


def main_sabdab(args, orig_embed = False):

    pdb_ids_path = "/data/cb/rsingh/work/antibody/ci_data/processed/sabdab_pure_042522/valid_ids_all.txt"
    with open(pdb_ids_path, 'r') as f:
        pdb_ids = f.read().splitlines()
    print(len(pdb_ids))

    out_folder = "/data/cb/rsingh/work/antibody/ci_data/processed/sabdab_pure_042522/cdrembed_maskaug4"
    # out_folder = "/data/cb/rsingh/work/antibody/ci_data/processed/sabdab_pure_042522/original_embeddings"

    # embed_type = 'beplerberger'
    # embed_type = 'protbert'
    embed_type = 'esm1b'
    # embed_type = 'tape'

    reload_models_to_device(args.device_num)

    k=100
    for c_type in ['H', 'L']:
        for pdb_id in tqdm(pdb_ids):

            # file_name = 'sabdab_{}_{}_{}_k{}.p'.format(pdb_id, 'cat2', c_type, k)
            # if os.path.exists(os.path.join(out_folder, embed_type, file_name)):
            #     continue


            seq_h, seq_l = find_sequence(dataset='sabdab_pure', pdb_id = pdb_id)
            seq = seq_h if c_type == 'H' else seq_l
            prot_embed = ProteinEmbedding(seq, c_type, fold='x')

            out_path = os.path.join(out_folder, embed_type)
            if not os.path.isdir(out_path):
                os.mkdir(out_path)

            prot_embed.embed_seq(embed_type = embed_type)

            if orig_embed is True:
                out_folder = "/data/cb/rsingh/work/antibody/ci_data/processed/sabdab_pure_042522/original_embeddings"
                out_path = os.path.join(out_folder, embed_type)

                file_name = 'sabdab_{}_{}_orig.p'.format(pdb_id, prot_embed.chain_type)
                with open(os.path.join(out_path, file_name), 'wb') as fh:
                    print("Saving", pdb_id)
                    pickle.dump(prot_embed.embedding, fh)
                continue

            # cdr_embed = prot_embed.embedding

            try:
                prot_embed.create_cdr_mask()
            except:
                print("pdb id {} didn't work...".format(pdb_id))
                continue

            kmut_matr_h = prot_embed.create_kmut_matrix(num_muts=k, embed_type=embed_type)
            cdr_embed = prot_embed.create_cdr_embedding(kmut_matr_h, sep = False, mask = True)

            # --------------------------------------------------
            # TRYING TO CONCATENATE NOMUT! (PROTBERT ONLY)
            # cdr_embed_nomut = prot_embed.create_cdr_embedding_nomut().cuda(args.device_num)
            # cdr_embed = torch.cat((cdr_embed_nomut, cdr_embed.cuda(args.device_num)), dim=-1)
            # --------------------------------------------------

            file_name = 'sabdab_{}_{}_{}_k{}.p'.format(pdb_id, 'cat2', c_type, k)
            with open(os.path.join(out_path, file_name), 'wb') as fh:
                print("Saving", pdb_id)
                pickle.dump(cdr_embed, fh)
        



def create_covabdab_embeddings(args):
    reload_models_to_device(args.device_num)

    df = pd.read_csv("/data/cb/rsingh/work/antibody/ci_data/raw/covabdab/neutralize_sarscov2.csv")
    device = torch.device(base_config.device if torch.cuda.is_available() else "cpu")
    out_dir = "/data/cb/rsingh/work/antibody/ci_data/processed/covabdab/"
    chain_type = 'H'

    # load valid ids:
    with open('/data/cb/rsingh/work/antibody/ci_data/processed/covabdab/valid_ids.txt', 'r') as f:
        valid_ids_list = f.read().splitlines()

    k=100
    invalids = []

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        # if index < 1710:
        #     continue

        seq_id = row['Name']
        if seq_id not in valid_ids_list:
            continue

        seq = row['VH or VHH'] if chain_type == 'H' else row['VL']

        if ' ' in seq:
            invalids.append(seq_id)
            continue

        prot_embed = ProteinEmbedding(seq, chain_type, dev=args.device_num)
        try:
            prot_embed.create_cdr_mask()
            # print(prot_embed.cdr_mask)
            # print("HAW"+prot_embed.seq+"HEE")
        except:
            invalids.append(seq_id)
            continue

        for embed_type in ['beplerberger']:
            out_path = os.path.join(out_dir, 'cdrembed_maskaug4', embed_type)
            prot_embed.embed_seq(embed_type=embed_type)
            kmut_matr_h = prot_embed.create_kmut_matrix(num_muts=k, embed_type=embed_type)
            cdr_embed = prot_embed.create_cdr_embedding(kmut_matr_h, sep=False, mask=True)

            # save the embedding first:
            with open(os.path.join(out_path, "{}_cat2_{}_k{}.p".format(seq_id, chain_type, k)), 'wb') as fc:
                pickle.dump(cdr_embed, fc)

    print(invalids)




def main_covabdab(args):
    reload_models_to_device(args.device_num)

    df = pd.read_csv("/data/cb/rsingh/work/antibody/ci_data/raw/covabdab/neutralize_sarscov2.csv")
    device = torch.device(base_config.device if torch.cuda.is_available() else "cpu")
    covabdab_dir = "/data/cb/rsingh/work/antibody/ci_data/processed/covabdab"
    embed_dir = "/data/cb/rsingh/work/antibody/ci_data/processed/covabdab/cdrembed_maskaug4/{}".format(args.embed_type)
    out_dir = "/data/cb/rsingh/work/antibody/ci_data/processed/covabdab/covabdab_features_120122/{}".format(args.embed_type)

    # Load our pretrained AbNet Model
    from model import AntibodyNetMultiAttn
    model = AntibodyNetMultiAttn(embed_dim=2200, mid_dim2=1024, mid_dim3=512,
                                      proj_dim=252, num_enc_layers=1, num_heads=16).to(device)
    pretrained_path = "../model_ckpts/091522_bb_newnum_{}/beplerberger_epoch50.pt".format(args.chain_type)
    checkpoint = torch.load(pretrained_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Loaded the pre-trained model!")

    k=100

    with open(os.path.join(covabdab_dir, 'invalids.txt'), 'r') as f:
        invalids_list = f.read().splitlines()

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        seq_id, seq_h = row['Name'], row['VH or VHH']

        # if the sequence id is invalid, skip
        if seq_id in invalids_list:
            continue

        # prot_embed = ProteinEmbedding(seq_h, 'H', dev=1)
        # prot_embed.create_cdr_mask()

        # --------------------------------------
        # FOR PROTBERT ONLY:

        # embed_type = args.embed_type
        # prot = ProteinEmbedding(seq_h, 'H')
        # try:
        #     prot.embed_seq(embed_type = args.embed_type)
        # except:
        #     continue
        # with open(os.path.join(out_dir, embed_type, '{}_H_orig.p'.format(seq_id)), 'wb') as f:
        #     pickle.dump(prot.embedding, f)
        # continue
        # --------------------------------------

        # load the embedding:
        with open(os.path.join(embed_dir, "{}_cat2_H_k100.p".format(seq_id)), 'rb') as p:
            cdr_embed = pickle.load(p)

        cdr_embed = cdr_embed.to(device)
        cdr_embed = torch.unsqueeze(cdr_embed, dim=0) # create the batch dimension (singleton)

        # create and save STRUCTURE specific feature:
        with torch.no_grad():
            feat_task, _ = model(cdr_embed, cdr_embed, None, None, 0, task_specific=True)
        assert feat_task.shape[-1] == 512
        with open(os.path.join(out_dir, "{}_struc_H.p".format(seq_id)), 'wb') as p:
            pickle.dump(torch.squeeze(feat_task), p)

        # create and save FUNCTION specific feature:
        with torch.no_grad():
            feat_task, _ = model(cdr_embed, cdr_embed, None, None, 1, task_specific=True)
        assert feat_task.shape[-1] == 512
        with open(os.path.join(out_dir, "{}_func_H.p".format(seq_id)), 'wb') as p:
            pickle.dump(torch.squeeze(feat_task), p)

        # create and save INTERMEDIATE n' x k feature, before Transformer:
        with torch.no_grad():
            feat_interm, _ = model(cdr_embed, cdr_embed, None, None, None, return2=True)
        assert feat_interm.shape[-1] == 256
        with open(os.path.join(out_dir, "{}_interm_H.p".format(seq_id)), 'wb') as p:
            pickle.dump(torch.squeeze(feat_interm), p)



def make_sabdab_features(args):
    reload_models_to_device(args.device_num)

    device = torch.device(base_config.device if torch.cuda.is_available() else "cpu")
    
    chain_type = args.chain_type
    out_dir = "/data/cb/rsingh/work/antibody/ci_data/processed/sabdab_pure_042522/set1_features"
    embed_path = "/data/cb/rsingh/work/antibody/ci_data/processed/sabdab_pure_042522/cdrembed_maskaug4/beplerberger"
    pdb_ids_path = "/data/cb/rsingh/work/antibody/ci_data/processed/sabdab_pure_042522/valid_ids_set1.txt"
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


def temp2(args):
    reload_models_to_device(0)
    device = torch.device(0 if torch.cuda.is_available() else "cpu")
    
    chain_type = 'H'
    out_dir = "/data/cb/rsingh/work/antibody/ci_data/processed/libraseq"
    embed_path = "/data/cb/rsingh/work/antibody/ci_data/processed/libraseq/cdrembed_maskaug4/beplerberger"

    pretrained = AntibodyNetMulti(embed_dim=2200, lstm_layers=1).to(device)
    best_arch_path = '/net/scratch3.mit.edu/scratch3-3/chihoim/model_ckpts/best_pretrained'
    best_path = os.path.join(best_arch_path, 'multi_{}_best.pt'.format(chain_type))
    checkpoint = torch.load(best_path, map_location=device)
    pretrained.load_state_dict(checkpoint['model_state_dict'])
    pretrained.eval()


    feats = []
    for file_name in tqdm(os.listdir(embed_path)):
        if '_H_' in file_name:
            with open(os.path.join(embed_path, file_name), 'rb') as f:
                prot_emb = pickle.load(f)

            cdr_embed = torch.unsqueeze(prot_emb, dim=0) # create the batch dimension (singleton)
            cdr_embed_batch = cdr_embed.to(device)

            with torch.no_grad():
                x, x_pos = cdr_embed_batch[:,:,-2204:-4], cdr_embed_batch[:,:,-4:]
                x = pretrained.project(x)
                x = torch.cat([x, x_pos], dim=-1)
                feat = torch.squeeze(pretrained.recurrent(x, 0)).detach().cpu().numpy()

            feats.append(feat)

    feats = np.stack(feats, axis=0)
    print("Features Shape:", feats.shape)
    with open(os.path.join(out_dir, "libraseq_{}_struc.p".format(chain_type)), 'wb') as f:
        pickle.dump(feats, f)

def save_sabdab_cdrs(args):

    pdb_ids_path = "/data/cb/rsingh/work/antibody/ci_data/processed/sabdab_pure_042522/valid_ids_all.txt"
    with open(pdb_ids_path, 'r') as f:
        pdb_ids = f.read().splitlines()

    results_dict_H, results_dict_L = dict(), dict()
    for pdb_id in tqdm(pdb_ids):
        seq_h, seq_l = find_sequence(dataset='sabdab_pure', pdb_id=pdb_id)

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

    with open('/data/cb/rsingh/work/antibody/ci_data/processed/sabdab_pure_042522/sabdab_cdrH_strs.p', 'wb') as p:
        pickle.dump(results_dict_H, p)
    with open('/data/cb/rsingh/work/antibody/ci_data/processed/sabdab_pure_042522/sabdab_cdrL_strs.p', 'wb') as p:
        pickle.dump(results_dict_L, p)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--embed_type', type=str, default='beplerberger',
                        help='Type of model to embed proten sequence with.')

    parser.add_argument('--num_proc', type=int, default=4,
                        help='Number of cores to use for parallel processing')

    parser.add_argument('--device_num', type=int, default=0,
                        help='GPU Device number.')

    parser.add_argument('--chain_type', type=str, default='',
                        help='input string of protein sequence.')

    parser.add_argument('--output_path', type=str, default='.',
                        help='path for the output file of embeddings.')

    args = parser.parse_args()
    # base_config.device = 'cuda:0'

    if torch.cuda.is_available():
        print("Using the GPU!")
    else:
        print("WARNING: Could not find GPU! Using CPU only")

    # get_valid_ids()
    # main_libra(args, orig_embed=False)
    # main_sabdab(args, orig_embed=True)
    # create_covabdab_embeddings(args)
    main_covabdab(args)
    # make_sabdab_features(args)
    # save_sabdab_cdrs(args)
