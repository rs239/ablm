import os
import sys
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
import torch
import h5py

from abmap.utils import get_boolean_mask, find_sequence, parse, log
from abmap.mutate import generate_mutation
from abmap.plm_embed import embed_sequence, reload_models_to_device


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

    def create_cdr_embedding_nomut(self, embed_type, sep = False, mask=False):
        """
        splice the CDR regions of the embedding together with no mutation adjustments
        if sep is True, tensor of 0's are inserted as pad's between each CDR region
        """
        self.embed_seq(embed_type = embed_type)

        self.create_cdr_mask()

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
            out_embed = torch.cat((result_matr[cdr1, :].cuda(), torch.zeros(1, emb_dim).cuda(),
                              result_matr[cdr2, :].cuda(), torch.zeros(1, emb_dim).cuda(),
                              result_matr[cdr3, :].cuda()), dim=0)
        else:
            cdr_idxs = cdr1 + cdr2 + cdr3
            out_embed = result_matr[cdr_idxs, :]

        out_embed = out_embed.detach().cpu()

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

            out_embed = torch.cat((out_embed.cpu(), cdr_label), dim=-1)

        return out_embed


    def create_cdr_specific_embedding(self, embed_type, k=100, separator = False, mask = True):
        """
        Create a CDR-specific embedding directly from sequence
        embed_type : embed with which general purpose PLM? (i.e. beplerberger)
        k : number of mutant sequences you'd wish to generate
        separator : separate the CDR regions with 0-tensor tokens? (default: False)
        mask : Appending a mask that indicates which CDR a residue belongs to (1, 2, 3) (default: True)
        """

        self.embed_seq(embed_type = embed_type)

        self.create_cdr_mask()

        kmut_matr_h = self.create_kmut_matrix(num_muts=k, embed_type=embed_type)
        cdr_embed = self.create_cdr_embedding(kmut_matr_h, sep = separator, mask = mask)

        return cdr_embed
            

def augment_from_fasta(fastaPath, outputPath, chain_type, embed_type,
                       num_mutations, separators, cdr_masks,
                       device=0, verbose=False):
    """
    Embed sequences with existing PLMs and augment with mutagenesis & CDR isolation
    
    :param fastaPath: Input sequence file (``.fasta`` format)
    :type fastaPath: str
    :param outputPath: Output embedding file (``.h5`` format)
    :type outputPath: str
    :param device: Compute device to use for embeddings [default: 0]
    :type device: int
    :param verbose: Print embedding progress
    :type verbose: bool
    """

    reload_models_to_device(device, embed_type)

    use_cuda = (device >= 0) and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(device)
        if verbose:
            log(f"# Using CUDA device {device} - {torch.cuda.get_device_name(device)}")
    else:
        if verbose:
            log("# Using CPU")

    if verbose:
        log("# Loading Model...")

    names, seqs = parse(fastaPath)

    # check if outputPath is an existing directory:
    if not os.path.isdir(outputPath):
        os.mkdir(outputPath)


    log("# Storing to {}...".format(outputPath))
    # with torch.no_grad(), h5py.File(outputPath, "a") as h5fi:
    try:
        for (name, seq) in tqdm(zip(names, seqs), total=len(names)):
            fname = os.path.join(outputPath, f'{name}.p')
            if not os.path.isfile(fname):
            # if name not in h5fi:
                prot = ProteinEmbedding(seq, chain_type, embed_device=f'cuda:{device}', dev=device)
                if num_mutations > 0:
                    z = prot.create_cdr_specific_embedding(embed_type, num_mutations,
                                                                        separators, cdr_masks)
                else:
                    z = prot.create_cdr_embedding_nomut(embed_type, separators, cdr_masks)

                with open(fname, 'wb') as f:
                    pickle.dump(z.cpu(), f)

    except KeyboardInterrupt:
        sys.exit(1)

