import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import glob
import os
import pickle
from scipy.stats import spearmanr
from matplotlib import pyplot as plt
import time, tempfile, json
from Bio import SeqIO
from datetime import datetime


def log(m, file=None, timestamped=True, print_also=False):
    curr_time = f"[{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}] "
    log_string = f"{curr_time if timestamped else ''}{m}"
    if file is None:
        print(log_string)
    else:
        print(log_string, file=file)
        if print_also:
            print(log_string)
        file.flush()


def parse(f, comment="#"):
    descriptions = []
    sequences = []

    for record in SeqIO.parse(f, "fasta"):
        descriptions.append(record.description)
        sequences.append(str(record.seq))

    return descriptions, sequences



def evaluate_spearman(pred, target):
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    rho, pval = spearmanr(pred, target)
    return rho



def get_boolean_mask(sequence, chain_type, scheme, buffer_region, dev, fold=0,
                     anarci_dir='../data/anarci_files'):
    chothia_nums = {'H': [[26, 32], [52, 56], [96, 101]],
                    'L': [[26, 32], [50, 52], [91, 96]]}

    # chothia_nums2 = {'H': [[26, 32], [52, 56], [95, 102]],
    #                  'L': [[24, 34], [50, 56], [89, 97]]}

    chothia_nums2 = {'H': [[24, 34], [50, 58], [94, 103]],
                     'L': [[24, 34], [48, 54], [89, 98]]}

    imgt_nums = {'H': [(26, 33), (51, 56), (93, 102)],
                 'L': [(27, 32), (50, 51), (89, 97)]}

    if scheme == 'chothia':
        all_regions = chothia_nums2
    elif scheme == 'imgt':
        all_regions = imgt_nums
    else:
        print("Such numbering does NOT exist!")
        raise NotImplementedError

    # increase each CDR region by 1 at each end if buffer_region
    if buffer_region:
        for c_type in all_regions.keys():
            for i, cdr_region in enumerate(all_regions[c_type]):
                all_regions[c_type][i][0] -= 1
                all_regions[c_type][i][1] += 1

    # change temp_path to a folder you'd like to save your ANARCI file to
    # temp_path = "/net/scratch3.mit.edu/scratch3-3/chihoim/misc/temp{}".format(fold)

    if not os.path.isdir(anarci_dir):
        os.makedirs(anarci_dir)
    
    temp_name = os.path.join(anarci_dir, 'temp{}'.format(dev))
    print(temp_name)
    print(f'Sequence: {sequence}')
    print(f'scheme: {scheme}')

    os.system('ANARCI -i {} --csv -o {} -s {}'.format(sequence, temp_name, scheme))
    
    ### debug!
    # print("chain_type variable:", chain_type)

    regions = all_regions[chain_type]
    if chain_type == 'H':
        file_name = glob.glob(anarci_dir+'/'+'*{}_H.csv'.format(dev))[0]
    else:
        file_name = glob.glob(anarci_dir+'/'+'*{}_KL.csv'.format(dev))[0]

    try:
        temp = pd.read_csv(file_name)
    except:
        print("Can't READ this file! file name is: {}".format(file_name))
        raise ValueError
    df = pd.DataFrame(temp)

    df = df.drop(columns=df.columns[(df == '-').any()])

    prot = df.iloc[:, 13:]  # important data starting at 13th col.

    seq_list = prot.columns.values.tolist()

    # for DEBUG:
    if len(seq_list) == 0:
        raise ValueError
    if int(seq_list[0]) > 34 or int(seq_list[-1]) < 89:
        print(sequence)
        print(seq_list)
        raise ValueError

    cdr_mask = torch.zeros(len(sequence))

    seqstart_idx = df.iloc[0]['seqstart_index']
    for r, (start_val, end_val) in enumerate(regions):
        start_pointer, end_pointer = start_val, end_val

        found = False
        while not found:
            if str(start_pointer) in seq_list:
                h_cdr_start = seqstart_idx + seq_list.index(str(start_pointer))
                found = True
            else:
                start_pointer += 1
                if start_pointer >= end_val:
                    print("Region seems invalid!")
                    print("ANARCI list:", seq_list)
                    raise ValueError

        found = False
        while not found:
            if str(end_pointer) in seq_list:
                h_cdr_end = seqstart_idx + seq_list.index(str(end_pointer))
                found = True
            else:
                end_pointer -= 1
                if end_pointer <= start_val:
                    print("Region seems invalid!")
                    print("ANARCI list:", seq_list)
                    raise ValueError

        cdr_mask[h_cdr_start : h_cdr_end + 1] = r+1

    return cdr_mask


def find_sequence(root_path, pdb_id = ''):
    """
    given a dataset, find the heavy and light chain sequences
    """

    # path = '../data/raw/sabdab/sabdab_dataset'

    prot_path = os.path.join(root_path, pdb_id, "sequence")

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
