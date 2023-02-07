import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import glob
import os
import pickle
from scipy.stats import spearmanr
from matplotlib import pyplot as plt

import os, time, tempfile, json


def get_free_gpu():
    fname = tempfile.mkstemp(prefix="AB1")[1]

    os.system(f"/data/cb/rsingh/miniconda3/envs/antibody/bin/gpustat --json > {fname}")

    time.sleep(2)

    #print(fname)

    jobj = json.load(open(fname,'rt'))

    gpuutil1 = sorted([ (d['utilization.gpu'], d['memory.used'], d['index']) for d in jobj['gpus'] ])

    gpuutil = [(a,b,c) for a,b,c in gpuutil1 if b < 16000] #a < 70
    #print(gpuutil, jobj["gpus"])
    os.system(f"rm -f {fname}")

    if len(gpuutil) > 0: # and gpuutil[0][0] < 30:
        cuda_device = gpuutil[0][2]
        print('Free gpu id: ', cuda_device)
        return cuda_device
    else:
        print('No free gpu')
        return -1


def scatter_plot(comb, emb_type):

    tms_list, dot_list, points_list = [], [], []
    for id1, id2, tms in tqdm(comb):

        emb1 = pickle.load(open(id1, 'rb')).cpu()
        emb2 = pickle.load(open(id2, 'rb')).cpu()

        x1 = torch.mean(emb1, dim=0)
        x2 = torch.mean(emb2, dim=0)

        x1 = x1 / torch.norm(x1)
        x2 = x2 / torch.norm(x2)

        cos_sim = torch.dot(x1, x2).item()

        tms_list.append(tms)
        dot_list.append(cos_sim)
        points_list.append((tms, cos_sim))

    sorted_pairs = sorted(points_list, key=lambda x:x[0], reverse=False)

    split_arrays = np.array_split(sorted_pairs, 10)

    means = []
    for bucket in split_arrays:
        temp = []
        for tms, dot in bucket:
            temp.append(dot)
        means.append(np.mean(temp))

    bucket_fig = '/net/scratch3/scratch3-3/chihoim/figures/bucket_{}.png'.format(emb_type)
    plt.plot(means)
    plt.xlabel("TM Score Bucket")
    plt.ylabel("Avg Dot Product")
    plt.savefig(bucket_fig, bbox_inches='tight')

    plt.clf()

    scatter_fig = '/net/scratch3/scratch3-3/chihoim/figures/scatter_{}.png'.format(emb_type)
    plt.scatter(dot_list, tms_list)
    plt.xlabel("Cosine Similarity (Dot Product)")
    plt.ylabel("TM Score")
    plt.savefig(scatter_fig, bbox_inches='tight')

    plt.clf()


def evaluate_spearman(pred, target):
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    rho, pval = spearmanr(pred, target)
    return rho


def diff_normalize(mutated, base, dim = (1,2)):
    diff_matrix = base - mutated
    # print(diff_matrix.shape)
    norm_diff = torch.norm(diff_matrix.cpu(), dim=dim)
    return norm_diff/torch.norm(base)


def get_boolean_mask(sequence, chain_type, scheme, buffer_region, dev, fold=0):
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
        for chain_type in all_regions.keys():
            for i, cdr_region in enumerate(all_regions[chain_type]):
                all_regions[chain_type][i][0] -= 1
                all_regions[chain_type][i][1] += 1

    # change temp_path to a folder you'd like to save your ANARCI file to
    temp_path = "/net/scratch3.mit.edu/scratch3-3/chihoim/misc/temp{}".format(fold)
    if not os.path.isdir(temp_path):
        os.mkdir(temp_path)
    
    temp_name = os.path.join(temp_path, 'temp{}'.format(dev))

    os.system('ANARCI -i {} --csv -o {} -s {}'.format(sequence, temp_name, scheme))
    
    regions = all_regions[chain_type]
    if chain_type == 'H':
        file_name = glob.glob(temp_path+'/'+'*{}_H.csv'.format(dev))[0]
    else:
        file_name = glob.glob(temp_path+'/'+'*{}_KL.csv'.format(dev))[0]

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
