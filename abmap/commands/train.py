from itertools import combinations
import os, sys
import pickle
from tqdm import tqdm
import argparse
import random
import numpy as np
from matplotlib import pyplot as plt
import time

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch import nn
from torch import optim as opt
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from typing import Callable, NamedTuple

from abmap.model import AbMAPAttn, MultiTaskLossWrapper
from abmap.utils import evaluate_spearman
from abmap.dataloader import create_dataloader_sabdab, create_dataloader_libra
sys.path.append('..') # for access to model.py when calling torch.load()


type_to_dim = {'beplerberger':6165, 'esm1b':1280, 'tape':768, 'dscript':100}


class TrainArguments(NamedTuple):
    cmd: str
    device_num: int
    log_file: str
    exec_type: str
    num_epochs: int
    embed_dir: str
    batch_size: int
    print_every: int
    ckpt_every: int
    lr: float
    gamma: float
    emb_type: str
    model_loadpath: str
    model_savepath: str
    mut_type: str
    chain_type: str
    region: str
    lambda1: float
    init_alpha: float
    update_alpha: bool


def add_args(parser):
    parser.add_argument('--device_num', type=int, default=0,
                        help='Indicate which GPU device to train the model on')

    parser.add_argument('--log_file', type=str,
                        help='Name of the log file with plot values.')

    parser.add_argument('--exec_type', type=str, default='train',
                        choices=['train', 'evaluate'],
                        help='Execution Mode.')

    parser.add_argument('--num_epochs', type=int, default=1,
                        help='Number of epochs for training.')

    parser.add_argument('--embed_dir', type=str, required=True,
                        help='Name of the folder that contains the embeddings')

    parser.add_argument('--batch_size', type=int, default=32,
                        help='Input batch size for the model.')

    parser.add_argument('--print_every', type=int, default=100,
                        help='Print every ... batch')

    parser.add_argument('--ckpt_every', type=int, default=5,
                        help='How often (in epochs) to save the model checkpoint.')

    parser.add_argument('--lr', type=float, default=0.01,
                        help='Model learning rate')

    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Decay rate for the lr scheduler.')

    parser.add_argument('--emb_type', type=str, default='beplerberger',
                        help='Embedding Type.')

    parser.add_argument('--model_loadpath', type=str, default='None',
                        help='Path to an already trained model.')

    parser.add_argument('--model_savepath', type=str, default='',
                        help='Path to save the trained model.')

    parser.add_argument('--mut_type', type=str, default='cat2',
                        choices=['cat1', 'cat2'],
                        help='mutation type for random CDR mutations.')

    parser.add_argument('--chain_type', type=str, default='H',
                        choices=['H', 'L'],
                        help='Which chain in the Ab sequence to use.')

    parser.add_argument('--region', type=str, default='whole',
                        help='Which region on the sequence you would like to train on')

    parser.add_argument('--lambda1', type=float, default=0.0005,
                        help='Weight on the regularization loss')

    parser.add_argument('--init_alpha', type=float, default = 0.0,
                        help='Weight on the mse losses. 0 means an equal weight.')

    parser.add_argument('--update_alpha', action="store_true", dest='update_alpha',
                        help='The alpha value for the mse losses will be updated iteratively.')

    return parser


def train_model(device_num, log_file, exec_type, num_epochs, model_loadpath, model_savepath, region,
                embed_dir, batch_size, chain_type, init_alpha, update_alpha, lambda1, print_every=100, 
                lr=0.01, emb_type='beplerberger', ckpt_every=10, mut_type='cat2', gamma=0.5, **kwargs):
    start_time = time.time()

    if log_file == None:
        print("Your log will not be saved...")

    device = torch.device("cuda:{}".format(device_num) if torch.cuda.is_available() else "cpu")

    if embed_dir == "":
        print("You must provide the name of the directory for input embeddings")
        raise ValueError

    print("Loading data....")
    train_loader_sab, val_loader_sab, test_loader_sab = create_dataloader_sabdab(data_dir = embed_dir, 
                     batch_size=batch_size, emb_type=emb_type, mut_type=mut_type, chain_type=chain_type, region=region)
    train_loader_lib, val_loader_lib, test_loader_lib = create_dataloader_libra(data_dir = embed_dir, 
                     batch_size=batch_size, emb_type=emb_type, mut_type=mut_type, chain_type=chain_type)

    print("done loading!")
    train_size = len(train_loader_sab.dataset)

    # load model/operators
    if emb_type == 'beplerberger':
        model = AbMAPAttn(embed_dim=2200, mid_dim2=1024, mid_dim3=512, 
                                     proj_dim=252, num_enc_layers=1, num_heads=16).to(device)
    elif emb_type == 'protbert':
        model = AbMAPAttn(embed_dim=1024, mid_dim2=512, mid_dim3=256, 
                                     proj_dim=252, num_enc_layers=1, num_heads=16).to(device)
    elif emb_type == 'esm1b' or emb_type == 'esm2':
        model = AbMAPAttn(embed_dim=1280, mid_dim2=512, mid_dim3=256, 
                                     proj_dim=252, num_enc_layers=1, num_heads=16).to(device)
    elif emb_type == 'tape':
        model = AbMAPAttn(embed_dim=768, mid_dim2=256, mid_dim3=128, 
                                     proj_dim=60, num_enc_layers=1, num_heads=8).to(device)
    else:
        raise ValueError(f"The Embed Type {emb_type} Doesn't Exist!")

    loss_fn = nn.MSELoss()
    loss_wrapper = MultiTaskLossWrapper(2, init_alpha, update_alpha).to(device)
    # loss_opt = opt.SGD(loss_wrapper.parameters(), lr=lr)
    
    if model_loadpath == 'None':
        prev_epochs = 0
        optimizer = opt.SGD([{'params': model.parameters()},
                             {'params': loss_wrapper.parameters(), 'lr': 1e-3},], lr=lr)
        scheduler = MultiStepLR(optimizer, milestones=[40, 45], gamma=gamma)
    else:
        checkpoint = torch.load(model_loadpath, map_location=torch.device("cuda:{}".format(device_num)))
        prev_epochs = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded a saved model from provided path!")
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if model_savepath == '':
        print("Not saving the model checkpoints.")

    if exec_type == 'train':

        # writer = SummaryWriter(log_dir='../../runs')
        stept, stepv = 0, 0
        train_spearmans_s, train_spearmans_f, val_spearmans_s, val_spearmans_f = [], [], [], []
        train_losses, val_losses = [], []
        for i in range(num_epochs):

            total_loss, total_mse_loss, total_reg_loss = 0, 0, 0
            total_mse_loss_s, total_mse_loss_f = 0, 0

            ########################################################################
            # TRAINING
            ########################################################################
            model.train()
            loss_wrapper.train()

            train_preds_s, train_targs_s, train_preds_f, train_targs_f = [], [], [], []

            assert len(train_loader_sab) == len(train_loader_lib)
            
            batch_count = 0
            for k, (batch_s, batch_f) in tqdm(enumerate(zip(train_loader_sab, train_loader_lib)), total=len(train_loader_sab)):
                batch_count += 1

                x1_s, x2_s, tmscore, x1_s_padding_mask, x2_s_padding_mask = batch_s
                x1_f, x2_f, func_sc, x1_f_padding_mask, x2_f_padding_mask = batch_f

                optimizer.zero_grad()
                # loss_opt.zero_grad()
                
                x1_s, x2_s = x1_s.cuda(device), x2_s.cuda(device)
                x1_f, x2_f = x1_f.cuda(device), x2_f.cuda(device)
                x1_s_padding_mask, x2_s_padding_mask = x1_s_padding_mask.to(device), x2_s_padding_mask.to(device)
                x1_f_padding_mask, x2_f_padding_mask = x1_f_padding_mask.to(device), x2_f_padding_mask.to(device)

                # feed the input data into the model
                try:
                    pred_s, e1_s, e2_s = model(x1_s, x2_s, x1_s_padding_mask, x2_s_padding_mask, 0)
                    pred_f, e1_f, e2_f = model(x1_f, x2_f, x1_f_padding_mask, x2_f_padding_mask, 1)
                    # unit_norm = F.normalize(torch.ones(e1_s.shape)).cuda(device)
                except:
                    print(x1_s.shape, x2_s.shape)
                    print(x1_f.shape, x2_f.shape)
                    print('input min maxes', torch.max(x1_s), torch.min(x1_s), torch.max(x1_f), torch.min(x1_f),
                          torch.max(x2_s), torch.min(x2_s), torch.max(x2_f), torch.min(x2_f))
                    raise ValueError

                # compute loss and back propagate
                reg_term = torch.zeros(e1_s.shape).cuda(device)
                for em in [e1_s, e2_s, e1_f, e2_f]:
                    em_sq = em ** 2
                    reg_term += em_sq*torch.log(em_sq + 1e-7)
                reg_loss = lambda1*torch.sum(reg_term)

                mse_loss = loss_wrapper(pred_s, tmscore.cuda(device), pred_f, func_sc.cuda(device))

                total_reg_loss += reg_loss.item()
                total_mse_loss += mse_loss.item()
                total_mse_loss_s += loss_wrapper.loss_s.item()
                total_mse_loss_f += loss_wrapper.loss_f.item()
                loss = mse_loss + reg_loss
                
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                # loss_opt.step()

                # for evaluation
                train_preds_s.append(pred_s)
                train_preds_f.append(pred_f)
                train_targs_s.append(tmscore.cuda(device))
                train_targs_f.append(func_sc.cuda(device))

                # print intermediate loss values
                if k % print_every == 0:
                    print("TRAIN Epoch {}, Batch {}, MSE Loss S: {}, MSE Loss F: {}, Reg Loss: {}, Alpha: {}".format(prev_epochs+i+1, 
                                            k, total_mse_loss_s/(k+1), total_mse_loss_f/(k+1), total_reg_loss/(k+1), loss_wrapper.alpha.item()))

            train_losses.append((total_loss/batch_count, total_mse_loss/batch_count, total_reg_loss/batch_count))

            # computing spearman rank scores
            train_preds_s, train_targs_s = torch.cat(train_preds_s), torch.cat(train_targs_s)
            train_preds_f, train_targs_f = torch.cat(train_preds_f), torch.cat(train_targs_f)
            spear_s = evaluate_spearman(train_preds_s, train_targs_s)
            spear_f = evaluate_spearman(train_preds_f, train_targs_f)
            train_spearmans_s.append(spear_s)
            train_spearmans_f.append(spear_f)

            # print learned loss ratio and Spearman ranks scores every epoch
            # print("Loss weights:", torch.exp(-1*loss_wrapper.weights))
            print("Epoch {} [Training] Model Evaluation <Sabdab> (Spearman): {}".format(prev_epochs+i+1, spear_s))
            print("Epoch {} [Training] Model Evaluation <LibraSeq> (Spearman): {}".format(prev_epochs+i+1, spear_f))
            # print("TRAIN Epoch {}, MSE Loss: {}, Reg Loss: {}, Alpha: {}".format(prev_epochs+i+1, 
            #                                  total_mse_loss/batch_count, total_reg_loss/batch_count, alpha_print))
            print('----------------------------------------------------------')


            ########################################################################
            # VALIDATION SET
            ########################################################################
            total_loss, total_reg_loss, total_mse_loss = 0, 0, 0
            model.eval()
            loss_wrapper.eval()
            val_preds_s, val_targs_s, val_preds_f, val_targs_f = [], [], [], []

            batch_count = 0
            for k, (batch_s, batch_f) in tqdm(enumerate(zip(val_loader_sab, val_loader_lib)), total=len(val_loader_sab)):
                batch_count += 1
                x1_s, x2_s, tmscore, x1_s_padding_mask, x2_s_padding_mask = batch_s
                x1_f, x2_f, func_sc, x1_f_padding_mask, x2_f_padding_mask = batch_f

                # feed the input with no gradient calculation
                with torch.no_grad():
                    x1_s, x2_s = x1_s.cuda(device), x2_s.cuda(device)
                    x1_f, x2_f = x1_f.cuda(device), x2_f.cuda(device)

                    x1_s_padding_mask, x2_s_padding_mask = x1_s_padding_mask.to(device), x2_s_padding_mask.to(device)
                    x1_f_padding_mask, x2_f_padding_mask = x1_f_padding_mask.to(device), x2_f_padding_mask.to(device)

                    pred_s, e1_s, e2_s = model(x1_s, x2_s, x1_s_padding_mask, x2_s_padding_mask, 0)
                    pred_f, e1_f, e2_f = model(x1_f, x2_f, x1_f_padding_mask, x2_f_padding_mask, 1)

                    # LOSS CALCULATIONS
                    reg_term = torch.zeros(e1_s.shape).cuda(device)
                    for em in [e1_s, e2_s, e1_f, e2_f]:
                        em_sq = em ** 2
                        reg_term += em_sq*torch.log(em_sq + 1e-7)
                    reg_loss = lambda1*torch.sum(reg_term)
                    # reg_loss = lambda1*torch.sum((e1_s - unit_norm)**2 + (e2_s - unit_norm)**2 + (e1_f - unit_norm)**2 + (e2_f - unit_norm)**2)

                    mse_loss = loss_wrapper(pred_s, tmscore.cuda(device), pred_f, func_sc.cuda(device))


                    total_reg_loss += reg_loss.item()
                    total_mse_loss += mse_loss.item()
                    loss = mse_loss + reg_loss

                total_loss += loss.item()

                # for evaluation
                val_preds_s.append(pred_s)
                val_preds_f.append(pred_f)
                val_targs_s.append(tmscore.cuda(device))
                val_targs_f.append(func_sc.cuda(device))

            val_losses.append((total_loss/batch_count, total_mse_loss/batch_count, total_reg_loss/batch_count))

            # computing spearman rank scores
            val_preds_s, val_targs_s = torch.cat(val_preds_s), torch.cat(val_targs_s)
            val_preds_f, val_targs_f = torch.cat(val_preds_f), torch.cat(val_targs_f)
            spear_s = evaluate_spearman(val_preds_s, val_targs_s)
            spear_f = evaluate_spearman(val_preds_f, val_targs_f)
            val_spearmans_s.append(spear_s)
            val_spearmans_f.append(spear_f)

            # print("Epoch {} Average Validation Loss. MSE: {}, Reg: {}".format(prev_epochs+i+1, total_mse_loss/batch_count, total_reg_loss/batch_count))
            # print("Loss weights:", torch.exp(-1*loss_wrapper.weights))
            print("VALIDATION Epoch {} Model Evaluation <Sabdab> (Spearman): {}".format(prev_epochs+i+1, spear_s))
            print("VALIDATION Epoch {} Model Evaluation <LibraSeq> (Spearman): {}".format(prev_epochs+i+1, spear_f))

            print('----------------------------------------------------------')

            # saving models:
            if (i+1)%ckpt_every == 0 and model_savepath != "":
                if not os.path.isdir(model_savepath):
                    os.mkdir(model_savepath)

                raw_model = model.module if hasattr(model, "module") else model
                model_sum = raw_model.state_dict()
                savepath = os.path.join(model_savepath, 'AbMAP_{}_{}_epoch{}.pt'.format(emb_type, chain_type, prev_epochs+i+1))
                torch.save(model_sum, savepath)

            # learning rate schedule
            scheduler.step()

    # test set
    print("Evaluating on Test Set...")
    model.eval()
    logs = [dict(), dict()]
    for k, test_loader in enumerate([test_loader_sab, test_loader_lib]):
        test_preds, test_targs = [], []
        cos = nn.CosineSimilarity(dim=1)
        tms_list, cos_list, points_list, pred_list = [], [], [], []

        for batch in tqdm(test_loader):
            x1_data, x2_data, tmscore, x1_padding_mask, x2_padding_mask = batch

            with torch.no_grad():
                x1 = x1_data.cuda(device)
                x2 = x2_data.cuda(device)

                x1_padding_mask, x2_padding_mask = x1_padding_mask.to(device), x2_padding_mask.to(device)

                pred, _, _ = model(x1, x2, x1_padding_mask, x2_padding_mask, k) # to criss-cross structure and function, use: (k+1)%2

                model = model.module if hasattr(model, "module") else model

                x1_mean = model.mean_over_seq(x1, x1_padding_mask)
                x2_mean = model.mean_over_seq(x2, x2_padding_mask)

                # x1_mean, x2_mean = model(x1, x2, x1_padding_mask, x2_padding_mask, task=k, task_specific=True)

                # for evaluation
                test_preds.append(pred)
                test_targs.append(tmscore.cuda(device))

            pred_list += pred.tolist()

            # x1_mean = torch.mean(x1_p, dim=1)
            # x2_mean = torch.mean(x2_p, dim=1)

            cos_sim = cos(x1_mean, x2_mean)
            cos_temp = cos_sim.tolist()
            cos_list += cos_temp

            tms_temp = tmscore.tolist()
            tms_list += tms_temp

            for i in range(len(tms_temp)):
                points_list.append((tms_temp[i], cos_temp[i]))

        test_preds = torch.cat(test_preds)
        test_targs = torch.cat(test_targs)
        spear = evaluate_spearman(test_preds, test_targs)
        print("Task {}, Test set spearman rank: {}".format(k, spear))


        sorted_pairs = sorted(points_list, key=lambda x:x[0], reverse=False)

        split_arrays = np.array_split(sorted_pairs, 10)

        means = []
        for bucket in split_arrays:
            temp = []
            for tms, dot in bucket:
                temp.append(dot)
            means.append(np.mean(temp))

        logs[k]["bucket"] = means

        logs[k]["scatter_proj"] = [tms_list, cos_list]

        logs[k]["scatter_pred"] = [tms_list, pred_list]

        if exec_type == 'train':
        # Creating Loss and Spearman Plots
            if k == 0:
                plots_dict = {'train_losses': train_losses, 'val_losses': val_losses,
                              'train_spearmans': train_spearmans_s, 'val_spearmans':val_spearmans_s}
            else:
                plots_dict = {'train_losses': train_losses, 'val_losses': val_losses,
                              'train_spearmans': train_spearmans_f, 'val_spearmans':val_spearmans_f}

            logs[k]["train_losses"] = plots_dict['train_losses']
            logs[k]["val_losses"] = plots_dict['val_losses']
            logs[k]["train_spearmans"] = plots_dict['train_spearmans']
            logs[k]["val_spearmans"] = plots_dict['val_spearmans']

    if log_file != None:
        with open(log_file, "wb") as f:
            pickle.dump(logs, f)

    end_time = time.time()
    print("It took {:.2f} minutes to train!".format((end_time-start_time)/60))


def main(args):

    print(args)
    train_model(**vars(args))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    add_args(parser)

    args = parser.parse_args()

    if torch.cuda.is_available():
        print("Using the GPU!")
    else:
        print("WARNING: Could not find GPU! Using CPU only")

    torch.multiprocessing.set_start_method('spawn')
    main(args)
    print("DONE")