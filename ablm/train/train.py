from itertools import combinations
import os, sys
import pickle
from tqdm import tqdm
import argparse
import random
import pickle
import numpy as np
from matplotlib import pyplot as plt

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch import nn
from torch import optim as opt
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

sys.path.append('..')
from model import AntibodyNet #, AntibodyNet2
from utils import evaluate_spearman, scatter_plot
from dataloader import create_dataloader, create_dataloader_concat, create_dataloader_libra

type_to_dim = {'beplerberger':6165, 'esm1b':1280, 'tape':768, 'dscript':100}


def train_model(data_name, device_num, exec_type, num_epochs, model_loadpath, model_savepath, batch_size=16, print_every=100,
                lr=0.01, emb_type='beplerberger', ckpt_every=10, mut_type='cat2', chain_type='H', gamma=0.5, **kwargs):

    device = torch.device("cuda:{}".format(args.device_num) if torch.cuda.is_available() else "cpu")

    data_dir = os.path.join('/data/cb/rsingh/work/antibody/ci_data/processed', data_name)
    print("Loading data...")
    if data_name == 'sabdab_all':
        train_loader, val_loader, test_loader = create_dataloader(data_dir = data_dir, batch_size=batch_size, emb_type=emb_type, 
                                                    mut_type=mut_type, chain_type=chain_type)
    elif data_name == 'libraseq':
        train_loader, val_loader, test_loader = create_dataloader_libra(data_dir = data_dir, batch_size=batch_size, emb_type=emb_type, 
                                                    mut_type=mut_type, chain_type=chain_type)
    # train_loader, val_loader, test_loader = create_dataloader_concat(data_dir = data_dir, batch_size=batch_size,
    #                                                 mut_type=mut_type, chain_type=chain_type)

    print("done loading from {}!".format(data_name))
    train_size = len(train_loader.dataset)

    # load model/operators
    model = AntibodyNet(embed_dim=type_to_dim[emb_type], lstm_layers=1).to(device)
    # model = AntibodyNet(embed_dim=8313, mid_dim1 = 4096, mid_dim2 = 1024, mid_dim3 = 256, lstm_layers=1).to(device)
    # model = nn.DataParallel(model, device_ids=[0,1,2,3])#.to(device)
    loss_fn = nn.MSELoss()
    if model_loadpath == 'None':
        prev_epochs = 0
        optimizer = opt.SGD(model.parameters(), lr=lr)
        scheduler = MultiStepLR(optimizer, milestones=[15, 25, 35], gamma=gamma)
    else:
        checkpoint = torch.load(model_loadpath, map_location=torch.device("cuda:{}".format(device_num)))
        prev_epochs = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if exec_type == 'train':
        writer = SummaryWriter(log_dir='../runs')
        stept, stepv = 0, 0
        train_spearmans, val_spearmans = [], []
        train_losses, val_losses = [], []
        for i in range(num_epochs):

            total_loss = 0

            # train the model
            model.train()
            train_preds, train_targs = [], []
            temp_losses = []
            for k, batch in enumerate(train_loader):

                x1_data, x2_data, tmscore = batch

                # for batch norm:
                if tmscore.shape[0] < 8:
                    break

                optimizer.zero_grad()

                x1 = x1_data.cuda(device)
                x2 = x2_data.cuda(device)

                try:
                    pred, e1, e2 = model(x1, x2)
                    unit_norm = F.normalize(torch.ones(e1.shape))
                except:
                    print(x1.shape, x2.shape)
                    raise ValueError

                # back-prop
                loss = loss_fn(pred, tmscore.cuda(device)) + lambda1*torch.sum((e1-unit_norm)**2 + (e2-unit_norm)**2)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()

                # for evaluation
                train_preds.append(pred)
                train_targs.append(tmscore.cuda(device))
                temp_losses.append(loss.item())

                if k % print_every == 0:
                    writer.add_scalar('Loss/train', total_loss/(k+1), stept)#i+(k+1)*batch_size/train_size)
                    stept += 1
                    print("TRAIN Epoch {}, Batch {}, Loss: {}".format(prev_epochs+i+1, k, total_loss/(k+1)))

            train_losses.append(np.mean(temp_losses))

            train_preds = torch.cat(train_preds)
            train_targs = torch.cat(train_targs)
            spear = evaluate_spearman(train_preds, train_targs)
            train_spearmans.append(spear)

            # print("Epoch {} [Training] Model Evaluation (L2): {}".format(i+1, l2))
            print("Epoch {} [Training] Model Evaluation (Spearman): {}".format(prev_epochs+i+1, spear))
            print('----------------------------------------------------------')


            # VALIDATION SET
            total_loss = 0
            model.eval()
            val_preds, val_targs = [], []
            temp_losses = []
            for k, batch in enumerate(val_loader):
                x1_data, x2_data, tmscore = batch

                with torch.no_grad():
                    x1 = x1_data.cuda(device)
                    x2 = x2_data.cuda(device)

                    pred, e1, e2 = model(x1, x2)
                    unit_norm = F.normalize(torch.ones(e1.shape))

                    # free memory
                    x1.cpu()
                    x2.cpu()
                    x1_data.cpu()
                    x2_data.cpu()
                    del x1, x2, x1_data, x2_data

                    # for evaluation
                    val_preds.append(pred)
                    val_targs.append(tmscore.cuda(device))
                    temp_losses.append(loss.item())

                loss = loss_fn(pred, tmscore.cuda(device)) + lambda1*torch.sum((e1-unit_norm)**2 + (e2-unit_norm)**2)
                total_loss += loss.item()

                if k % print_every == 0:
                    writer.add_scalar('Loss/validation', total_loss/(k+1), stepv)#i+(k+1)*batch_size/train_size)
                    stepv += 1
                    print("VAL Epoch {}, Batch {}, Loss: {}".format(prev_epochs+i+1, k, total_loss/(k+1)))

            val_losses.append(np.mean(temp_losses))

            val_preds = torch.cat(val_preds)
            val_targs = torch.cat(val_targs)
            spear = evaluate_spearman(val_preds, val_targs)
            val_spearmans.append(spear)

            # print("Epoch {} [Validation] Model Evaluation (L2): {}".format(i+1, l2))
            print("Epoch {} [Validation] Model Evaluation (Spearman): {}".format(prev_epochs+i+1, spear))        
            print('----------------------------------------------------------')

            # saving models:
            if (i+1)%ckpt_every == 0:
                model_savepath = args.model_savepath
                # model_savepath = '/data/cb/rsingh/work/antibody/model_ckpts/sabdab_100k'
                model_savepath = os.path.join(model_savepath, emb_type)
                raw_model = model.module if hasattr(model, "module") else model
                torch.save({'epoch':prev_epochs+i, 'model_state_dict': raw_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                            os.path.join(model_savepath, 'AbLSTM_{}_epoch{}.pt'.format(emb_type, prev_epochs+i+1)))

            # print out the weight and bias of last layer:
            raw_model = model.module if hasattr(model, "module") else model
            print("Final Layer Weight: {}, Bias: {}".format(raw_model.transform2.weight.item(), raw_model.transform2.bias.item()))
            print('----------------------------------------------------------')

            # learning rate schedule
            scheduler.step()

    # test set
    model.eval()
    test_preds, test_targs = [], []
    cos = nn.CosineSimilarity(dim=1)
    tms_list, cos_list, points_list, pred_list = [], [], [], []
    print("Evaluating on Test Set...")
    for batch in tqdm(test_loader):
        x1_data, x2_data, tmscore = batch

        with torch.no_grad():
            x1 = x1_data.cuda(device)
            x2 = x2_data.cuda(device)

            pred, _, _ = model(x1, x2)

            model = model.module if hasattr(model, "module") else model

            x1_p = model.project(x1)
            x2_p = model.project(x2)

            # for evaluation
            test_preds.append(pred)
            test_targs.append(tmscore.cuda(device))

        pred_list += pred.tolist()

        x1_mean = torch.mean(x1_p, dim=1)
        x2_mean = torch.mean(x2_p, dim=1)        

        cos_sim = cos(x1_mean, x2_mean)
        # cos_sim = cos(x1_p, x2_p)
        cos_temp = cos_sim.tolist()
        cos_list += cos_temp

        tms_temp = tmscore.tolist()
        tms_list += tms_temp

        for i in range(len(tms_temp)):
            points_list.append((tms_temp[i], cos_temp[i]))

    test_preds = torch.cat(test_preds)
    test_targs = torch.cat(test_targs)
    spear = evaluate_spearman(test_preds, test_targs)
    print("Test set spearman rank: {}".format(spear))
    # with open('../figures/{}_spearman.txt'.format(emb_type), 'w') as s:
    #     s.write('spearman: {}'.format(spear))

    sorted_pairs = sorted(points_list, key=lambda x:x[0], reverse=False)

    split_arrays = np.array_split(sorted_pairs, 10)

    means = []
    for bucket in split_arrays:
        temp = []
        for tms, dot in bucket:
            temp.append(dot)
        means.append(np.mean(temp))


    bucket_fig = '../figures/{}_pred_bucket.png'.format(emb_type)
    plt.plot(means)
    plt.xlabel("Actual Score Bucket")
    plt.ylabel("Avg Cosine Similarity")
    plt.savefig(bucket_fig, bbox_inches='tight')

    plt.clf()

    scatter_fig = '../figures/{}_scatter_projcossim.png'.format(emb_type)
    plt.scatter(tms_list, cos_list)
    plt.xlabel("Actual Score")
    plt.ylabel("CosSim of Projections")
    plt.savefig(scatter_fig, bbox_inches='tight')

    plt.clf()

    scatter_fig_tms = '../figures/{}_scatter_finalpred.png'.format(emb_type)
    plt.scatter(tms_list, pred_list)
    plt.xlabel("Actual Score")
    plt.ylabel("Predicted Score")
    plt.savefig(scatter_fig_tms, bbox_inches='tight')

    plt.clf()


    # Creating Loss and Spearman Plots
    if exec_type == 'train':
        plots_dict = {'train_losses': train_losses, 'val_losses': val_losses,
                      'train_spearmans': train_spearmans, 'val_spearmans':val_spearmans}

        loss_fig = '../figures/{}_loss.png'.format(emb_type)
        plt.plot(plots_dict['train_losses'])
        plt.plot(plots_dict['val_losses'])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(['train_loss', 'val_loss'], loc='upper right')
        plt.savefig(loss_fig, bbox_inches='tight')

        plt.clf()

        spear_fig = '../figures/{}_spearman.png'.format(emb_type)
        plt.plot(plots_dict['train_spearmans'])
        plt.plot(plots_dict['val_spearmans'])
        plt.xlabel("Epoch")
        plt.ylabel("Spearman's rho")
        plt.legend(['train_spearman', 'val_spearman'], loc='lower right')
        plt.savefig(spear_fig, bbox_inches='tight')


def main(args):

    print(args)
    train_model(**vars(args))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_name', type=str, default='sabdab_all',
                        choices=['sabdab_all', 'libraseq'],
                        help='Name of the dataset to use to train the model')

    parser.add_argument('--device_num', type=int, default=0,
                        help='Indicate which GPU device to train the model on')

    parser.add_argument('--exec_type', type=str, default='train',
                        choices=['train', 'evaluate'],
                        help='Execution Mode.')

    parser.add_argument('--num_epochs', type=int, default=1,
                        help='Number of epochs for training.')

    parser.add_argument('--batch_size', type=int, default=8,
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
                        choices=['beplerberger', 'esm1b', 'tape', 'dscript', 'concat'],
                        help='Embedding Type.')

    parser.add_argument('--model_loadpath', type=str, default='None',
                        help='Path to an already trained model.')

    parser.add_argument('--model_savepath', type=str, default='/data/cb/rsingh/work/antibody/model_ckpts/libra_100k',
                        help='Path to save the trained model.')

    parser.add_argument('--mut_type', type=str, default='cat2',
                        choices=['cat1', 'cat2'],
                        help='mutation type for random CDR mutations.')

    parser.add_argument('--chain_type', type=str, default='H',
                        choices=['H', 'L'],
                        help='Which chain in the Ab sequence to use.')

    args = parser.parse_args()

    if torch.cuda.is_available():
        print("Using the GPU!")
    else:
        print("WARNING: Could not find GPU! Using CPU only")

    main(args)