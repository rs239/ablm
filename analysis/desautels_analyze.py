import torch
import os, sys
import pickle
from tqdm import tqdm
import time
import argparse
import h5py
import numpy as np

from torch import nn
from torch import optim as opt
from torch import autograd
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F

from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

sys.path.append('..')
from model import AbMAPAttn, DesautelsPredictor, MultiTaskLossWrapper
from dataloader import create_dataloader_desautels
from utils import evaluate_spearman
from embed import reload_models_to_device
from protein_embedding import ProteinEmbedding


pred_cols = ['FoldX_Average_Whole_Model_DDG', 'FoldX_Average_Interface_Only_DDG',
            'Statium', 'Sum_of_Rosetta_Flex_single_point_mutations',
            'Sum_of_Rosetta_Total_Energy_single_point_mutations']

all_aas = ['A', 'G', 'V', 'I', 'L', 'F', 'P', 'Y', 'M', 'T',
           'S', 'H', 'N', 'Q', 'W', 'R', 'K', 'D', 'E', 'C', 'X']

H_cdr_idx = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107]
L_cdr_idx = [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 46, 47, 48, 49, 50, 51, 52, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97]

def train_model_onehot(args, device_num, num_epochs, batch_size, exec_type, lr, traintest_split):
    """
    This function trains the ddG prediction network using 
    the one-hot representations of input amino acids
    """

    device = torch.device("cuda:{}".format(device_num) if torch.cuda.is_available() else "cpu")

    # load the model/optimizer
    model = DesautelsPredictor(input_dim=84, mid_dim = 16, embed_dim = 21, hidden_dim=21, num_heads=7).to(device)
    loss_fn = nn.MSELoss()
    optimizer = opt.SGD(model.parameters(), lr=lr)
    scheduler = MultiStepLR(optimizer, milestones=[50, 80], gamma=0.1)

    print("Building dataloader...")
    # load the dataloader
    train_loader_des, val_loader_des, test_loader_des = create_dataloader_desautels(batch_size=batch_size, 
                                                            traintest_split = traintest_split, onehot=True)
    print("Done building dataloader!")

    # load WT features:
    with open('/data/cb/rsingh/work/antibody/ci_data/processed/desautels/WT_feats/WT_H_onehot.p', 'rb') as f:
        WT_H_feat = pickle.load(f).to(device)
    with open('/data/cb/rsingh/work/antibody/ci_data/processed/desautels/WT_feats/WT_L_onehot.p', 'rb') as f:
        WT_L_feat = pickle.load(f).to(device)


    X_train, y_train = [],[]
    X_test, y_test = [],[]

    if exec_type == 'train':
        # with autograd.detect_anomaly():
        for i in range(num_epochs):

            # train the model
            model.train()
            total_loss = 0

            for k, (feat_H_batch, feat_L_batch, label_batch) in enumerate(train_loader_des):

                # predict the label using the model
                feat_H_batch, feat_L_batch, label_batch = feat_H_batch.to(device), feat_L_batch.to(device), label_batch.to(device)

                # # create the WT features batch:
                WT_H_feat_batch = WT_H_feat.repeat(feat_H_batch.shape[0], 1, 1)
                WT_L_feat_batch = WT_L_feat.repeat(feat_L_batch.shape[0], 1, 1)

                # change to floats:
                feat_H_batch = feat_H_batch.float()
                feat_L_batch = feat_L_batch.float()
                WT_H_feat_batch = WT_H_feat_batch.float()
                WT_L_feat_batch = WT_L_feat_batch.float()

                pred_batch = model(feat_H_batch, WT_H_feat_batch, feat_L_batch, WT_L_feat_batch)

                # update the parameters (loss, backprop, etc.)
                loss = loss_fn(pred_batch, label_batch)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if k == len(train_loader_des) - 1:
                    print("TRAIN Epoch {}, Batch {}, Loss: {}".format(i, k, total_loss/(k+1)))

            scheduler.step()

            # print("********************")

            # validate the model
            model.eval()
            total_loss = 0

            for k, (feat_H_batch, feat_L_batch, label_batch) in enumerate(val_loader_des):

                # predict the label using the model
                feat_H_batch, feat_L_batch, label_batch = feat_H_batch.to(device), feat_L_batch.to(device), label_batch.to(device)

                # # create the WT features batch:
                WT_H_feat_batch = WT_H_feat.repeat(feat_H_batch.shape[0], 1, 1)
                WT_L_feat_batch = WT_L_feat.repeat(feat_L_batch.shape[0], 1, 1)

                # change to floats:
                feat_H_batch = feat_H_batch.float()
                feat_L_batch = feat_L_batch.float()
                WT_H_feat_batch = WT_H_feat_batch.float()
                WT_L_feat_batch = WT_L_feat_batch.float()

                with torch.no_grad():
                    pred_batch = model(feat_H_batch, WT_H_feat_batch, feat_L_batch, WT_L_feat_batch)


                # update the parameters (loss, backprop, etc.)
                loss = loss_fn(pred_batch, label_batch)
                total_loss += loss.item()


                if k == len(val_loader_des) - 1:
                    print("VAL Epoch {}, Batch {}, Loss: {}".format(i, k, total_loss/(k+1)))

            print("----------------------------------------------")

    # feed the model through the test set
    model.eval()
    losses = []
    labels_total, preds_total = [], []

    for k, (feat_H_batch, feat_L_batch, label_batch) in enumerate(test_loader_des): #, total=len(test_loader_des):

        # predict the label using the model
        feat_H_batch, feat_L_batch, label_batch = feat_H_batch.to(device), feat_L_batch.to(device), label_batch.to(device)

        # # create the WT features batch:
        WT_H_feat_batch = WT_H_feat.repeat(feat_H_batch.shape[0], 1, 1)
        WT_L_feat_batch = WT_L_feat.repeat(feat_L_batch.shape[0], 1, 1)

        # change to floats:
        feat_H_batch = feat_H_batch.float()
        feat_L_batch = feat_L_batch.float()
        WT_H_feat_batch = WT_H_feat_batch.float()
        WT_L_feat_batch = WT_L_feat_batch.float()

        with torch.no_grad():
            pred_batch = model(feat_H_batch, WT_H_feat_batch, feat_L_batch, WT_L_feat_batch)

        #-----------------------------------------------------

        labels_total.append(label_batch)
        preds_total.append(pred_batch)

        # update the parameters (loss, backprop, etc.)
        loss = loss_fn(pred_batch, label_batch)
        losses.append(loss.item())

    save_dict = dict()

    # Calculate Average Loss over the TEST SET:
    print("TEST SET Average LOSS: {}".format(sum(losses)/len(losses)))

    # calculate spearman rank for each value:
    labels_total = torch.cat(labels_total, dim=0)
    preds_total = torch.cat(preds_total, dim=0)

    for j in range(labels_total.shape[-1]):
        spear = evaluate_spearman(preds_total[:, j], labels_total[:, j])
        print("Split: {}, Spearman Rank Scores for {}: {}".format(traintest_split, pred_cols[j], spear))
        save_dict[pred_cols[j]] = (preds_total[:, j], labels_total[:, j])

    with open('/data/cb/rsingh/work/antibody/ci_data/processed/desautels/plots/{}_spl{}.p'.format(args.logname, traintest_split), 'wb') as p:
        pickle.dump(save_dict, p)



def train_model(args, device_num, num_epochs, batch_size, exec_type, lr, ridge, traintest_split, embed_type):
    """
    This function executes either:
    a) Ridge Regression using our model's fixed-length embeddings

    or

    b) Training the ddG prediction model using either:
        1) our model's representations
        2) ESM-1b representations
        3) ProtBert representations

    """

    device = torch.device("cuda:{}".format(device_num) if torch.cuda.is_available() else "cpu")

    embed_size_dict = {'bb': 2200, 'esm1b': 1280, 'protbert': 1024}
    if embed_type in ['bb', 'esm1b', 'protbert']:
        embed_size = embed_size_dict[embed_type]
        mid_dim = 128*2 if embed_type == 'bb' else 64*2

    if embed_type == 'ours_bb':
        embed_size = 256
        mid_dim = 32*2

        # load the pre-trained multi-task training model:
        pretrained_H = AbMAPAttn(embed_dim=2200, mid_dim2=1024, mid_dim3=512,
                                      proj_dim=252, num_enc_layers=1, num_heads=16).to(device)
        pretrained_L = AbMAPAttn(embed_dim=2200, mid_dim2=1024, mid_dim3=512,
                                      proj_dim=252, num_enc_layers=1, num_heads=16).to(device)
        root_path = '/net/scratch3.mit.edu/scratch3-3/chihoim/model_ckpts'
        H_best_path = os.path.join(root_path, '091522_bb_newnum_H', 'beplerberger_epoch50.pt')
        L_best_path = os.path.join(root_path, '091522_bb_newnum_L', 'beplerberger_epoch50.pt')

    if embed_type == 'ours_protbert':
        embed_size = 256
        mid_dim = 32*2

        # load the pre-trained multi-task training model:
        pretrained_H = AbMAPAttn(embed_dim=1024, mid_dim2=512, mid_dim3=256,
                                      proj_dim=252, num_enc_layers=1, num_heads=16).to(device)
        pretrained_L = AbMAPAttn(embed_dim=1024, mid_dim2=512, mid_dim3=256,
                                      proj_dim=252, num_enc_layers=1, num_heads=16).to(device)
        root_path = '/net/scratch3.mit.edu/scratch3-3/chihoim/model_ckpts'
        H_best_path = os.path.join(root_path, '101722_abnet_protbert_H_whole', 'protbert_epoch50.pt')
        L_best_path = os.path.join(root_path, '101722_abnet_protbert_L_whole', 'protbert_epoch50.pt')

    if embed_type == 'ours_esm1b':
        embed_size = 256
        mid_dim = 32*2

        # load the pre-trained multi-task training model:
        pretrained_H = AbMAPAttn(embed_dim=1280, mid_dim2=512, mid_dim3=256,
                                      proj_dim=252, num_enc_layers=1, num_heads=16).to(device)
        pretrained_L = AbMAPAttn(embed_dim=1280, mid_dim2=512, mid_dim3=256,
                                      proj_dim=252, num_enc_layers=1, num_heads=16).to(device)
        root_path = '/net/scratch3.mit.edu/scratch3-3/chihoim/model_ckpts'
        H_best_path = os.path.join(root_path, '103122_abnet_esm1b_Hwhole', 'esm1b_epoch50.pt')
        L_best_path = os.path.join(root_path, '103122_abnet_esm1b_Lwhole', 'esm1b_epoch50.pt')

    if embed_type in ['ours_bb', 'ours_protbert', 'ours_esm1b']:
        checkpoint_H = torch.load(H_best_path, map_location=device)
        checkpoint_L = torch.load(L_best_path, map_location=device)
        pretrained_H.load_state_dict(checkpoint_H['model_state_dict'])
        pretrained_L.load_state_dict(checkpoint_L['model_state_dict'])
        pretrained_H.eval()
        pretrained_L.eval()


    # load the predictor model/optimizer
    model = DesautelsPredictor(input_dim=embed_size*4, mid_dim = mid_dim, embed_dim = embed_size, 
                               hidden_dim=embed_size, num_heads=16).to(device)
    loss_fn = nn.MSELoss()
    optimizer = opt.SGD(model.parameters(), lr=lr)
    scheduler = MultiStepLR(optimizer, milestones=[50, 80], gamma=0.1)

    print("Building dataloader...")
    # load the dataloader
    train_loader_des, val_loader_des, test_loader_des = create_dataloader_desautels(batch_size=batch_size, 
                                             traintest_split = traintest_split, embed_type=args.embed_type)
    print("Done building dataloader!")

    # load WT features:
    with open('/data/cb/rsingh/work/antibody/ci_data/processed/desautels/WT_feats/WT_H_{}.p'.format(args.embed_type), 'rb') as f:
        WT_H_feat = pickle.load(f).to(device)
    with open('/data/cb/rsingh/work/antibody/ci_data/processed/desautels/WT_feats/WT_L_{}.p'.format(args.embed_type), 'rb') as f:
        WT_L_feat = pickle.load(f).to(device)


    X_train, y_train = [],[]
    X_test, y_test = [],[]

    if exec_type == 'train':
        # with autograd.detect_anomaly():
        for i in range(num_epochs):

            # train the model
            model.train()
            total_loss = 0

            for k, (feat_H_batch, feat_L_batch, label_batch) in enumerate(train_loader_des):

                # predict the label using the model
                feat_H_batch, feat_L_batch, label_batch = feat_H_batch.to(device), feat_L_batch.to(device), label_batch.to(device)

                # # create the WT features batch:
                WT_H_feat_batch = WT_H_feat.repeat(feat_H_batch.shape[0], 1, 1)
                WT_L_feat_batch = WT_L_feat.repeat(feat_L_batch.shape[0], 1, 1)

                # for RIDGE REGRESSION:
                with torch.no_grad():
                    if ridge:
                        task = 0
                        feat_h, _ = pretrained_H(feat_H_batch, feat_H_batch, None, None, task=task, task_specific=True)
                        feat_wt_h, _ = pretrained_H(WT_H_feat_batch, WT_H_feat_batch, None, None, task=task, task_specific=True)
                        feat_l, _ = pretrained_L(feat_L_batch, feat_L_batch, None, None, task=task, task_specific=True)
                        feat_wt_l, _ = pretrained_L(WT_L_feat_batch, WT_L_feat_batch, None, None, task=task, task_specific=True)
                        final_input_feat = torch.cat([feat_h, feat_wt_h, feat_l, feat_wt_l], dim=-1)

                        if torch.isnan(final_input_feat).any():
                            print("Train Tensor Contains Nan!")

                        if torch.isnan(label_batch).any():
                            print("Train Label Contains Nan!")

                        X_train.append(final_input_feat.detach().cpu())
                        y_train.append(label_batch.detach().cpu())
                        continue

                    # print(feat_h.shape, feat_wt_h.shape, feat_l.shape, feat_wt_l.shape)

                    if embed_type in ['ours_bb', 'ours_protbert', 'ours_esm1b']:
                        # project all feature dimensions down to 256 (for our model):
                        feat_h, _ = pretrained_H(feat_H_batch, feat_H_batch, None, None, task=0, return2=True)
                        feat_wt_h, _ = pretrained_H(WT_H_feat_batch, WT_H_feat_batch, None, None, task=0, return2=True)
                        feat_l, _ = pretrained_L(feat_L_batch, feat_L_batch, None, None, task=0, return2=True)
                        feat_wt_l, _ = pretrained_L(WT_L_feat_batch, WT_L_feat_batch, None, None, task=0, return2=True)

                    if embed_type in ['bb', 'esm1b', 'protbert']:
                        # for other embeddings (ESM, ProtBERT, etc.)
                        feat_h = feat_H_batch
                        feat_wt_h = WT_H_feat_batch
                        feat_l = feat_L_batch
                        feat_wt_l = WT_L_feat_batch


                pred_batch = model(feat_h, feat_wt_h, feat_l, feat_wt_l)

                # update the parameters (loss, backprop, etc.)
                loss = loss_fn(pred_batch, label_batch)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if k == len(train_loader_des) - 1:
                    print("TRAIN Epoch {}, Batch {}, Loss: {}".format(i, k, total_loss/(k+1)))

            

            # print("********************")

            # validate the model
            model.eval()
            total_loss = 0

            for k, (feat_H_batch, feat_L_batch, label_batch) in enumerate(val_loader_des):

                # predict the label using the model
                feat_H_batch, feat_L_batch, label_batch = feat_H_batch.to(device), feat_L_batch.to(device), label_batch.to(device)

                # # create the WT features batch:
                WT_H_feat_batch = WT_H_feat.repeat(feat_H_batch.shape[0], 1, 1)
                WT_L_feat_batch = WT_L_feat.repeat(feat_L_batch.shape[0], 1, 1)

                with torch.no_grad():

                    # for RIDGE REGRESSION:
                    if ridge:
                        task = 0
                        feat_h, _ = pretrained_H(feat_H_batch, feat_H_batch, None, None, task=task, task_specific=True)
                        feat_wt_h, _ = pretrained_H(WT_H_feat_batch, WT_H_feat_batch, None, None, task=task, task_specific=True)
                        feat_l, _ = pretrained_L(feat_L_batch, feat_L_batch, None, None, task=task, task_specific=True)
                        feat_wt_l, _ = pretrained_L(WT_L_feat_batch, WT_L_feat_batch, None, None, task=task, task_specific=True)
                        final_input_feat = torch.cat([feat_h, feat_wt_h, feat_l, feat_wt_l], dim=-1)

                        if torch.isnan(final_input_feat).any():
                            print("VAL Tensor Contains Nan!")

                        if torch.isnan(label_batch).any():
                            print("VAL Label Contains Nan!")

                        X_train.append(final_input_feat.detach().cpu())
                        y_train.append(label_batch.detach().cpu())
                        continue

                    if embed_type in ['ours_bb', 'ours_protbert', 'ours_esm1b']:
                        # project all feature dimensions down to 256 (for our model):
                        feat_h, _ = pretrained_H(feat_H_batch, feat_H_batch, None, None, task=0, return2=True)
                        feat_wt_h, _ = pretrained_H(WT_H_feat_batch, WT_H_feat_batch, None, None, task=0, return2=True)
                        feat_l, _ = pretrained_L(feat_L_batch, feat_L_batch, None, None, task=0, return2=True)
                        feat_wt_l, _ = pretrained_L(WT_L_feat_batch, WT_L_feat_batch, None, None, task=0, return2=True)

                    if embed_type in ['bb', 'esm1b', 'protbert']:
                        # for other embeddings (ESM, ProtBERT, etc.)
                        feat_h = feat_H_batch
                        feat_wt_h = WT_H_feat_batch
                        feat_l = feat_L_batch
                        feat_wt_l = WT_L_feat_batch


                    pred_batch = model(feat_h, feat_wt_h, feat_l, feat_wt_l)

                # update the parameters (loss, backprop, etc.)
                loss = loss_fn(pred_batch, label_batch)
                total_loss += loss.item()

                if k == len(val_loader_des) - 1:
                    print("VAl Epoch {}, Batch {}, Loss: {}".format(i, k, total_loss/(k+1)))

            print("----------------------------------------------")

            if ridge:
                break
            
            scheduler.step()


    # FIT the RIDGE REGRESSION MODEL HERE:
    # X, y = torch.cat(X_train, dim=0).numpy(), torch.cat(y_train, dim=0).numpy()
    # assert len(X) == len(y)
    # kf = KFold(n_splits=5, shuffle=True)
    # alphas = [0.1, 0.5, 1, 5, 10]
    # print("Cross Validation over alpha values: {}".format(alphas))
    # for i, (train_index, val_index) in enumerate(kf.split(X)):
    #     X_train, X_val = X[train_index], X[val_index]
    #     y_train, y_val = y[train_index], y[val_index]
    #     alpha = alphas[i]
    #     clf = Ridge(alpha=alpha)
    #     clf.fit(X_train, y_train)

    #     # prediction on Val:
    #     sc = clf.score(X_val, y_val)
    #     print("Alpha: {}, Ridge Regression Score: {}".format(alpha, sc))

    # sys.exit("Done Testing Cross Validation")


    # feed the model through the test set
    model.eval()
    losses = []
    labels_total, preds_total = [], []

    for k, (feat_H_batch, feat_L_batch, label_batch) in enumerate(test_loader_des): #, total=len(test_loader_des):

        # predict the label using the model
        feat_H_batch, feat_L_batch, label_batch = feat_H_batch.to(device), feat_L_batch.to(device), label_batch.to(device)

        # # create the WT features batch:
        WT_H_feat_batch = WT_H_feat.repeat(feat_H_batch.shape[0], 1, 1)
        WT_L_feat_batch = WT_L_feat.repeat(feat_L_batch.shape[0], 1, 1)

        with torch.no_grad():

            # for RIDGE REGRESSION
            if ridge:
                task = 0
                feat_h, _ = pretrained_H(feat_H_batch, feat_H_batch, None, None, task=task, task_specific=True)
                feat_wt_h, _ = pretrained_H(WT_H_feat_batch, WT_H_feat_batch, None, None, task=task, task_specific=True)
                feat_l, _ = pretrained_L(feat_L_batch, feat_L_batch, None, None, task=task, task_specific=True)
                feat_wt_l, _ = pretrained_L(WT_L_feat_batch, WT_L_feat_batch, None, None, task=task, task_specific=True)
                final_input_feat = torch.cat([feat_h, feat_wt_h, feat_l, feat_wt_l], dim=-1)

                if torch.isnan(final_input_feat).any():
                    print("Test Tensor Contains Nan!")

                if torch.isnan(label_batch).any():
                    print("Test Label Contains Nan!")

                X_test.append(final_input_feat.detach().cpu())
                y_test.append(label_batch.detach().cpu())
                continue

            if embed_type in ['ours_bb', 'ours_protbert', 'ours_esm1b']:
                # project all feature dimensions down to 256 (for our model):
                feat_h, _ = pretrained_H(feat_H_batch, feat_H_batch, None, None, task=0, return2=True)
                feat_wt_h, _ = pretrained_H(WT_H_feat_batch, WT_H_feat_batch, None, None, task=0, return2=True)
                feat_l, _ = pretrained_L(feat_L_batch, feat_L_batch, None, None, task=0, return2=True)
                feat_wt_l, _ = pretrained_L(WT_L_feat_batch, WT_L_feat_batch, None, None, task=0, return2=True)

            if embed_type in ['bb', 'esm1b', 'protbert']:
                # for other embeddings (ESM, ProtBERT, etc.)
                feat_h = feat_H_batch
                feat_wt_h = WT_H_feat_batch
                feat_l = feat_L_batch
                feat_wt_l = WT_L_feat_batch

            pred_batch = model(feat_h, feat_wt_h, feat_l, feat_wt_l)

        #-----------------------------------------------------

        labels_total.append(label_batch)
        preds_total.append(pred_batch)

        # update the parameters (loss, backprop, etc.)
        loss = loss_fn(pred_batch, label_batch)
        losses.append(loss.item())

    save_dict = dict()

    # perform RIDGE REGRESSION HERE:
    if ridge:
        X_train, y_train = torch.cat(X_train, dim=0).numpy(), torch.cat(y_train, dim=0).numpy()
        X_test, y_test = torch.cat(X_test, dim=0).numpy(), torch.cat(y_test, dim=0).numpy()
        clf = Ridge(alpha=0.5)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        for j in range(y_pred.shape[-1]):
            spear = evaluate_spearman(y_pred[:, j], y_test[:, j])
            print("Spearman Rank Scores for {}: {}".format(pred_cols[j], spear))
            save_dict[pred_cols[j]] = (y_pred[:, j], y_test[:, j])
        test_score = clf.score(X_test, y_test)
        print("Train/Test Split: {}, Ridge Regression Score on Test: {:.2f}".format(traintest_split, test_score))

        with open('/data/cb/rsingh/work/antibody/ci_data/processed/desautels/plots/{}_spl{}.p'.format(args.logname, traintest_split), 'wb') as p:
            pickle.dump(save_dict, p)

        return


    # Calculate Average Loss over the TEST SET:
    print("TEST SET Average LOSS: {}".format(sum(losses)/len(losses)))

    # calculate spearman rank for each value:
    labels_total = torch.cat(labels_total, dim=0)
    preds_total = torch.cat(preds_total, dim=0)

    for j in range(labels_total.shape[-1]):
        spear = evaluate_spearman(preds_total[:, j], labels_total[:, j])
        print("Split: {}, Spearman Rank Scores for {}: {}".format(traintest_split, pred_cols[j], spear))
        save_dict[pred_cols[j]] = (preds_total[:, j], labels_total[:, j], spear)

    with open('/data/cb/rsingh/work/antibody/ci_data/processed/desautels/plots/{}_spl{}.p'.format(args.logname, traintest_split), 'wb') as p:
        pickle.dump(save_dict, p)


def calc_topk_overlap(args, traintest_split, r1, r2):
    plots_path = '/data/cb/rsingh/work/antibody/ci_data/processed/desautels/plots'
    with open(os.path.join(plots_path, '{}_spl{}.p'.format(args.logname, traintest_split)), 'rb') as p:
        results = pickle.load(p)

    all_overlaps = []
    for key in results.keys():
        preds_total, labels_total = results[key][:2]

        k1 = int(r1*len(labels_total))
        k2 = int(r2*len(preds_total))

        if not torch.is_tensor(labels_total):
            labels_total = torch.Tensor(labels_total)
            preds_total = torch.Tensor(preds_total)

        labels_topk_idx = torch.topk(labels_total, k1).indices.detach().cpu()
        preds_topk_idx = torch.topk(preds_total, k2).indices.detach().cpu()

        overlap = np.intersect1d(labels_topk_idx, preds_topk_idx)
        print("Score {}. Overlap Ratio: {}/{} = {:.2f}".format(key, len(overlap), k1, len(overlap)/k1))
        all_overlaps.append(len(overlap)/k1)

    print("Average Overlap: {:.2f}".format(np.mean(all_overlaps)))




def preprocess_raw_data():
    """
    Take the raw Desautels dataset and generate the heavy and light chain sequence fasta files
    """

    df = pd.read_csv('../desaut_things/Desautels_m396_seqs.csv')

    for chain in ['heavy_chains', 'light_chains']:
        list_save = []
        for idx, row in df.iterrows():
            list_save.append('>'+row['Antibody_ID']+'\n')

            if chain == 'heavy_chains':
                seq = row['Antibody_Sequence'][:245]
            else:
                seq = row['Antibody_Sequence'][245:]

            if idx != -1:
                seq += '\n'
            list_save.append(seq)

        with open('/data/cb/rsingh/work/antibody/ci_data/raw/desautels/{}.fasta'.format(chain), 'w') as f:
            f.writelines(list_save)

    print("Divided the chains into heavy and light chains and saved as fasta files.")


def generate_hdf5_onehot(args, file_name):
    reload_models_to_device(args.device_num)

    with h5py.File('/data/cb/rsingh/work/antibody/ci_data/processed/desautels/{}.h5'.format(file_name), 'w') as g:

        dset_h = g.create_dataset('H', (89263, 32, 21))
        dset_l = g.create_dataset('L', (89263, 29, 21))

        for chain_type in ['H', 'L']:
            if chain_type == 'H':
                dset = dset_h
                list_path = '/data/cb/rsingh/work/antibody/ci_data/raw/desautels/heavy_chains.fasta'
                cdr_idx = H_cdr_idx
            else:
                dset = dset_l
                list_path = '/data/cb/rsingh/work/antibody/ci_data/raw/desautels/light_chains.fasta'
                cdr_idx = L_cdr_idx
            
            
            with open(list_path, 'r') as f:
                seqs_list = f.read().splitlines()

            k=50
            for i in tqdm(range(0, len(seqs_list), 2)):
                seq = seqs_list[i+1]

                seq_num = torch.Tensor([all_aas.index(char) for char in seq])
                seq_num_cdr = seq_num[cdr_idx].type(torch.int64)
                one_hot = F.one_hot(seq_num_cdr, num_classes=len(all_aas))
                dset[int(i/2),:,:] = one_hot

                    # continue


                # prot = ProteinEmbedding(seq, chain_type, dev=1, fold=1)
                # prot.embed_seq('beplerberger')
                # prot.create_cdr_mask()
                # kmut_matr_h = prot.create_kmut_matrix(num_muts=k, embed_type='beplerberger')
                # cdr_embed = prot.create_cdr_embedding(kmut_matr_h, sep = False, mask = True)


                # dset[int(i/2), :, :] = cdr_embed
                # del cdr_embed

def generate_h5_from_tens(args):
    embed_dir_H = '/data/cb/rsingh/work/antibody/ci_data/processed/desautels/desautels_embeds_{}_H'.format(args.embed_type)
    embed_dir_L = '/data/cb/rsingh/work/antibody/ci_data/processed/desautels/desautels_embeds_{}_L'.format(args.embed_type)
    file_name = 'desautels_cdrembed_{}'.format(args.embed_type)
    file_name = 'desautels_bb'

    # valid_nums = list(range(89263))
    # invalid_nums_1 = [9, 79, 979, 8979]
    embed_size_dict = {'bb': 6165, 'ours_bb': 6165, 'ours_esm1b': 1280, 'ours_protbert': 1024}

    invalid_nums = []

    H_CDR_LEN, L_CDR_LEN = 245, 213
    embed_size = embed_size_dict[args.embed_type] + 4
    embed_size = 6165
    # Check if all antibodies have the above CDR lens! (need to do this just once)
    total_size = len(os.listdir(embed_dir_H))
    # total_size = 5000 # FOR TESTING!
    for num in tqdm(range(total_size)):
        with open(os.path.join(embed_dir_H, '{}.p'.format(num)), 'rb') as p:
            tens_H = pickle.load(p)
        with open(os.path.join(embed_dir_L, '{}.p'.format(num)), 'rb') as p:
            tens_L = pickle.load(p)
        if tens_H.shape[0] != H_CDR_LEN or tens_L.shape[0] != L_CDR_LEN:
            invalid_nums.append(num)
            continue
        if tens_H is None or tens_L is None:
            invalid_nums.append(num)

    # invalid_nums = list(set(invalid_nums_1) | set(invalid_nums_2))

    filtered_size = total_size - len(invalid_nums)
    print(invalid_nums)

    with open('/data/cb/rsingh/work/antibody/ci_data/processed/desautels/invalid_idx.p', 'wb') as f:
        pickle.dump(invalid_nums, f)

    with h5py.File('/data/cb/rsingh/work/antibody/ci_data/processed/desautels/{}.h5'.format(file_name), 'w') as g:

        count = 0
        dset_h = g.create_dataset('H', (total_size, H_CDR_LEN, embed_size))
        dset_l = g.create_dataset('L', (total_size, L_CDR_LEN, embed_size))
        for num in tqdm(range(total_size)):
            if num in invalid_nums:
                continue
            with open(os.path.join(embed_dir_H, '{}.p'.format(num)), 'rb') as p:
                tens_H = pickle.load(p)
            with open(os.path.join(embed_dir_L, '{}.p'.format(num)), 'rb') as p:
                tens_L = pickle.load(p)

            try:
                dset_h[num, :, :] = tens_H
                dset_l[num, :, :] = tens_L
                count += 1
            except:
                print(f"idx {num} invalid! Something is wrong.")

        assert count == filtered_size

        print(f"# of valid entries for {args.embed_type}: {count}")
        print("Desautels H dataset shape:", dset_h.shape)
        print("Desautels L dataset shape:", dset_l.shape)

    # print("Additionally {} are invalid".format(additional_invalid))




def init(dev_num, c_type, emb_type):
    global device_n, chain_type, embed_type
    chain_type = c_type
    device_n = dev_num
    embed_type = emb_type
    device = torch.device("cuda:{}".format(device_n) if torch.cuda.is_available() else "cpu")
    reload_models_to_device(device_n)

def work_parallel(pair):
    dev, seq = pair
    prot = ProteinEmbedding(seq, chain_type, embed_device=f"cuda:{device_n}", dev=dev, fold=device_n)

    prot.embed_seq(embed_type)
    embedding = prot.embedding
    return (dev, embedding.detach().cpu().numpy())

    try:
        prot.create_cdr_mask()
    except:
        return (None, None)

    k=50
    cdr_embed = prot.create_cdr_specific_embedding(embed_type=embed_type, k=k)

    return (dev, cdr_embed.detach().cpu().numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, help='which code path to run.')
    parser.add_argument('--chain_type', type=str, help='Which chain are we processing', choices=['H', 'L'])
    parser.add_argument('--is_onehot', type=bool, help='Which model are we running: ours/onehot')
    parser.add_argument('--device_num', type=int, help='GPU device we are using.', default=0)
    parser.add_argument('--embed_type', type=str, help='Embedding Type')
    parser.add_argument('--logname', type=str, help='Name of the file for saving')

    args = parser.parse_args()

    if args.mode == '1_preprocess_raw_data':
        preprocess_raw_data()

    elif args.mode == '2_generate_features_onehot':
        generate_hdf5_onehot(args, file_name='desautels_onehot')

    elif args.mode == '3_generate_hdf5':
        # load the sequences:
        c_type = args.chain_type
        des_proc = '/data/cb/rsingh/work/antibody/ci_data/processed/desautels'
        des_raw = '/data/cb/rsingh/work/antibody/ci_data/raw/desautels'

        if c_type == 'H':
            list_path = os.path.join(des_raw, 'heavy_chains.fasta')
        else:
            list_path = os.path.join(des_raw, 'light_chains.fasta')

        with open(list_path, 'r') as f:
            seqs_list = f.read().splitlines()

        input_seqs = []
        for i in range(0, len(seqs_list), 2):
            input_seqs.append((int(i/2), seqs_list[i+1]))

        #TESTING with few input seqs:
        # input_seqs = input_seqs[:1000]

        import multiprocessing as mp
        from multiprocessing import get_context
        embeds_result = []

        with get_context('spawn').Pool(processes=16, initializer=init, initargs=[args.device_num, c_type, args.embed_type]) as p:
            for result in tqdm(p.imap_unordered(work_parallel, input_seqs, chunksize=4), total=len(input_seqs)):
                idx_, emb = result
                with open(os.path.join(des_proc, f'desautels_embeds_bb_{c_type}/{idx_}.p'), 'wb') as a:
                    pickle.dump(emb, a)


    elif args.mode == '4_train_model':
        for s in [0.2, 0.1, 0.05, 0.02, 0.01, 0.005]: # 0.2, 0.1, 0.05, 0.02, 0.01, 
            print("Start Training...")
            start_time = time.time()
            print("Train/Test Split: {}".format(s))
            if args.is_onehot:
                print("Training one-hot embedding based model")
                train_model_onehot(args, device_num = args.device_num, num_epochs = 100, batch_size = 32, 
                        exec_type = 'train', lr=0.01, traintest_split = s)
            else:
                print("Training regular embedding based model")
                train_model(args, device_num = args.device_num, num_epochs = 100, batch_size = 32, 
                        exec_type = 'train', lr=0.01, ridge=True, traintest_split = s, embed_type=args.embed_type)
            
            end_time = time.time()
            print("Done Training! Took {:.1f} Seconds.".format(end_time-start_time))
            print()

    elif args.mode == '5_generate_features':
        generate_features(args)

    elif args.mode == 'generate_h5_from_tens':
        generate_h5_from_tens(args)

    elif args.mode == '7_calc_topk_overlap':
        calc_topk_overlap(args, traintest_split = 0.02, r1 = 0.1, r2 = 0.1)

    else:
        assert False