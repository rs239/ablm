import torch
import os, sys
import pickle
from tqdm import tqdm
import time
import argparse

from torch import nn
from torch import optim as opt
from torch import autograd
from torch.optim.lr_scheduler import MultiStepLR

from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

sys.path.append('../antibody_ml')
from model import AntibodyNetMulti, DesautelsPredictor, MultiTaskLossWrapper
from dataloader import create_dataloader_desautels
from utils import evaluate_spearman


pred_cols = ['FoldX_Average_Whole_Model_DDG', 'FoldX_Average_Interface_Only_DDG',
            'Statium', 'Sum_of_Rosetta_Flex_single_point_mutations',
            'Sum_of_Rosetta_Total_Energy_single_point_mutations']

def train_model_baseline(device_num, num_epochs, batch_size, exec_type, lr, ridge, traintest_split):
    device = torch.device("cuda:{}".format(device_num) if torch.cuda.is_available() else "cpu")

    # load the model/optimizer
    model = DesautelsPredictor(embed_dim = 20).to(device)
    loss_fn = nn.MSELoss()
    optimizer = opt.SGD(model.parameters(), lr=lr)
    scheduler = MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)

    print("Building dataloader...")
    # load the dataloader
    train_loader_des, val_loader_des, test_loader_des = create_dataloader_desautels(batch_size=batch_size, 
                                                            traintest_split = traintest_split, baseline=True)
    print("Done building dataloader!")

    # load WT features:
    with open('/net/scratch3/scratch3-3/chihoim/desaut_things/WT_feats/WT_H_baseline.p', 'rb') as f:
        WT_H_feat = pickle.load(f).to(device)
    with open('/net/scratch3/scratch3-3/chihoim/desaut_things/WT_feats/WT_L_baseline.p', 'rb') as f:
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
        print("Spearman Rank Scores for {}: {}".format(pred_cols[j], spear))
        save_dict[pred_cols[j]] = (preds_total[:, j], labels_total[:, j])

    with open('/net/scratch3/scratch3-3/chihoim/desaut_things/plots/lstm_baseline_spl{}.p'.format(traintest_split), 'wb') as p:
        pickle.dump(save_dict, p)



def train_model(device_num, num_epochs, batch_size, exec_type, lr, ridge, traintest_split):
    device = torch.device("cuda:{}".format(device_num) if torch.cuda.is_available() else "cpu")

    # load the model/optimizer
    model = DesautelsPredictor(embed_dim = 50).to(device)
    loss_fn = nn.MSELoss()
    optimizer = opt.SGD(model.parameters(), lr=lr)
    scheduler = MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)

    # TODO: load the pre-trained multi-task training model:
    pretrained_H = AntibodyNetMulti(embed_dim=2200, lstm_layers=1).to(device)
    pretrained_L = AntibodyNetMulti(embed_dim=2200, lstm_layers=1).to(device)
    best_arch_path = '/net/scratch3/scratch3-3/chihoim/model_ckpts/best_pretrained'
    H_best_path = os.path.join(best_arch_path, 'multi_H_best.pt')
    L_best_path = os.path.join(best_arch_path, 'multi_L_best.pt')
    checkpoint_H = torch.load(H_best_path, map_location=device)
    checkpoint_L = torch.load(L_best_path, map_location=device)
    pretrained_H.load_state_dict(checkpoint_H['model_state_dict'])
    pretrained_L.load_state_dict(checkpoint_L['model_state_dict'])
    pretrained_H.eval()
    pretrained_L.eval()

    print("Building dataloader...")
    # load the dataloader
    train_loader_des, val_loader_des, test_loader_des = create_dataloader_desautels( 
                                                           batch_size=batch_size, traintest_split = traintest_split)
    print("Done building dataloader!")

    # load WT features:
    with open('/net/scratch3/scratch3-3/chihoim/desaut_things/WT_H.p', 'rb') as f:
        WT_H_feat = pickle.load(f).to(device)
    with open('/net/scratch3/scratch3-3/chihoim/desaut_things/WT_L.p', 'rb') as f:
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

                # create the WT features batch:
                WT_H_feat_batch = WT_H_feat.repeat(feat_H_batch.shape[0], 1, 1)
                WT_L_feat_batch = WT_L_feat.repeat(feat_L_batch.shape[0], 1, 1)

                # project all feature dimensions down to 50:
                with torch.no_grad():
                    feat_h = pretrained_H.project(feat_H_batch[:,:,-2204:-4])
                    feat_wt_h = pretrained_H.project(WT_H_feat_batch[:,:,-2204:-4])
                    feat_l = pretrained_L.project(feat_L_batch[:,:,-2204:-4])
                    feat_wt_l = pretrained_L.project(WT_L_feat_batch[:,:,-2204:-4])

                # RIDGE REGRESSION:
                if ridge:
                    task = 0
                    with torch.no_grad():
                        x1 = pretrained_H.recurrent(torch.cat([feat_h, feat_H_batch[:,:,-4:]], dim=-1), task)
                        x2 = pretrained_H.recurrent(torch.cat([feat_wt_h, WT_H_feat_batch[:,:,-4:]], dim=-1), task)
                        x3 = pretrained_L.recurrent(torch.cat([feat_l, feat_L_batch[:,:,-4:]], dim=-1), task)
                        x4 = pretrained_L.recurrent(torch.cat([feat_wt_l, WT_L_feat_batch[:,:,-4:]], dim=-1), task)
                        final_input_feat = torch.cat([x1, x2, x3, x4], dim=-1)

                        X_train.append(final_input_feat.detach().cpu())
                        y_train.append(label_batch.detach().cpu())
                    continue

                pred_batch = model(feat_h, feat_wt_h, feat_l, feat_wt_l)

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

                with torch.no_grad():
                    # project all feature dimensions down to 50:
                    feat_h = pretrained_H.project(feat_H_batch[:,:,-2204:-4])
                    feat_wt_h = pretrained_H.project(WT_H_feat_batch[:,:,-2204:-4])
                    feat_l = pretrained_L.project(feat_L_batch[:,:,-2204:-4])
                    feat_wt_l = pretrained_L.project(WT_L_feat_batch[:,:,-2204:-4])

                    # RIDGE REGRESSION:
                    if ridge:
                        task = 0
                        x1 = pretrained_H.recurrent(torch.cat([feat_h, feat_H_batch[:,:,-4:]], dim=-1), task)
                        x2 = pretrained_H.recurrent(torch.cat([feat_wt_h, WT_H_feat_batch[:,:,-4:]], dim=-1), task)
                        x3 = pretrained_L.recurrent(torch.cat([feat_l, feat_L_batch[:,:,-4:]], dim=-1), task)
                        x4 = pretrained_L.recurrent(torch.cat([feat_wt_l, WT_L_feat_batch[:,:,-4:]], dim=-1), task)
                        final_input_feat = torch.cat([x1, x2, x3, x4], dim=-1)

                        X_train.append(final_input_feat.detach().cpu())
                        y_train.append(label_batch.detach().cpu())
                        continue

                    pred_batch = model(feat_h, feat_wt_h, feat_l, feat_wt_l)

                # update the parameters (loss, backprop, etc.)
                loss = loss_fn(pred_batch, label_batch)
                total_loss += loss.item()

                if k == len(val_loader_des) - 1:
                    print("VAl Epoch {}, Batch {}, Loss: {}".format(i, k, total_loss/(k+1)))

            print("----------------------------------------------")


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
            # project all feature dimensions down to 50:
            feat_h = pretrained_H.project(feat_H_batch[:,:,-2204:-4])
            feat_wt_h = pretrained_H.project(WT_H_feat_batch[:,:,-2204:-4])
            feat_l = pretrained_L.project(feat_L_batch[:,:,-2204:-4])
            feat_wt_l = pretrained_L.project(WT_L_feat_batch[:,:,-2204:-4])

            if ridge:
                task = 0
                x1 = pretrained_H.recurrent(torch.cat([feat_h, feat_H_batch[:,:,-4:]], dim=-1), task)
                x2 = pretrained_H.recurrent(torch.cat([feat_wt_h, WT_H_feat_batch[:,:,-4:]], dim=-1), task)
                x3 = pretrained_L.recurrent(torch.cat([feat_l, feat_L_batch[:,:,-4:]], dim=-1), task)
                x4 = pretrained_L.recurrent(torch.cat([feat_wt_l, WT_L_feat_batch[:,:,-4:]], dim=-1), task)
                final_input_feat = torch.cat([x1, x2, x3, x4], dim=-1)

                X_test.append(final_input_feat.detach().cpu())
                y_test.append(label_batch.detach().cpu())
                continue

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

        with open('/net/scratch3/scratch3-3/chihoim/desaut_things/plots/base440Ridge_spl{}.p'.format(traintest_split), 'wb') as p:
            pickle.dump(save_dict, p)

        return


    # Calculate Average Loss over the TEST SET:
    print("TEST SET Average LOSS: {}".format(sum(losses)/len(losses)))

    # calculate spearman rank for each value:
    labels_total = torch.cat(labels_total, dim=0)
    preds_total = torch.cat(preds_total, dim=0)

    for j in range(labels_total.shape[-1]):
        spear = evaluate_spearman(preds_total[:, j], labels_total[:, j])
        print("Spearman Rank Scores for {}: {}".format(pred_cols[j], spear))
        save_dict[pred_cols[j]] = (preds_total[:, j], labels_total[:, j])

    with open('/net/scratch3/scratch3-3/chihoim/desaut_things/plots/lstm_oursMLP_spl{}.p'.format(traintest_split), 'wb') as p:
        pickle.dump(save_dict, p)


def preprocess_raw_data():
    """
    Take the raw Desautels dataset and generate the heavy and light chain sequence fasta files
    """

    df = pd.read_csv('')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, help='which code path to run.',
                        choices=['1_preprocess_raw_data', '2_generate_hdf5', '3_train_model'])

    args = parser.parse_args()

    if args.mode == '1_preprocess_raw_data':
        pass

    elif args.mode == '2_generate_hdf5':
        pass

    elif args.mode == '3_train_model':
        for s in [0.2, 0.1, 0.05, 0.02, 0.01, 0.005]:
            print("Start Training...")
            start_time = time.time()
            print("Train/Test Split: {}".format(s))
            train_model(device_num = 2, num_epochs = 1, batch_size = 32, 
                        exec_type = 'train', lr=0.1, ridge=True, traintest_split = s)
            # train_model_baseline(device_num = 2, num_epochs = 1, batch_size = 32, 
            #             exec_type = 'train', lr=0.1, ridge=True, traintest_split = s)
            end_time = time.time()
            print("Done Training! Took {:.1f} Seconds.".format(end_time-start_time))
            print()