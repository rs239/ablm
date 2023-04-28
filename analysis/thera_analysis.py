import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import os, sys
import h5py
import pickle
import random

from torch.nn.utils.rnn import pad_sequence

from abmap.abmap_augment import ProteinEmbedding
from abmap.model import AbMAPAttn
from abmap.plm_embed import reload_models_to_device

def generate_embeddings(args):
    reload_models_to_device(args.device_num)

    device = torch.device("cuda:{}".format(args.device_num) if torch.cuda.is_available() else "cpu")

    data_dir = "/data/cb/rsingh/work/antibody/ci_data/raw/thera-sabdab/TheraSAbDab_SeqStruc_OnlineDownload.csv"

    df = pd.read_csv(data_dir)
    # focus only on Whole mAb
    df = df.loc[df['Format'] == 'Whole mAb']
    df.reset_index(drop=True, inplace=True)

    chains_feats = {'H':[], 'L':[]}
    phaseI, est_status, discontinuous = [], [], []
    thera_ids = []
    for idx, row in tqdm(df.iterrows(), total = df.shape[0]):

        thera_id = row['Therapeutic']
        seq_h, seq_l = row['Heavy Sequence'], row['Light Sequence']

        phaseI.append("Phase-I" in row["Highest_Clin_Trial (Oct '21)"])
        est_status.append("Active" in row["Est. Status"])
        discontinuous.append("Discontinued" in row["Est. Status"])

        for seq, chain_type in [(seq_h, 'H'), (seq_l, 'L')]:
            prot = ProteinEmbedding(seq, chain_type, embed_device="cuda:{}".format(args.device_num), dev=0)

            try:
                prot.create_cdr_mask()
            except:
                chains_feats[chain_type].append(None)
                continue

            k=50
            prot.embed_seq('beplerberger')
            kmut_matr_h = prot.create_kmut_matrix(num_muts=k, embed_type='beplerberger')
            cdr_embed = prot.create_cdr_embedding(kmut_matr_h, sep = False, mask = True)

            chains_feats[chain_type].append(cdr_embed)
            if chain_type == 'H':
                thera_ids.append(thera_id)

    # only keep features where both heavy and light chain features are present:
    H_final, L_final = [], []
    phaseI_final, est_status_final, discontinuous_final = [], [], []
    for i in range(len(chains_feats['H'])):
        if chains_feats['H'][i] is not None and chains_feats['L'][i] is not None:
            H_final.append(chains_feats['H'][i])
            L_final.append(chains_feats['L'][i])
            phaseI_final.append(phaseI[i])
            est_status_final.append(est_status[i])
            discontinuous_final.append(discontinuous[i])


    save_dir = '/data/cb/rsingh/work/antibody/ci_data/processed/thera-sabdab/cdrembed_maskaug4_n{}.h5'.format(len(chains_feats['H']))
    if os.path.exists(save_dir):
        os.system('rm -f {}'.format(save_dir))

    # pack the dataset into a single tensor
    cdrembeds_H = pad_sequence(H_final, batch_first=True)
    cdrembeds_L = pad_sequence(L_final, batch_first=True)
    phaseI_final = np.array(phaseI_final)
    est_status_final = np.array(est_status_final)
    discontinuous_final = np.array(discontinuous_final)
    assert len(cdrembeds_H) == len(cdrembeds_L) == len(phaseI_final) == len(est_status_final)

    print("Dataset Shapes, H: {}, L: {}".format(cdrembeds_H.shape, cdrembeds_L.shape))

    # SAVE IT
    with h5py.File(save_dir, "w") as f:
        dset_H = f.create_dataset('H', cdrembeds_H.shape)
        dset_H[:,:,:] = cdrembeds_H

        dset_L = f.create_dataset('L', cdrembeds_L.shape)
        dset_L[:,:,:] = cdrembeds_L

        dset_p = f.create_dataset('Phase-I', phaseI_final.shape)
        dset_p[:] = phaseI_final

        dset_a = f.create_dataset('Active', est_status_final.shape)
        dset_a[:] = est_status_final

        dset_d = f.create_dataset('Discontinued', discontinuous_final.shape)
        dset_d[:] = discontinuous_final

    with open('/data/cb/rsingh/work/antibody/ci_data/processed/thera-sabdab/thera_ids_n{}.p'.format(len(discontinuous_final)), 'wb') as f:
        pickle.dump(thera_ids, f)




def generate_features_from_embeddings(args):
    thera_dir = "/data/cb/rsingh/work/antibody/ci_data/processed/thera-sabdab/cdrembed_maskaug4_n547.h5"
    out_path = "/data/cb/rsingh/work/antibody/ci_data/processed/thera-sabdab/thera_features"
    chain_type = args.chain_type

    with h5py.File(thera_dir, 'r') as f:
        thera_embeds = np.array(f[chain_type])
        print(thera_embeds.shape)
    with open("/data/cb/rsingh/work/antibody/ci_data/processed/thera-sabdab/thera_ids_n547.p", 'rb') as p:
        thera_ids_list = pickle.load(p)

    print(len(thera_ids_list), len(thera_embeds))
    assert len(thera_ids_list) == len(thera_embeds)

    for i in tqdm(range(len(thera_embeds))):
        feat_np = thera_embeds[i:i+1, :, :]
        idx = unpad_embedding(feat_np)
        if idx == 0:
            embedding = torch.FloatTensor(feat_np).to(device)
        else:
            embedding = torch.FloatTensor(feat_np[:,:idx,:]).to(device)

        with torch.no_grad():
            thera_feat, _ = pretrained(embedding, embedding, None, None, task=0, task_specific=True)
            thera_feat = thera_feat.detach().cpu()
        assert thera_feat.shape[-1] == 512
        with open(os.path.join(out_path, "{}_struc_{}.p".format(thera_ids_list[i], chain_type)), 'wb') as p:
            pickle.dump(torch.squeeze(thera_feat), p)
        
        with torch.no_grad():
            thera_feat, _ = pretrained(embedding, embedding, None, None, task=1, task_specific=True)
            thera_feat = thera_feat.detach().cpu()
        assert thera_feat.shape[-1] == 512
        with open(os.path.join(out_path, "{}_func_{}.p".format(thera_ids_list[i], chain_type)), 'wb') as p:
            pickle.dump(torch.squeeze(thera_feat), p)
        
        with torch.no_grad():
            thera_feat, _ = pretrained(embedding, embedding, None, None, task=None, return2=True)
            thera_feat = thera_feat.detach().cpu()
        assert thera_feat.shape[-1] == 256
        with open(os.path.join(out_path, "{}_interm_{}.p".format(thera_ids_list[i], chain_type)), 'wb') as p:
            pickle.dump(torch.squeeze(thera_feat), p)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="which code path to run. see main(..) for details", type=str, choices = ["1_generate_embeddings", "2_generate_features"])
    parser.add_argument("--device_num", help="GPU device for computation.", type=int, default=0)
    parser.add_argument("--extra", help="put this as the LAST option and arbitrary space-separated key=val pairs after that", type=str, nargs='*')

    args = parser.parse_args()
    assert args.mode is not None
    
    extra_args = dict([a.split("=") for a in args.extra]) if args.extra else {}
    args.extra = extra_args
    
    pd.set_option('use_inf_as_na', True)

    
    if args.mode == '1_generate_embeddings':
        generate_embeddings(args)
    elif args.mode == '2_generate_features':
        generate_features(args)
    else:
        assert False

