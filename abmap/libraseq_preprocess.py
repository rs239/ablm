import pandas as pd
import csv
import os, sys
from tqdm import tqdm
import numpy as np
import torch
import pickle

from abmap_augment import ProteinEmbedding
from plm_embed import reload_models_to_device


def create_standardized_dataset(device_num=0):

    reload_models_to_device(device_num)

    libra_data_path = '../data/raw/libraseq'
    cell_lines_path = os.path.join(libra_data_path, 'LIBRA-seq_data_abseqs-LSS', 'cell-lines_mergedDEDUP_2434-01-resub_07-27-2019_18-19-52_abs-LSS.csv')
    d45_path = os.path.join(libra_data_path, 'LIBRA-seq_data_abseqs-LSS', 'd45_mergedDEDUP_2723-resub_07-27-2019_17-38-36_ab-LSS.csv')
    n90_path = os.path.join(libra_data_path, 'LIBRA-seq_data_abseqs-LSS', 'n90_mergedDEDUP_3602-n90_07-27-2019_17-40-05_ab-lss.csv')

    df1 = pd.read_csv(cell_lines_path, keep_default_na=False)
    df2 = pd.read_csv(d45_path, keep_default_na=False)
    df3 = pd.read_csv(n90_path, keep_default_na=False)

    libra_out_path = '../data/processed/libraseq'

    with open(os.path.join(libra_data_path, 'libraseq_standardized_all.csv'), 'w') as f:
        writer = csv.writer(f)

        # column names:
        names = ['id','vh_seq','vl_seq','BG505_CLR_z','CZA97_CLR_z','HA_CLR_z','B41_CLR_z','BG505_N332T_CLR_z','HA_Anhui_CLR_z','HA_NC99_CLR_z','HA_indo_CLR_z','HA_michigan_CLR_z','KNH1144_CLR_z','ZM106_9_CLR_z','ZM197_CLR_z']
        writer.writerow(names)

        count = 0
        X_count_h = 0
        X_count_l = 0
        X_count_1 = 0
        for df, name in [(df1, 'cell-lines'), (df2, 'd45'), (df3, 'n90')]:

            means = []
            stds = []
            for label in names[3:]:
                try:
                    temp = df[label].mean()
                    temp2 = df[label].std()
                    means.append(temp)
                    stds.append(temp2)
                except:
                    means.append(None)
                    stds.append(None)

            for index, row in tqdm(df.iterrows(), total=df.shape[0]):
                iden = 'libraseq_{}_{}'.format(name, row['cell_barcode'])
                vh_seq = row['SEQUENCE_VDJ_x']
                vl_seq = row['SEQUENCE_VDJ_y']

                if vh_seq != '':
                    gene_path = os.path.join(libra_data_path, 'temp', 'vh_gene_temp.txt')
                    pep_path = os.path.join(libra_data_path, 'temp', 'prot_temp_vh.txt')

                    with open(gene_path, 'w') as fh:
                        fh.write(vh_seq)

                    os.system('../tools/transeq -sequence {} -outseq {} -trim True -clean True'.format(gene_path, pep_path))
                    with open(pep_path, 'r') as ph:
                        pep_h = "".join(ph.read().splitlines()[1:])

                else:
                    pep_h = 'N/A'

                if vl_seq != '':
                    gene_path = os.path.join(libra_data_path, 'temp', 'vl_gene_temp.txt')
                    pep_path = os.path.join(libra_data_path, 'temp', 'prot_temp_vl.txt')

                    with open(gene_path, 'w') as fl:
                        fl.write(vl_seq)

                    os.system('../tools/transeq -sequence {} -outseq {} -trim True -clean True'.format(gene_path, pep_path))
                    with open(pep_path, 'r') as pl:
                        pep_l = "".join(pl.read().splitlines()[1:])
                else:
                    pep_l = 'N/A'

                content = [iden, pep_h, pep_l]

                if 'X' in pep_h:
                    X_count_h += 1
                if 'X' in pep_l:
                    X_count_l += 1
                # if pep_h.count('X') == 1 or pep_l.count('X') == 1:
                #     X_count_1 += 1

                # take out seqs with only one of H or L Chain
                if pep_h == 'N/A' and pep_l == 'N/A':
                    continue

                # take out seqs with more than one 'X'
                if pep_h.count('X') > 1 or pep_l.count('X') > 1:
                    continue

                # filter out seqs with total CDR length < 10:
                prot = ProteinEmbedding(pep_h, 'H', dev=device_num)
                try:
                    prot.create_cdr_mask()
                except:
                    continue
                if prot.cdr_mask.count_nonzero() < 10:
                    continue

                prot = ProteinEmbedding(pep_l, 'L', dev=device_num)
                try:
                    prot.create_cdr_mask()
                except:
                    continue
                if prot.cdr_mask.count_nonzero() < 10:
                    continue


                # write the z scores
                for i, label in enumerate(names[3:]):
                    try:
                        temp = (row[label]-means[i])/stds[i]
                        content.append(temp)
                    except:
                        content.append('N/A')

                writer.writerow(content)
                count += 1

    print(count)
    print('Number of seqs with X in heavy chain:', X_count_h)
    print('Number of seqs with X in light chain:', X_count_l)
    # print('percentage of seqs with just one X:', int(X_count_1/count*100))


def divide_sets():
    libra_path = '../data/processed/libraseq'

    df = pd.read_csv(os.path.join(libra_path, 'libraseq_standardized_all.csv'), keep_default_na=False)

    # drop seqs outside of length range 27 < l < 35:
    # drop_list = []
    # embeds_dir = '/data/cb/rsingh/work/antibody/ci_data/processed/libraseq/cdrembed_maskaug4/beplerberger'
    # for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
    #     id_ = row['id']
    #     for chain in ['H', 'L']:
    #         embed_path = os.path.join(embeds_dir, '{}_cat2_{}_k100.p'.format(id_, chain))
    #         with open(embed_path, 'rb') as f:
    #             b = pickle.load(f)
    #         if b.shape[0] < 27 or b.shape[0] > 35:
    #             drop_list.append(idx)
    # drop_list = list(set(drop_list))
    # print(len(drop_list))
    # df = df.drop(drop_list)

    df_set1 = df.sample(frac=0.8)
    df_set2 = df.drop(df_set1.index)
    print("{} split into {} and {}".format(df.shape[0], df_set1.shape[0], df_set2.shape[0]))
    df_set1.to_csv(os.path.join(libra_path, 'libraseq_standardized_Set1.csv'), index=False)
    df_set2.to_csv(os.path.join(libra_path, 'libraseq_standardized_Set2.csv'), index=False)


def generate_all_scorepairs():
    libra_path = '../data/processed/libraseq'
    df1 = pd.read_csv(os.path.join(libra_path, 'libraseq_standardized_Set1.csv'), keep_default_na=False)

    with open(os.path.join(libra_path, 'libraseq_allpairs_Set1.csv'), 'w') as f:
        writer = csv.writer(f)

        # column names:
        names = ['id','vh_seq','vl_seq','BG505_CLR_z','CZA97_CLR_z','HA_CLR_z',
                 'B41_CLR_z','BG505_N332T_CLR_z', 'HA_Anhui_CLR_z','HA_NC99_CLR_z',
                 'HA_indo_CLR_z','HA_michigan_CLR_z','KNH1144_CLR_z','ZM106_9_CLR_z','ZM197_CLR_z']
        write_names = ['source_seq1', 'source_seq2', 'chain_type', 'seq1', 'seq2', 'score']
        writer.writerow(write_names)

        cell_lines = df1[df1['id'].str.contains('cell-lines')]
        d45 = df1[df1['id'].str.contains('d45')]

        for source in [cell_lines, d45]:
            for index, row in tqdm(source.iterrows(), total=source.shape[0]):
                for index2, row2 in source.iterrows():
                    s1 = [float(row['BG505_CLR_z']), float(row['CZA97_CLR_z']), float(row['HA_CLR_z'])]
                    s2 = [float(row2['BG505_CLR_z']), float(row2['CZA97_CLR_z']), float(row2['HA_CLR_z'])]
                    sc = np.dot(s1, s2)

                    if row['id'] != row2['id']:
                        for chain in ['H', 'L']:

                            #sc = distance.cosine(s1, s2)
                            seq_type = 'vh_seq' if chain == 'H' else 'vl_seq'
                            content = [row['id'], row2['id'], chain, row[seq_type], row2[seq_type], sc]
                            if 'N/A' not in content:
                                writer.writerow(content)

def sample_from_allpairs(num_sample=100000):
    libra_path = '../data/processed/libraseq'
    df1 = pd.read_csv(os.path.join(libra_path, 'libraseq_allpairs_Set1.csv'), keep_default_na=False)
    df_h = df1[df1['chain_type'] == 'H']
    df_l = df1[df1['chain_type'] == 'L']

    # filter scores within reasonable Z-score range
    df_h = df_h[df_h['score'].between(-10, 10)]
    df_l = df_l[df_l['score'].between(-10, 10)]

    samp_h = df_h.sample(n=num_sample)
    samp_l = df_l.sample(n=num_sample)

    save_dir = '../data/processed/libraseq'

    samp_h.to_csv(os.path.join(save_dir, 'libraseq_pairs_H_Set1_100k.csv'), index=False)
    samp_l.to_csv(os.path.join(save_dir, 'libraseq_pairs_L_Set1_100k.csv'), index=False)


def main_libra(args, orig_embed=False):
    reload_models_to_device(args.device_num)

    # seqs_path = "/data/cb/rsingh/work/antibody/ci_data/raw/libraseq/libraseq_standardized.csv"
    seqs_path = "../data/processed/libraseq/libraseq_standardized_Set1.csv"
    out_folder = '../data/processed/libraseq/cdrembed_maskaug4'
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
                        out_folder = "../data/processed/libraseq/original_embeddings"
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



if __name__ == "__main__":
    # create_standardized_dataset()
    # divide_sets()
    # generate_all_scorepairs()
    sample_from_allpairs()