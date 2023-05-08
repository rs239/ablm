"""
Given fasta sequences and a pre-trained AbMAP model,
generate their AbMAP embeddings (fixed or variable).
"""

from __future__ import annotations
import argparse

from abmap.plm_embed import reload_models_to_device
from abmap.abmap_augment import ProteinEmbedding
from abmap.model import AbMAPAttn
from abmap.utils import parse
import sys
sys.path.append('../') # for access to model.py when calling torch.load()

from typing import Callable, NamedTuple
import torch
import pickle
from tqdm import tqdm
import os

class EmbedArguments(NamedTuple):
    cmd: str
    device: int
    chain_type: str
    pretrained_path: str
    input_path: str
    variable_length: bool
    task: int
    func: Callable[[EmbedArguments], None]


def add_args(parser):
    """
    Create parser for command line utility.
    :meta private:
    """
    parser.add_argument(
        "-d", "--device", type=int, default=-1, help="Compute device to use"
    )
    parser.add_argument(
        "--chain-type", dest="chain_type", choices=['H', 'L'],
        help="Chain type of the fasta sequences (H or L)", required=True
    )
    parser.add_argument(
        "--pretrained-path", dest="pretrained_path",
        help="path for the pre-trained AbMAP Model", required=True
    )
    parser.add_argument(
        "--input-file", dest="input_file",
        help="path for fasta file containing sequences to be embeded using the pre-trained AbMAP model", required=True
    )
    parser.add_argument(
        "--output-dir", dest="output_dir",
        help="path to save the output AbMAP embedding", required=True
    )
    parser.add_argument(
        "--plm-name", dest="plm_name",
        help="Name of the foundational PLM used when generating the augmented input embedding", default='beplerberger'
    )
    parser.add_argument(
        "--variable-length", action="store_true", dest='variable_length',
        help="Output variable length AbMAP Embeddings. (Default: Fixed Length)"
    )
    parser.add_argument(
        "--task", type=int, default='structure', choices=['structure', 'function'],
        help="Which task to be specified for generating the embedding (0: structure or 1: function). Default: 0"
    )
    return parser


def load_abmap(pretrained_path, plm_name, device=0):
    '''
    Load AbMAP weights and foundational model
    '''
    dev = torch.device(f'cuda:{device}' if torch.cuda.is_available() else "cpu")
    reload_models_to_device(device, plm_name)

    # load pre-trained model into device
    if plm_name == 'beplerberger':
        pretrained = AbMAPAttn(embed_dim=2200, mid_dim2=1024, mid_dim3=512, 
                                     proj_dim=252, num_enc_layers=1, num_heads=16).to(dev)
    if plm_name == 'protbert':
        pretrained = AbMAPAttn(embed_dim=1024, mid_dim2=512, mid_dim3=256, 
                                     proj_dim=252, num_enc_layers=1, num_heads=16).to(dev)
    if plm_name == 'esm1b':
        pretrained = AbMAPAttn(embed_dim=1280, mid_dim2=512, mid_dim3=256, 
                                     proj_dim=252, num_enc_layers=1, num_heads=16).to(dev)
    if plm_name == 'tape':
        pretrained = AbMAPAttn(embed_dim=768, mid_dim2=256, mid_dim3=128, 
                                     proj_dim=60, num_enc_layers=1, num_heads=8).to(dev)
    checkpoint = torch.load(pretrained_path, map_location=dev)
    if 'model_state_dict' in checkpoint:
        pretrained.load_state_dict(checkpoint['model_state_dict'])
    else:
        pretrained.load_state_dict(checkpoint)
    pretrained.eval()
    print("Loaded the Pre-trained Model!")
    return pretrained


def abmap_embed_batch(device, pretrained_path, input_path, variable_length, plm_name, task):
    '''
    Like abmap_embed, but works for a directory of inputs
    '''
    pretrained = load_abmap(pretrained_path, plm_name, device)
    dev = torch.device(f'cuda:{device}' if torch.cuda.is_available() else "cpu")

    # Get list of files or casts single file as a list
    if os.path.isdir(input_path): input_iter = os.listdir(input_path)
    elif os.path.isfile(input_path): input_iter = list(input_path)
    
    outputs = []
    for input in tqdm(input_iter):
        input_ = os.path.join(input_path, input)
        with open(input_, 'rb') as p:
            input_embed = pickle.load(p).to(dev)
        input_embed = torch.unsqueeze(input_embed, 0)
        try:
            assert len(input_embed.shape) == 3
        except:
            raise ValueError("input embedding should be of shape n'(CDR length) x d")

        # generate the abmap embedding
        with torch.no_grad():
            if variable_length:
                out_feature = pretrained.embed(input_embed, task=task, embed_type='variable')
            else:
                out_feature = pretrained.embed(input_embed, task=task, embed_type='fixed')
        out_feature = torch.squeeze(out_feature, 0)
        outputs.append(out_feature)
        
    return outputs

        
def abmap_embed(device, pretrained_path, chain_type, input_file, output_dir, variable_length, plm_name, task):

    """
    Given fasta sequences and a pre-trained AbMAP model,
    generate their AbMAP embeddings (fixed or variable).
    
    ***
    For Pre-Trained AbMAP models, the augmented embeddings have the following parameters:
        Foundational PLM: Bepler & Berger
        CDR-masks (4 extra dims)
        NO separators
        # mutations = 100
    """
    embed_type = 'beplerberger'
    task_ = 'structure' if task == 0 else 'function'
    output_type = 'variable' if variable_length else 'fixed'
    
    pretrained = load_abmap(pretrained_path, plm_name, device)
    dev = torch.device(f'cuda:{device}' if torch.cuda.is_available() else "cpu")

    # get the names and sequences of fasta files
    names, seqs = parse(input_file)
    
    # check if outputPath is an existing directory:
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # generate the embeddings for fasta sequences:    
    for (name, seq) in tqdm(zip(names, seqs), total=len(names)):
        fname = os.path.join(output_dir, f'{name}_AbMAP.p')
        if not os.path.isfile(fname):

            prot = ProteinEmbedding(seq, chain_type, embed_device=f'cuda:{device}')
            z = prot.create_cdr_specific_embedding(embed_type)
            z = torch.unsqueeze(z, dim=0).to(dev)

            with torch.no_grad():
                output = pretrained.embed(z, task=task_, embed_type=output_type)
            output = torch.squeeze(output, dim=0)
            
            # save the abmap embedding
            with open(fname, 'wb') as f:
                pickle.dump(output.cpu(), f)
                
    return
                


def main(args):
    """
    Run embedding from arguments.
    :meta private:
    """
    device = args.device
    chain_type = args.chain_type
    pretrained_path = args.pretrained_path
    input_file = args.input_file
    output_dir = args.output_dir
    variable_length = args.variable_length
    plm_name = args.plm_name
    task = args.task
    abmap_embed(device, pretrained_path, chain_type, input_file, output_dir,
                 variable_length, plm_name, task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())
