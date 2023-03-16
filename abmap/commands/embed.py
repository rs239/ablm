"""
Given an augmented embedding and a pre-trained AbMAP model,
generate an AbMAP embedding (fixed or variable)
"""

from __future__ import annotations
import argparse
from ..plm_embed import reload_models_to_device
from ..model import AbMAPAttn

from typing import Callable, NamedTuple
import torch
import pickle
import os

class EmbedArguments(NamedTuple):
    cmd: str
    device: int
    pretrained_path: str
    input_path: str
    variable_length: bool
    task: int


def add_args(parser):
    """
    Create parser for command line utility.
    :meta private:
    """
    parser.add_argument(
        "-d", "--device", type=int, default=-1, help="Compute device to use"
    )
    parser.add_argument(
        "--pretrained-path", dest="pretrained_path",
        help="path for the pre-trained AbMAP Model", required=True
    )
    parser.add_argument(
        "--input-path", dest="input_path",
        help="path for the augmented embedding to be inputted to pre-trained AbMAP model", required=True
    )
    parser.add_argument(
        "--output-path", dest="output_path",
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
    Load AbMAP weights
    '''
    dev = torch.device(f'cuda:{device}' if torch.cuda.is_available() else "cpu")
    reload_models_to_device(device, plm_name)

    # load pre-trained model into device
    pretrained = AbMAPAttn(embed_dim=2200, mid_dim2=1024, mid_dim3=512,
                           proj_dim=252, num_enc_layers=1, num_heads=16).to(dev)
    checkpoint = torch.load(pretrained_path, map_location=dev)
    pretrained.load_state_dict(checkpoint['model_state_dict'])
    pretrained.eval()
    print("Loaded the Pre-trained Model!")
    return pretrained


def abmap_embed_batch(device, pretrained_path, input_path, output_path, variable_length, plm_name, task):
    '''
    Like abmap_embed, but works for a directory of inputs
    '''
    pretrained = load_abmap(device, pretrained_path, plm_name)
    dev = torch.device(f'cuda:{device}' if torch.cuda.is_available() else "cpu")

    # Get list of files or casts single file as a list
    if os.path.isdir(input_path): input_iter = os.listdir(input_path)
    elif os.path.isfile(input_path): input_iter = list(input_path)
    
    outputs = []
    for input in input_iter:
        with open(input, 'rb') as p:
            input_embed = pickle.load(p).to(dev)
        input_embed = torch.unsqueeze(input_embed, 0)
        try:
            assert len(input_embed.shape) == 3
        except:
            raise ValueError("input embedding should be of shape n'(CDR length) x d")

        # generate the abmap embedding
        with torch.no_grad():
            if variable_length:
                out_feature, _ = pretrained.embed(input_embed, task=task, embed_type='variable')
            else:
                out_feature, _ = pretrained.embed(input_embed, task=task, embed_type='fixed')
        out_feature = torch.squeeze(out_feature, 0)
        outputs.append(out_feature)
        
        return outputs

        

def abmap_embed(device, pretrained_path, input_path, output_path, variable_length, plm_name, task):
    """
    # TODO - Allow for a batch of inputs
    """
    pretrained = load_abmap(device, pretrained_path, plm_name)
    dev = torch.device(f'cuda:{device}' if torch.cuda.is_available() else "cpu")

    # load the input embedding into device
    # input_path = '/net/scratch3/scratch3-3/chihoim/ablm/data/processed/sabdab/cdrembed_maskaug4/beplerberger/sabdab_7wvm_cat2_H_k100.p' # comment out later
    with open(input_path, 'rb') as p:
        input_embed = pickle.load(p).to(dev)
    input_embed = torch.unsqueeze(input_embed, 0)
    try:
        assert len(input_embed.shape) == 3
    except:
        raise ValueError("input embedding should be of shape n'(CDR length) x d")

    # generate the abmap embedding
    with torch.no_grad():
        if variable_length:
            out_feature, _ = pretrained.embed(input_embed, task=task, embed_type='variable')
        else:
            out_feature, _ = pretrained.embed(input_embed, task=task, embed_type='fixed')
    out_feature = torch.squeeze(out_feature, 0)
    print("out feature shape:", out_feature.shape)

    # save the abmap embedding
    with open(output_path, 'wb') as f:
        pickle.dump(out_feature.cpu(), f)
    
    return out_feature


def main(args):
    """
    Run embedding from arguments.
    :meta private:
    """
    device = args.device
    pretrained_path = args.pretrained_path
    input_path = args.input_path
    output_path = args.output_path
    variable_length = args.variable_length
    plm_name = args.plm_name
    task = args.task
    abmap_embed(device, pretrained_path, input_path, output_path,
                 variable_length, plm_name, task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())
