"""
Given a sequence, generate a PLM embedding augmented with
in-silico mutagenesis and CDR isolation.
"""

from __future__ import annotations
import argparse
from abmap.abmap_augment import augment_from_fasta

from typing import Callable, NamedTuple


class AugmentArguments(NamedTuple):
    cmd: str
    device: int
    outdir: str
    seqs: str
    chain_type: str
    plm: str
    num_mutations: int
    separators: bool
    cdr_masks: bool
    func: Callable[[AugmentArguments], None]


def add_args(parser):
    """
    Create parser for command line utility.
    :meta private:
    """
    parser.add_argument(
        "--seqs", help="Sequences to be embedded", required=True
    )
    parser.add_argument(
        "-o", "--outdir", help="h5 file to write results", required=True
    )
    parser.add_argument(
        "-d", "--device", type=int, default=-1, help="Compute device to use"
    )
    parser.add_argument(
        "--chain-type", dest="chain_type",
        help="Chain type of the fasta sequences (H or L)", required=True
    )
    parser.add_argument(
        "--plm", help="which foundational PLM to use for embedding", required=True
    )
    parser.add_argument(
        "--num-mutations", type=int, default=100, dest='num_mutations',
        help="Number of mutations for in-silico mutagenesis. 0 means no mutations."
    )
    parser.add_argument(
        "--separators", action="store_true", dest='separators',
        help="Place separators (0 pad) between the CDRs"
    )
    parser.add_argument(
        "--cdr-masks", action="store_true", dest='cdr_masks',
        help="concatenate 4 extra dimensions for CDR position masks."
    )
    return parser


def main(args):
    """
    Run embedding from arguments.
    :meta private:
    """
    inPath = args.seqs
    outPath = args.outdir
    device = args.device
    num_mutations = args.num_mutations
    chain_type = args.chain_type
    embed_type = args.plm
    seperators = args.separators
    cdr_masks = args.cdr_masks
    augment_from_fasta(inPath, outPath, chain_type, embed_type,
                       num_mutations, seperators, cdr_masks, 
                       device, verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())