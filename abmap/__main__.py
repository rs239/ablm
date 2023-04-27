"""
AbMAP: Augmenting PLM Embeddings of Antibodies
with in-silico mutagenesis and CDR isolation
"""
import argparse
import os
import sys
from typing import Union

from abmap.commands.augment import AugmentArguments
from abmap.commands.train import TrainArguments
from abmap.commands.embed import EmbedArguments

AbMAPArguments = Union[AugmentArguments, TrainArguments, EmbedArguments]


class CitationAction(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super(CitationAction, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        from . import __citation__

        print(__citation__)
        setattr(namespace, self.dest, values)
        sys.exit(0)


def main():
    from . import __version__

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "-v", "--version", action="version", version="AbMAP " + __version__
    )
    parser.add_argument(
        "-c",
        "--citation",
        action=CitationAction,
        nargs=0,
        help="show program's citation and exit",
    )

    subparsers = parser.add_subparsers(title="AbMAP Commands", dest="cmd")
    subparsers.required = True

    from .commands import augment, train , embed

    modules = {
        "augment": augment,
        "train": train,
        "embed": embed,
    }

    for name, module in modules.items():
        sp = subparsers.add_parser(name, description=module.__doc__)
        module.add_args(sp)
        sp.set_defaults(func=module.main)

    args: AbMAPArguments = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
