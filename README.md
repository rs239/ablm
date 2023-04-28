# AbMAP: Antibody Mutagenesis-Augmented Processing
*This repository is a work in progress.*

This repository contains code and pre-trained model checkpoints for AbMAP, a Protein Language Model (PLM) customized for antibodies as featured in Learning the Language of Antibody Hypervariability (Singh, Im et al. 2023) [Link](Link). AbMAP leverages information from foundational PLMs as well as antibody structure and function, offering a multi-functional tool useful for predicting structure, functional properties, and analyzing B-cell repertoires.

### Installation
```bash
pip install abmap  # (recommended) latest release from PyPI 
pip install git+https://github.com/rs239/ablm.git  # the live main branch
```

### Usage:
After installation, AbMAP can be easily imported into your python projects or run from the command line. Please see [examples/demo.ipynb](examples/demo.ipynb/) for common use cases. Instructions for running via CLI are below.

## Command Line Usage
### Augment
*Instructions In Progress*
Given a sequence, generate a foundational PLM embedding augmented with in-silico mutagenesis and CDR isolation.
### Train
Given a dataset of labeled pairs of sequences and their augmented embeddings, train the AbMAP model on downstream prediction tasks.
### Embed
Given fasta sequences and a pre-trained AbMAP model, generate their AbMAP embeddings (fixed or variable).

## Reference
<a id="1">[1]</a>
Madeira, Fábio, et al. "Search and sequence analysis tools services from EMBL-EBI in 2022." Nucleic acids research 50.W1 (2022): W276-W279. (Transeq)
