# AbMAP: Antibody Mutagenesis-Augmented Processing
![image](https://user-images.githubusercontent.com/6614489/235450484-2ad78557-0deb-43fb-ba8d-6fb570c3a052.png)

*This repository is a work in progress.*

This repository contains code and pre-trained model checkpoints for AbMAP, a Protein Language Model (PLM) customized for antibodies as featured in **Learning the Language of Antibody Hypervariability** ([_Singh, Im et al. 2023_](https://www.biorxiv.org/content/10.1101/2023.04.26.538476)). AbMAP leverages information from foundational PLMs as well as antibody structure and function, offering a multi-functional tool useful for predicting structure, functional properties, and analyzing B-cell repertoires.

### Installation
AbMAP relies on ANARCI to assign IMGT labels to antibody sequences. Please see the [ANARCI](https://github.com/oxpig/ANARCI/blob/master/INSTALL) repo or run the following in a new conda environment: 
```bash
conda install -c biocore hmmer # Can also install using `brew/port/apt/yum install hmmer`
git clone https://github.com/oxpig/ANARCI.git
cd ANARCI
python setup.py install
```

Then install abmap using:
```bash
pip install abmap  # (recommended) latest release from PyPI 
pip install git+https://github.com/rs239/ablm.git  # the live main branch
```

### Usage:
After installation, AbMAP can be easily imported into your python projects or run from the command line. Please see [examples/demo.ipynb](examples/demo.ipynb/) for common use cases. Instructions for running via CLI are below.

## Command Line Usage 
*Instructions In Progress*

### Augment
Given a sequence, generate a foundational PLM embedding augmented with in-silico mutagenesis and CDR isolation.
### Train
Given a dataset of labeled pairs of sequences and their augmented embeddings, train the AbMAP model on downstream prediction tasks.
### Embed
Given fasta sequences and a pre-trained AbMAP model, generate their AbMAP embeddings (fixed or variable).

Please provide feedback on the issues page or by opening a pull request. If AbMAP is useful in your work, please consider citing our [bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2023.04.26.538476). 

