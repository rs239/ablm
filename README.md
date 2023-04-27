# AbMAP: Antibody Mutagenesis-Augmented Processing
*This repository is a work in progress.*

Protein language model customized for antibodies.

## Commands:
### Augment
Given a sequence, generate a foundational PLM embedding augmented with in-silico mutagenesis and CDR isolation.
### Train
Given a dataset of labeled pairs of sequences and their augmented embeddings, train the AbMAP model on downstream prediction tasks.
### Embed
Given fasta sequences and a pre-trained AbMAP model, generate their AbMAP embeddings (fixed or variable).

## Reference
<a id="1">[1]</a>
Madeira, Fábio, et al. "Search and sequence analysis tools services from EMBL-EBI in 2022." Nucleic acids research 50.W1 (2022): W276-W279. (Transeq)
