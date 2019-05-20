# t-cell-relation-extraction

This repository contains scripts and analysis necessary to extract relationships between T cells, cytokines, and transcription factors from a large PMC corpus using [Data Programming](https://arxiv.org/abs/1605.07723).  Results and methods are discussed in more detail in [Extracting T Cell Function and Differentiation Characteristics from Immunology Literature]().  

### Organization

Notebooks within [pm_subtype_protein_relations](pm_subtype_protein_relations) contain the scripts used to download (as xml) and extract the PMC articles analyzed.  Numbered in order of expected execution, there are also notebooks used to tag named entities and manage the metadata necessary for doing so.  Everything within [pm_subtype_protein_relations/snorkel](pm_subtype_protein_relations/snorkel) is relevant to the relation classifier training process and is intended to run within a separate, Snorkel-compatible environment (snorkel requires old versions of Spacy yet the tagging and initial processing relies on features available only in newer versions).  This directory also contains the analysis notebooks with final results and figures.
