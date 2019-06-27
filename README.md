# t-cell-relation-extraction

This repository contains scripts and analysis necessary to extract relationships between T cells, cytokines, and transcription factors from a large PMC corpus using [Data Programming](https://arxiv.org/abs/1605.07723).  Results and methods are discussed in more detail in [Extracting T Cell Function and Differentiation Characteristics from Immunology Literature]().  

### Organization

Notebooks within [pm_subtype_protein_relations](pm_subtype_protein_relations) contain the scripts used to download (as xml) and extract the PMC articles analyzed.  Numbered in order of expected execution, there are also notebooks used to tag named entities and manage the metadata necessary for doing so.  Everything within [pm_subtype_protein_relations/snorkel](pm_subtype_protein_relations/snorkel) is relevant to the relation classifier training process and is intended to run within a separate, [Snorkel](https://hazyresearch.github.io/snorkel/)-compatible environment (snorkel requires old versions of Spacy yet the tagging and initial processing relies on features available only in newer versions).  This directory also contains the analysis notebooks with final results and figures.

### Data Flow

- All raw articles are imported to $DATA_DIR/articles/import/<date>/data.csv
- A "corpus" is created from any of the raw import files
    - Current correspondence:
        - 20190314/data.csv -> corpus_00
        - 20190501/data.csv -> corpus_01
    - These corpus folders should contain all relevant exports such as .txt, .ann files, and tag/relation csvs
- Candidates should then be imported into snorkel db where:
    - Document IDs should be unique regardless of corpus (any corpus can be loaded into same Document table)
    - Candidates are loaded in one of two ways:
        - For training, train/dev/test splits are loaded
        - For inference, all canidates are inserted for an "inference" split (3)
- Inference/analysis can then be applied using any available candidate split
