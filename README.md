# T Cell Relation Extraction (TCRE)

This repository contains the scripts and analysis necessary to extract relationships between T cells, cytokines, and transcription factors from a large PMC corpus using [Data Programming](https://arxiv.org/abs/1605.07723).  In short, the purpose of this reasearch is to identify relations like this often referenced as a small part of larger cell signaling networks:

<img src="docs/images/relation_examples.png" width="100%"/>

## Information Flow

The relations are identified by a weakly supervised classifier trained using distant supervision from [immuneXpresso](http://immuneexpresso.org/immport-immunexpresso/public/immunexpresso/search), heuristics, text patterns, and standard supervised classifiers trained on a small manually labeled data split.  [Snorkel](http://snorkel.stanford.edu/) is used to develop a generative model on top of the classifications from these different sources and the weak labels from that model are then fed into a noise-aware classifier (trained on ~50k examples per relation).  A high-level overview of this information flow is shown below:

<img src="docs/images/training_outline.png" width="100%"/>

## Resources

This work is currently in progress, but this [Summary Notebook](https://nbviewer.jupyter.org/github/hammerlab/t-cell-relation-extraction/blob/master/results/summary.render.ipynb) contains a rolling account of many details such as how documents were selected, what labeling functions were developed, tokenization challenges, controlled vocabularies, preliminary classification performance results, etc.


An early draft of a pre-print is also available at [Extracting T Cell Function and Differentiation Characteristics from the Biomedical Literature](https://www.biorxiv.org/content/10.1101/643767v1).  
