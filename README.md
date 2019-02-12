# t-cell-relation-extraction

This repository contains simple methods for relating T cell subtypes to related intracellular, transcriptional, and surface proteins.
Currently, this only involves recognition of T cell subtype (e.g. NKT, MAIT, TSCM) and protein (e.g. RUNX3, CD25, EOMES) entities followed by
analysis of their co-occurrence.  Ideally, this would also include a categorization of the relationship between the two but for now, a basic
protein term relevance scoring can still capture proteins most specific to a particular cell subtype as shown below:

![Outline](docs/images/outline_diagram.png)

Cell type extractions are done using the experimental [SciSpacy](https://allenai.github.io/scispacy/) project as a search for particular parts of
speech that are immediately downstream to "cell" or "lymphocyte" lemmas and match heuristic rules common to T cell type acronyms.  
rotein tagging comes from [D3NER](https://www.ncbi.nlm.nih.gov/pubmed/29718118).  
