## Data Flow

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