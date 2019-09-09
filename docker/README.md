# Docker Instructions

Download and install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) for GPU 
utilization (or Docker CE otherwise)

## Build Instructions

```bash
cd docker
nvidia-docker build -t t-cell-relation-extraction -f Dockerfile .

# To build using a forked snorkel repo:
nvidia-docker build -t t-cell-relation-extraction \
    --build-arg SNORKEL_REPO_URL=https://github.com/eric-czech/snorkel.git#egg=snorkel \
    -f Dockerfile .
```

## Run Instructions

```bash
# Remote (w/ GPU)
export TCRE_DATA_DIR=/data/disk2/nlp/20190311-pubmed-tcell-relation
export TCRE_REPO_DIR=/home/eczech/repos/t-cell-relation-extraction
nvidia-docker run --rm -ti -p 8888:8888 -p 6006:6006 \
-v $TCRE_REPO_DIR:/lab/repos/t-cell-relation-extraction \
-v $TCRE_DATA_DIR:/lab/data \
t-cell-relation-extraction

# Local (Mac w/ no GPU)
export TCRE_DATA_DIR=/Users/eczech/data/research/hammer/nlp/20190311-pubmed-tcell-relation
export TCRE_REPO_DIR=/Users/eczech/repos/hammer/t-cell-relation-extraction
docker run --rm -ti -p 8888:8888 -p 6006:6006 \
-v $TCRE_REPO_DIR:/lab/repos/t-cell-relation-extraction \
-v $TCRE_DATA_DIR:/lab/data \
t-cell-relation-extraction
```
