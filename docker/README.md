# Docker Instructions

Download and install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) for GPU 
utilization (or Docker CE otherwise)

## Build Instructions

```bash
nvidia-docker build -t t-cell-relation-extraction -f Dockerfile .
```

## Run Instructions

```bash
export TCRE_DATA_DIR=/data/disk2/nlp/20190311-pubmed-tcell-relation
export TCRE_REPO_DIR=/home/eczech/repos/t-cell-relation-extraction
export SNORKEL_REPO_DIR=/home/eczech/repos/misc/snorkel
nvidia-docker run --rm -ti -p 8888:8888 -p 6006:6006 \
-v $SNORKEL_REPO_DIR:/lab/repos/snorkel \
-v $TCRE_REPO_DIR:/lab/repos/t-cell-relation-extraction \
-v $TCRE_DATA_DIR:/lab/data \
t-cell-relation-extraction
> 
```
