FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime
# Ubuntu 16.04 LTS
ARG REPO_DIR=/lab/repos
ARG DATA_DIR=/lab/data
ARG TCRE_REPO_DIR=$REPO_DIR/t-cell-relation-extraction
ARG SNORKEL_REPO_DIR=$REPO_DIR/snorkel
ARG SNORKEL_REPO_URL="https://github.com/HazyResearch/snorkel.git#egg=snorkel"

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN mkdir -p /lab/repos /lab/data
RUN apt-get update && apt-get install -y --no-install-recommends git vim wget

RUN conda create -n nlp python=3.6
RUN activate nlp
RUN pip install --upgrade pip

# Install scispacy models
RUN pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.0/en_core_sci_md-0.2.0.tar.gz
RUN pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.0/en_ner_jnlpba_md-0.2.0.tar.gz

COPY requirements.txt .
RUN pip install -r requirements.txt

# Install bratreader from specific commit for future stability
RUN pip install git+git://github.com/clips/bratreader@4095504278a0ff2742a1c8ce64de2ca83052adf3

# Install snorkel as editable from URL (which can be pinned to a commit -- and is a good idea as
# there aren't real releases yet and breaking changes are common)
RUN pip install --no-deps -e git+$SNORKEL_REPO_URL --src $REPO_DIR

ENV TCRE_REPO_DIR $TCRE_REPO_DIR
ENV SNORKEL_REPO_DIR $SNORKEL_REPO_DIR
ENV SHELL /bin/bash

WORKDIR /lab
CMD jupyter lab --port=8888 --ip=0.0.0.0 --allow-root