FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime
# Ubuntu 16.04 LTS
ARG REPO_DIR=/lab/repos
ARG DATA_DIR=/lab/data
ARG TCRE_REPO_DIR=$REPO_DIR/t-cell-relation-extraction
ARG SNORKEL_REPO_DIR=$REPO_DIR/snorkel
ARG SNORKEL_REPO_URL="https://github.com/HazyResearch/snorkel.git@c17c38ac8ddf569484cf9c0bde4702b2f2622033#egg=snorkel"

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN mkdir -p /lab/repos /lab/data
RUN apt-get update && apt-get install -y --no-install-recommends git vim wget

# Create environment and set default environment on login
# (note that the "conda [init|activate]" commands do not affect the
# rest of the docker build but instead only apply to user logins)
RUN conda create -n nlp python=3.6
RUN conda init bash
RUN echo "conda activate nlp" >> ~/.bashrc
# Set python for remainder of build (this will be overwritten by entries to bashrc above on login)
ENV PATH /opt/conda/envs/nlp/bin:$PATH
RUN pip install --upgrade pip

# Install scispacy models
RUN pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.0/en_core_sci_md-0.2.0.tar.gz
RUN pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.0/en_ner_jnlpba_md-0.2.0.tar.gz

# Install all other dependencies, making an exception for Cython which must already be
# present for the successful installation of some other packages
COPY requirements.txt .
RUN pip install Cython==0.29.10
RUN pip install -r requirements.txt

# Install bratreader from specific commit for future stability
RUN pip install git+git://github.com/clips/bratreader@4095504278a0ff2742a1c8ce64de2ca83052adf3

# Install snorkel as editable from URL (which can be pinned to a commit -- and is a good idea as
# there aren't real releases yet and breaking changes are common)
RUN pip install --no-deps -e git+$SNORKEL_REPO_URL --src $REPO_DIR
RUN pip install --no-deps git+https://github.com/HazyResearch/numbskull@master
RUN pip install --no-deps git+https://github.com/HazyResearch/treedlib@master

# Install skopt for hyperparameter search
RUN pip install -e git://github.com/scikit-optimize/scikit-optimize.git@v0.6#egg=scikit-optimize

# Install helpful jupyter extensions
RUN pip install jupyter_contrib_nbextensions && \
    jupyter contrib nbextension install --user && \
    jupyter nbextension enable python-markdown/main

# Add TCRE project source modules to PYTHONPATH
RUN echo "$TCRE_REPO_DIR/src" >> /opt/conda/envs/nlp/lib/python3.6/site-packages/local.pth

ENV TCRE_REPO_DIR $TCRE_REPO_DIR
ENV SNORKEL_REPO_DIR $SNORKEL_REPO_DIR
ENV DATA_DIR $DATA_DIR
ENV REPO_DIR $TCRE_REPO_DIR
ENV REPO_DATA_DIR $TCRE_REPO_DIR/data
ENV SNORKELDB sqlite:///$TCRE_REPO_DIR/data/snorkel/snorkel.db
ENV SHELL /bin/bash

WORKDIR /lab
CMD jupyter lab --port=8888 --ip=0.0.0.0 --allow-root