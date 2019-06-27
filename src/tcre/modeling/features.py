import os
import os.path as osp
import numpy as np
from tcre.env import *


class W2VFeaturizer(object):

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.words = set(model.vocab)

    def indices(self, sentence):
        tokens = [str(w) for w in self.tokenizer(sentence)]
        tokens = [t if t in self.words else 'UNK' for t in tokens]
        indices = [self.model.ix(t) for t in tokens]
        return np.array(indices), np.array(tokens)

    def embeddings(self, sentence):
        indices, tokens = self.indices(sentence)
        return np.stack([model.vectors[i] for i in indices]), tokens


def get_spacy_w2v_featurizer(nlp_model='en_core_sci_md'):
    import spacy
    import word2vec
    nlp = spacy.load(nlp_model)
    model = word2vec.load(W2V_MODEL_01)
    return W2VFeaturizer(model, nlp)
