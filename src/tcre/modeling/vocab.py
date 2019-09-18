import torch
import numpy as np
from torchtext.vocab import Vocab
from collections import defaultdict, Counter
import pandas.core.common as com


class W2VVocab(Vocab):

    def __init__(self, model, specials=None, random_state=None):
        """Build pretrained w2v vocab

        Args:
            model: Gensim model (e.g. KeyedVectors.load_word2vec_format(W2V_MODEL_01, binary=True, limit=50000))
            specials: Extra tokens to add with randomized vectors ("<pad>" is always added first)
            random_state: Random state used to initialized random vectors
        """
        super().__init__(Counter())

        # Remove any specials already present in the vocab and prepend pad token
        assert '<pad>' not in model.vocab
        specials = list(sorted(specials)) if specials else []
        specials = [v for v in specials if v not in model.vocab]
        specials = ['<pad>'] + specials

        self.itos = specials + list(model.vocab.keys())

        # Use zero vector for unk as well as pad
        def get_unk_index():
            return 0
        self.stoi = defaultdict(get_unk_index)
        self.stoi.update({w: i for i, w in enumerate(self.itos)})

        if len(self.itos) != len(self.stoi):
            raise ValueError(
                f'Vocab has repeated words (possibly due to unicode normalization) '
                f'(len(itos) = {len(itos)}, len(stoi) = {len(stoi)}')

        # Add single zero vector for pad token
        tensors = [torch.FloatTensor(np.zeros((1, model.vectors.shape[1])))]

        # Add random vectors for other specials
        if len(specials) > 1:
            rs = com.random_state(random_state)
            tensors.append(torch.FloatTensor(rs.normal(size=(len(specials)-1, model.vectors.shape[1]))))

        # Add remaining vectors
        tensors.append(torch.FloatTensor(model.vectors))

        # Concatenate vectors and ensure there are as many as there are words
        self.vectors = torch.cat(tensors, dim=0)
        if len(self.itos) != len(self.vectors):
            raise AssertionError(
                f'W2V vocab has unequal vector and vocab size '
                f'(len(vocab) = {len(self.itos)}, len(vectors) = {len(self.vectors)})'
            )
