import torch
import numpy as np
from torchtext.vocab import Vocab
from collections import defaultdict, Counter


class W2VVocab(Vocab):

    def __init__(self, model):
        """Build pretrained w2v vocab

        Args:
            model: Gensim model (e.g. KeyedVectors.load_word2vec_format(W2V_MODEL_01, binary=True, limit=50000))
        """
        super().__init__(Counter())
        specials = ['<pad>']
        self.itos = specials + list(model.vocab.keys())

        # Use zero vector for unk as well as pad
        def get_unk_index():
            return 0
        self.stoi = defaultdict(get_unk_index)
        self.stoi.update({w: i for i, w in enumerate(self.itos)})

        if len(self.itos) != len(self.stoi):
            raise ValueError(
                f'W2V vocab has repeated words (probably due to unicode normalization) '
                f'(len(itos) = {len(itos)}, len(stoi) = {len(stoi)}')
        self.vectors = torch.cat([
            torch.FloatTensor(np.zeros((len(specials), model.vectors.shape[1]))),
            torch.FloatTensor(model.vectors)
        ], dim=0)
