import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.optim as optim
import torch.nn.functional as F
from tcre.modeling import features
from tcre.modeling.utils import mark_entities

# BERT embeddings:
# https://github.com/zalandoresearch/flair/blob/master/flair/embeddings.py#L1416

# Creating custom batches with ignite:
# https://discuss.pytorch.org/t/using-ignite-with-torchtext/24093/8
# https://github.com/pytorch/ignite/blob/master/ignite/engine/__init__.py#L15

# Creating torchtext datasets manually:
# https://stackoverflow.com/questions/52602071/dataframe-as-datasource-in-torchtext

# Loading embeddings into torchtext vocab:
# https://stackoverflow.com/questions/49710537/pytorch-gensim-how-to-load-pre-trained-word-embeddings
# https://discuss.pytorch.org/t/aligning-torchtext-vocab-index-to-loaded-embedding-pre-trained-weights/20878/2

class RNNClassifier(nn.Module):
    """ Override choices in Snorkel abstractions"""

    def __init__(self, cardinality, max_seq_len):
        super().__init__()
        self.cardinality
        self.max_seq_len = max_seq_len

    def _output(self, X):
        n = len(X)
        hidden_state = self.initialize_hidden_state(n)
        max_batch_length = min(max(map(len, X)), self.max_seq_len)

        padded_X = torch.zeros((n, max_batch_length), dtype=torch.long)
        for idx, seq in enumerate(X):
            # TODO: Don't instantiate tensor for each row
            nseq = min(len(seq), max_batch_length)
            padded_X[idx, :nseq] = torch.LongTensor(seq[:nseq])

        output = self.forward(padded_X, hidden_state)
        if self.cardinality == 2:
            return output.view(-1)
        else:
            return output



class LSTM(RNNClassifier):

    def __init__(self, embeddings_shape, cardinality=2, max_seq_len=128, hidden_dim=50, num_layers=1,
                 dropout=0.25, bidirectional=False):
        super().__init__(cardinality, max_seq_len)
        self.embeddings_shape = embeddings_shape
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1


    def build(self, hidden_dim=50, num_layers=1, dropout=0.25, lr=.01, bidirectional=False):
        # Initialize and freeze embedding layer
        self.embedding = nn.Embedding(*self.embeddings_shape, padding_idx=0)
        self.embedding.weight.requires_grad = False
        self.embedding.weight.data = torch.FloatTensor(self.featurizer.model.vectors)
        self.lstm = nn.LSTM(self.embedding_dim, hidden_dim,
                            num_layers=num_layers, bidirectional=bidirectional,
                            dropout=dropout, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim * self.num_directions,
                                      self.cardinality if self.cardinality > 2 else 1)
        self.dropout_layer = nn.Dropout(p=dropout)

        # self.loss = nn.BCEWithLogitsLoss()
        # self.optimizer = optim.Adam([p for p in self.parameters() if p.requires_grad], lr)
        return self

    def forward(self, X, hidden_state):
        seq_lengths = torch.zeros((X.size(0)), dtype=torch.long)
        for i in range(X.size(0)):
            for j in range(X.size(1)):
                if X[i, j] == 0:
                    seq_lengths[i] = j
                    break
                seq_lengths[i] = X.size(1)

        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        X = X[perm_idx, :]
        inv_perm_idx = torch.tensor([i for i, _ in sorted(enumerate(perm_idx), key=lambda idx: idx[1])],
                                    dtype=torch.long)

        encoded_X = self.embedding(X)
        encoded_X = pack_padded_sequence(encoded_X, seq_lengths, batch_first=True)
        _, (ht, _) = self.lstm(encoded_X, hidden_state)
        output = ht[-1] if self.num_directions == 1 else torch.cat((ht[0], ht[1]), dim=1)
        return self.output_layer(self.dropout_layer(output[inv_perm_idx, :]))

    def initialize_hidden_state(self, batch_size):
        return (
            torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim),
            torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim)
        )



