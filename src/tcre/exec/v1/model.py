import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from tcre.modeling.features import MAX_POS_DIST
DIST_PAD_VAL = MAX_POS_DIST + 1
CELL_TYPES = {'LSTM': nn.LSTM, 'GRU': nn.GRU}

def pos_indices(pos, max_dist, pad_val):
    """Convert position features provided as positive or negative integers to embedding indices

    Example:
        pos = torch.IntTensor([-100, -4, -3, -2, -1, 0, 1, 2, 3, 4, 100, 99, 99])
        pos_indices(pos, 2, 99) -> [1, 1, 1, 2, 3, 4, 5, 6, 7, 7, 7, 0, 0]
    """
    if torch.any(pos > pad_val):
        raise ValueError(
            f'Position array has value > padding value (max pos = {pos.max().item()}, pad val = {pad_val})')
    offset = max_dist + 2
    pos = pos + offset
    pos = torch.clamp(pos, 1, pad_val + offset)
    pos[pos == (pad_val + offset)] = 0
    pos = torch.clamp(pos, 0, 2 * max_dist + 3)
    return pos


class RNN(nn.Module):

    def __init__(self, fields, cardinality=2, hidden_dim=50, wrd_embed_dim=None, pos_embed_dim=10, train_wrd_embed=None,
                 num_layers=1, cell_type='LSTM',
                 dropout=0, bidirectional=False, max_dist=50, device=None,
                 names=['text', 'label', 'e0_dist', 'e1_dist']):
        super().__init__()
        self.fields = fields
        self.cardinality = cardinality
        self.hidden_dim = hidden_dim
        self.pos_embed_dim = pos_embed_dim
        self.wrd_embed_dim = wrd_embed_dim
        self.train_wrd_embed = train_wrd_embed
        if self.train_wrd_embed is None:
            self.train_wrd_embed = wrd_embed_dim is not None
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        self.cell_type = cell_type
        if self.cell_type not in CELL_TYPES:
            raise ValueError(f'Cell type must be one of {list(keys(CELL_TYPES))}, not {self.cell_type}')
        self.cell_type = CELL_TYPES[self.cell_type]
        self.max_dist = max_dist
        self.device = device
        self.names = names

        self._init_embedding()
        self._init_cell()
        self.output = nn.Linear(self.hidden_dim * self.num_directions, self.cardinality if self.cardinality > 2 else 1)
        self.dropout = nn.Dropout(p=dropout)

    def _init_embedding(self):
        if self.wrd_embed_dim:
            self.wrd_embed_shape = (len(self.fields[self.names[0]].vocab), self.wrd_embed_dim)
            self.wrd_embed = nn.Embedding(*self.wrd_embed_shape, padding_idx=0)
        else:
            vectors = self.fields[self.names[0]].vocab.vectors
            self.wrd_embed_shape = tuple(vectors.shape)
            self.wrd_embed = nn.Embedding.from_pretrained(vectors, padding_idx=0)
        self.wrd_embed.weight.requires_grad = self.train_wrd_embed

        # Index values are 0=pad, 1=< -max_dist, 2=-max_dist, 3=-max_dist+1, ..., 2*(max_dist+2) > max_dist
        self.pos_embed_shape = (2 * (self.max_dist + 2), self.pos_embed_dim)
        # Define without looping/lists as placement on GPU will fail otherwise
        self.pos_embed_e0 = nn.Embedding(*self.pos_embed_shape, padding_idx=0)
        self.pos_embed_e1 = nn.Embedding(*self.pos_embed_shape, padding_idx=0)

    def _init_cell(self):
        self.cell = self.cell_type(
            self.wrd_embed_shape[1] + 2 * self.pos_embed_dim, self.hidden_dim,
            num_layers=self.num_layers, bidirectional=self.bidirectional,
            dropout=self.dropout, batch_first=True
        )

    def prepare(self, batch, **kwargs):
        """Extract seq token indices, seq lengths, and training labels"""
        text, label, e0_dist, e1_dist = [getattr(batch, n) if n in batch.fields else None for n in self.names]
        X, L = text[0].t(), text[1]
        # Convert relative positions (as pos/neg integers or pad) to embedding indices
        D0, D1 = [pos_indices(v.t(), self.max_dist, DIST_PAD_VAL) for v in [e0_dist, e1_dist]]
        Y = None if label is None else (label - 1).type(torch.FloatTensor).to(self.device)
        features = (X, L, D0, D1)
        return tuple([f.to(self.device) for f in features]), Y

    def transform(self, Y):
        return torch.sigmoid(Y) if self.cardinality == 2 else torch.softmax(Y)

    def classify(self, Y):
        return torch.round(Y) if self.cardinality == 2 else torch.argmax(Y, dim=0)

    def forward(self, features):
        X, L, D0, D1 = features
        H = self.initial_hidden_state(len(X))
        X = self.wrd_embed(X)
        D = torch.cat([self.pos_embed_e0(D0), self.pos_embed_e1(D1)], dim=-1)
        X = torch.cat([X, D], dim=-1)
        L = L.view(-1).tolist()
        X = nn.utils.rnn.pack_padded_sequence(X, L, batch_first=True)
        ht = self.cell(X, H)[1]
        ht = ht[0] if isinstance(ht, tuple) else ht
        Y = ht[-1] if self.num_directions == 1 else torch.cat((ht[0], ht[1]), dim=1)
        Y = self.output(self.dropout(Y))
        return Y.view(-1) if self.cardinality == 2 else Y

    def initial_hidden_state(self, batch_size):
        def get_h0():
            return torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim).to(self.device)

        if self.cell_type == nn.GRU:
            return get_h0()
        elif self.cell_type == nn.LSTM:
            return tuple([get_h0(), get_h0()])
        else:
            raise ValueError(f'RNN type {self.cell_type.__name__} not supported')
