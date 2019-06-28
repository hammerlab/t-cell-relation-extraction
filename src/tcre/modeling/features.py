import os
import os.path as osp
import numpy as np
import pandas as pd
from tcre.modeling.utils import mark_entities

# class W2VFeaturizer(object):
#
#     def __init__(self, model, tokenizer):
#         self.model = model
#         self.tokenizer = tokenizer
#         self.words = set(model.vocab)
#
#     def indices(self, sentence):
#         tokens = [str(w) for w in self.tokenizer(sentence)]
#         tokens = [t if t in self.words else 'UNK' for t in tokens]
#         indices = [self.model.ix(t) for t in tokens]
#         return np.array(indices), np.array(tokens)
#
#     def embeddings(self, sentence):
#         indices, tokens = self.indices(sentence)
#         return np.stack([model.vectors[i] for i in indices]), tokens
#
#
# def get_spacy_w2v_featurizer(nlp_model='en_core_sci_md'):
#     import spacy
#     import word2vec
#     nlp = spacy.load(nlp_model)
#     model = word2vec.load(W2V_MODEL_01)
#     return W2VFeaturizer(model, nlp)


def candidates_to_records(cands):
    return [
        dict(
            id=cand.id,
            text=str(cand.get_parent().text),
            words=list(cand.get_parent().words),
            label=cand.gold_labels[0].value if cand.gold_labels else 0,
            entities=[
                dict(word_range=cand.get_contexts()[i].get_word_range(), text=str(cand.get_contexts()[i].get_span()))
                for i in range(len(cand.get_contexts()))
            ]
        )
        for cand in cands
    ]


def get_dist_bin(i, rng):
    if i < rng[0]:
        v = i - rng[0]
    elif i > rng[1]:
        v = i - rng[1]
    else:
        v = 0
    return v


def get_record_features(records, markers=[], subtokenizer=None):

    if subtokenizer is None:
        subtokenizer = lambda t: [t]

    def get_features(rec):
        # First ensure that none of the markers appear in the text within which they should be unique
        text = rec['text']
        for m in markers:
            if m in text:
                raise AssertionError(f'Found marker "{m}" in text "{text}"')

        # Add markers around entity spans
        n = len(rec['entities'])
        words = list(rec['words'])
        positions = [rec['entities'][i]['word_range'] for i in range(n)]

        tokens = mark_entities(words, positions, style='insert', markers=markers)
        # Re-space word index list in the same manner as the words themselves
        indices = mark_entities(list(range(len(words))), positions, style='insert', markers=[None] * len(markers))
        # Use re-spaced index list to map original entity word positions to new positions
        positions = [(indices.index(positions[i][0]), indices.index(positions[i][1])) for i in range(n)]

        subtokens = [(i0, t0, i1, t1) for i0, t0 in enumerate(tokens) for i1, t1 in enumerate(subtokenizer(t0))]
        tokens = []
        word_indices = []
        token_indices = []
        entity_distances = [[] for _ in range(n)]
        for i, (i0, t0, i1, t1) in enumerate(subtokens):
            word_indices.append(indices[i0])
            token_indices.append(i0)
            tokens.append(t1)
            for j, rng in enumerate(positions):
                entity_distances[j].append(get_dist_bin(i0, rng))

        features = dict(
            tokens=tokens,
            word_indices=word_indices,
            token_indices=token_indices
        )
        for i in range(n):
            features[f'e{i}_dist'] = entity_distances[i]
            features[f'e{i}_text'] = rec['entities'][i]['text']
        return features

    df1 = pd.DataFrame([(rec['id'], rec['label']) for rec in records], columns=['id', 'label'])
    df2 = pd.DataFrame([get_features(rec) for rec in records])
    return pd.concat([df1, df2], axis=1)
