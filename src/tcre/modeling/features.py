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


def candidate_to_entities(cand):
    sent = cand.get_parent()
    # Get list of candidate entities as well as all spans, including those for non-target entities
    cand_entities = cand.get_contexts()
    sent_entities = sent.get_children()
    ents = []
    # Extract per-entity information
    for span in sent_entities:
        typ = sent.entity_types[span.get_word_start()]
        cid = sent.entity_cids[span.get_word_start()]
        assert typ != 'O' and cid != 'O'
        ents.append(dict(
            word_range=span.get_word_range(), text=span.get_span(),
            type=typ, is_candidate=span in cand_entities,
            cid=cid
        ))
    return ents


def candidates_to_records(cands, entity_predicate=None):
    def get_record(cand):
        ents = candidate_to_entities(cand)
        if entity_predicate is not None:
            ents = [e for e in ents if entity_predicate(e)]
        return dict(
            id=cand.id,
            text=str(cand.get_parent().text),
            words=list(cand.get_parent().words),
            label=cand.gold_labels[0].value if cand.gold_labels else 0,
            entities=ents
        )
    return [get_record(cand) for cand in cands]


MAX_POS_DIST = 127


def get_dist_bin(i, rng):
    if i < rng[0]:
        v = i - rng[0]
    elif i > rng[1]:
        v = i - rng[1]
    else:
        v = 0
    return np.clip(v, -MAX_POS_DIST, MAX_POS_DIST)


DEFAULT_MARKERS = {
    'primary': {'immune_cell_type': ['<<', '>>'], 'cytokine': ['[[', ']]'], 'transcription_factor': ['{{', '}}']},
    'secondary': {'immune_cell_type': ['##', '##'], 'cytokine': ['%%', '%%'], 'transcription_factor': ['**', '**']}
}


def get_record_features(records, markers=DEFAULT_MARKERS, swaps=None, subtokenizer=None, lower=False, assert_unique=True):

    if subtokenizer is None:
        subtokenizer = lambda t: [t]

    def get_features(rec):
        # First ensure that none of the markers appear in the text within which they should be unique
        text = rec['text']
        if assert_unique and markers:
            for m in [v for m in markers.values() for l in m.values() for v in l]:
                if m in text:
                    raise AssertionError(f'Found marker "{m}" in text "{text}"')

        # Determine target entities for candidate
        n = len(rec['entities'])
        entity_indices = [i for i, e in enumerate(rec['entities']) if e['is_candidate']]

        # Extract word sequence and word spans corresponding to all entities
        words = list(rec['words'])
        positions = [rec['entities'][i]['word_range'] for i in range(n)]

        # Push to uncased if requested
        if lower:
            words = [w.lower() for w in words]

        # Add markers around entity spans (if any markers are provided)
        status = {True: 'primary', False: 'secondary'}
        tokens = list(words)
        indices = list(range(len(words)))
        if markers:
            marks = [v for e in rec['entities'] for v in markers[status[e['is_candidate']]][e['type']]]
            tokens = mark_entities(tokens, positions, style='insert', markers=marks)
            # Re-space word index list in the same manner as the words themselves
            indices = mark_entities(indices, positions, style='insert', markers=[None] * len(marks))

        # Use potentially re-spaced index list to map original entity word positions to new positions
        positions = [(indices.index(positions[i][0]), indices.index(positions[i][1])) for i in range(n)]

        subtokens = [(i0, t0, i1, t1) for i0, t0 in enumerate(tokens) for i1, t1 in enumerate(subtokenizer(t0))]
        tokens = []
        word_indices = []
        token_indices = []
        entity_distances = [[] for _ in range(len(entity_indices))]
        for i, (i0, t0, i1, t1) in enumerate(subtokens):
            word_indices.append(indices[i0])
            token_indices.append(i0)
            token = t1
            for j, ei in enumerate(entity_indices):
                # Get token range corresponding to candidate entity and calculate
                # relative distance from current token
                rng = positions[ei]
                entity_distances[j].append(get_dist_bin(i0, rng))
                if swaps and rng[0] <= i0 <= rng[1]:
                    ent = rec['entities'][ei]
                    token = swaps[ent['type']]
            tokens.append(token)

        features = dict(
            tokens=tokens,
            word_indices=word_indices,
            token_indices=token_indices
        )
        for i, ei in enumerate(entity_indices):
            features[f'e{i}_dist'] = entity_distances[i]
            features[f'e{i}_text'] = rec['entities'][ei]['text']
        return features

    df1 = pd.DataFrame([(rec['id'], rec['label']) for rec in records], columns=['id', 'label'])
    df2 = pd.DataFrame([get_features(rec) for rec in records])
    return pd.concat([df1, df2], axis=1)
