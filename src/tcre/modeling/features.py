import os
import os.path as osp
import numpy as np
import pandas as pd
import tqdm
from tcre.modeling.utils import mark_entities
from tcre.supervision import LABEL_TYPE_MAP


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


def get_label(cand, label_type=LABEL_TYPE_MAP):

    # If label type is configured per split, resolve to scalar
    # lable type for candidate split
    if isinstance(label_type, dict):
        label_type = label_type[cand.split]

    if label_type not in ['gold', 'marginal']:
        raise ValueError(f'Label type must be "gold" or "marginal" (candidate = {cand})')

    # Return all labels in [0, 1] with default at 0 for candidates with no label
    if label_type == 'gold':
        if len(cand.gold_labels) > 1:
            raise AssertionError(
                f'Expecting <= 1 gold label for candidate id {cand.id} ({cand}) but got {len(cand.gold_labels)}')
        return max(cand.gold_labels[0].value, 0) if cand.gold_labels else 0
    else:
        if len(cand.marginals) > 1:
            raise AssertionError(
                f'Expecting <= 1 marginal value for candidate id {cand.id} ({cand}) but got {len(cand.marginals)}')
        return cand.marginals[0].probability if cand.marginals else 0


def candidates_to_records(cands, entity_predicate=None, label_type=LABEL_TYPE_MAP):
    def get_record(cand):
        ents = candidate_to_entities(cand)
        if entity_predicate is not None:
            ents = [e for e in ents if entity_predicate(e)]
        label = get_label(cand, label_type=label_type)
        return dict(
            id=cand.id,
            text=str(cand.get_parent().text),
            words=list(cand.get_parent().words),
            label=float(label),
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
    df2 = pd.DataFrame([get_features(rec) for rec in tqdm.tqdm(records)])
    return pd.concat([df1, df2], axis=1)
