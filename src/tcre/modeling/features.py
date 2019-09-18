import os
import os.path as osp
import numpy as np
import pandas as pd
import tqdm
from tcre.modeling.utils import mark_entities
from tcre.supervision import LABEL_TYPE_MAP, ENT_TYP_CT_L, ENT_TYP_CK_L, ENT_TYP_TF_L

MAX_POS_DIST = 127
DEFAULT_MARKERS = {
    'primary': {ENT_TYP_CT_L: ['<<', '>>'], ENT_TYP_CK_L: ['[[', ']]'], ENT_TYP_TF_L: ['{{', '}}']},
    'secondary': {ENT_TYP_CT_L: ['<#', '#>'], ENT_TYP_CK_L: ['<%', '%>'], ENT_TYP_TF_L: ['<*', '*>']}
}
DEFAULT_SWAPS = {
    'primary': {ENT_TYP_CT_L: '@CL', ENT_TYP_CK_L: '@CK', ENT_TYP_TF_L: '@TF'},
    'secondary': {ENT_TYP_CT_L: '$CL', ENT_TYP_CK_L: '$CK', ENT_TYP_TF_L: '$TF'}
}


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

    if label_type not in ['gold', 'marginal', 'none']:
        raise ValueError(
            f'Label type must be "gold", "marginal", or "none" (candidate = {cand}, label type = {label_type}')

    # For gold labels, expect labels in {NA, -1, 0, 1} and map to {0, 1} (missing candidates are assumed 0)
    if label_type == 'gold':
        if len(cand.gold_labels) > 1:
            raise AssertionError(
                f'Expecting <= 1 gold label for candidate id {cand.id} ({cand}) but got {len(cand.gold_labels)}')
        res = max(cand.gold_labels[0].value if cand.gold_labels else 0, 0)
        if res not in [0, 1]:
            raise AssertionError(f'Exptecting label in {{0, 1}} for candidate {cand} but got {res}')
        return res
    # For marginal labels, enforce in [0, 1] (or NAN if not present)
    elif label_type == 'marginal':
        if len(cand.marginals) > 1:
            raise AssertionError(
                f'Expecting <= 1 marginal value for candidate id {cand.id} ({cand}) but got {len(cand.marginals)}')
        res = cand.marginals[0].probability if cand.marginals else np.float64(np.nan)
        if res < 0 or res > 1:
            raise AssertionError(f'Exptecting label in [0, 1] for candidate {cand} but got {res}')
        return res
    # If no label is expected, return NAN
    else:
        return np.float64(np.nan)


def get_candidate_labels(session, splits=None, allow_mixed_presence=False):
    """Fetch data frame with all candidate ids, types, splits, and labels"""
    from snorkel.models import Candidate
    query = session.query(Candidate)
    if splits is not None:
        query = query.filter(Candidate.split.in_(splits))
    df = pd.DataFrame([
        dict(id=c.id, type=c.type, split=c.split, label=get_label(c))
        for c in query.all()
    ])
    if not allow_mixed_presence:
        # Ensure that for each split, labels are either completely present or completely absent
        cts = df.groupby('split').apply(lambda g: g['label'].notnull().nunique())
        if any(cts > 1):
            raise ValueError(f'The following splits have a mixture of present/absent labels: {cts[cts > 1].index}')
    return df


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


def get_dist_bin(i, rng):
    if i < rng[0]:
        v = i - rng[0]
    elif i > rng[1]:
        v = i - rng[1]
    else:
        v = 0
    return np.clip(v, -MAX_POS_DIST, MAX_POS_DIST)


def get_specials(markers, swaps):
    specials = []
    # Extract tokens, ignoring None or empty strings
    specials += [v3 for k1, v1 in (markers or {}).items() for v2 in (v1 or {}).values() for v3 in (v2 or []) if v3]
    specials += [v2 for k1, v1 in (swaps or {}).items() for v2 in (v1 or {}).values() if v2]
    return list(sorted(set(specials)))


def get_record_features(records, markers=DEFAULT_MARKERS, swaps=DEFAULT_SWAPS,
                        subtokenizer=None, lower=False, assert_unique=True, include_indices=False):

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

        # Note that if the subtokenizer returns an empty sequence, the main index in this loop will be
        # skipped, which is useful for removing unwanted tokens in the sequence
        subtokens = [(i0, t0, i1, t1) for i0, t0 in enumerate(tokens) for i1, t1 in enumerate(subtokenizer(t0))]
        tokens = []
        word_indices = []
        token_indices = []
        tags = ['O'] * len(subtokens)
        entity_distances = [[] for _ in range(len(entity_indices))]
        for i, (i0, t0, i1, t1) in enumerate(subtokens):
            word_indices.append(indices[i0])
            token_indices.append(i0)
            token = t1
            for ei, e in enumerate(rec['entities']):
                rng = positions[ei]
                typ = status[e['is_candidate']]  # primary/secondary
                if typ == 'primary':
                    entity_distances[entity_indices.index(ei)].append(get_dist_bin(i0, rng))
                if rng[0] <= i0 <= rng[1]:
                    tags[i] = f"E:{typ}:{e['type']}"
                    if swaps and swaps[typ]:
                        token = swaps[typ][e['type']]

            tokens.append(token)

        features = dict(tokens=tokens, tags=tags)

        # Include these fields only if necessary as they are largely only useful for debugging
        if include_indices:
            features['word_indices'] = word_indices
            features['token_indices'] = token_indices

        # Add relative distance and entity text fields
        for i, ei in enumerate(entity_indices):
            features[f'e{i}_dist'] = entity_distances[i]
            features[f'e{i}_text'] = rec['entities'][ei]['text']
        return features

    df1 = pd.DataFrame([(rec['id'], rec['label']) for rec in records], columns=['id', 'label'])
    df2 = pd.DataFrame([get_features(rec) for rec in tqdm.tqdm(records)])
    return pd.concat([df1, df2], axis=1)


def get_scibert_text(cand):
    """ Return sentence for candidate using markers as in SciBERT

    See: https://github.com/allenai/scibert/blob/5d72d0ec50e2d3ebe971122f8b282278c210eccd/scripts/chemprot_to_relation.py#L47
    """
    text = cand.get_parent().text
    ctx = cand.get_contexts()
    assert len(ctx) == 2, 'Expecting 2 contexts, found {} in candidate {}'.format(len(ctx), cand)

    # If first context is not a cell type, set the placemark characters accordingly
    chars = ['<< ', ' >>', '[[ ', ' ]]']
    if cand.get_parent().entity_types[ctx[0].get_word_start()] == ENT_TYP_CT_L:
        chars = ['[[ ', ' ]]', '<< ', ' >>']

    # Determine which entity is first, so that the substitution to follow
    # can occur with no knowledge of order
    if ctx[0].char_start < ctx[1].char_start:
        e1, e2 = ctx
    else:
        e2, e1 = ctx

    text = ''.join([
        text[:e1.char_start],
        chars[0], text[e1.char_start:(e1.char_end + 1)], chars[1],
        text[(e1.char_end + 1):e2.char_start],
        chars[2], text[e2.char_start:(e2.char_end + 1)], chars[3],
        text[(e2.char_end + 1):]
    ])
    return text