import collections
import string
import pandas as pd
import numpy as np
from snorkel.lf_helpers import rule_regex_search_tagged_text

ENT_TYP_CT = 'IMMUNE_CELL_TYPE'
ENT_TYP_CK = 'CYTOKINE'
ENT_TYP_TF = 'TRANSCRIPTION_FACTOR'
ENT_TYP_CT_L = ENT_TYP_CT.lower()
ENT_TYP_CK_L = ENT_TYP_CK.lower()
ENT_TYP_TF_L = ENT_TYP_TF.lower()
ENT_TYPES = [ENT_TYP_CT, ENT_TYP_CK, ENT_TYP_TF]

REL_CLASS_INDUCING_CYTOKINE = 'InducingCytokine'
REL_CLASS_SECRETED_CYTOKINE = 'SecretedCytokine'
REL_CLASS_INDUCING_TRANSCRIPTION_FACTOR = 'InducingTranscriptionFactor'

REL_FIELD_INDUCING_CYTOKINE = 'inducing_cytokine'
REL_FIELD_SECRETED_CYTOKINE = 'secreted_cytokine'
REL_FIELD_INDUCING_TRANSCRIPTION_FACTOR = 'inducing_transcription_factor'

REL_INDCK = 'indck'
REL_SECCK = 'secck'
REL_INDTF = 'indtf'

REL_ABBRS = collections.OrderedDict([
    (REL_INDCK, REL_CLASS_INDUCING_CYTOKINE),
    (REL_SECCK, REL_CLASS_SECRETED_CYTOKINE),
    (REL_INDTF, REL_CLASS_INDUCING_TRANSCRIPTION_FACTOR),
])

COMP_CLASSES = {
    REL_FIELD_INDUCING_CYTOKINE: REL_FIELD_SECRETED_CYTOKINE,
    REL_FIELD_SECRETED_CYTOKINE: REL_FIELD_INDUCING_CYTOKINE
}

SPLIT_TRAIN = 0
SPLIT_DEV = 1
SPLIT_TEST = 2
SPLIT_VAL = 3
SPLIT_INFER = 9

SPLIT_MAP = {SPLIT_TRAIN: 'train', SPLIT_DEV: 'dev', SPLIT_INFER: 'infer', SPLIT_TEST: 'test', SPLIT_VAL: 'val'}
SPLIT_MAPI = {v: k for k, v in SPLIT_MAP.items()}
SPLIT_GOLD_LABELS = [SPLIT_DEV, SPLIT_VAL, SPLIT_TEST]


def get_expected_label_type(split):
    if split in [SPLIT_DEV, SPLIT_TEST, SPLIT_VAL]:
        return 'gold'
    if split in [SPLIT_TRAIN]:
        return 'marginal'
    return 'none'


LABEL_TYPE_MAP = {k: get_expected_label_type(k) for k in SPLIT_MAP.keys()}
DISPLACY_ENT_OPTS = {
    "ents": [ENT_TYP_CK, ENT_TYP_CT, ENT_TYP_TF],
    "colors": {
        ENT_TYP_CK: "lightblue", 
        ENT_TYP_CT: "lightgreen", 
        ENT_TYP_TF: "lightred"
    }
}


class CandidateClass(object):
    
    def __init__(self, index, name, field, label, abbr, entity_types):
        from snorkel.models import candidate_subclass
        self.index = index  # Most often used as tie to snorkel key_group
        self.name = name
        self.field = field
        self.label = label
        self.abbr = abbr
        self.entity_types = entity_types
        self.subclass = candidate_subclass(name, entity_types)
        
    def __repr__(self):
        return 'CandidateClass({!r})'.format(self.__dict__)


class CandidateClasses(object):
    
    def __init__(self, classes):
        self.classes = collections.OrderedDict([(c.name, c) for c in classes])
        self.items = self.classes.items
        self.values = self.classes.values
        
    def __iter__(self):
        for k in self.classes:
            yield k
            
    def __getitem__(self, k):
        return self.classes[k]
    
    def index(self, k):
        return self.classes[k].index

    @property
    def inducing_cytokine(self):
        return self.classes[REL_CLASS_INDUCING_CYTOKINE]
    
    @property
    def secreted_cytokine(self):
        return self.classes[REL_CLASS_SECRETED_CYTOKINE]
    
    @property
    def inducing_transcription_factor(self):
        return self.classes[REL_CLASS_INDUCING_TRANSCRIPTION_FACTOR]


def get_candidate_classes():
    return CandidateClasses([
        CandidateClass(
            0, REL_CLASS_INDUCING_CYTOKINE, REL_FIELD_INDUCING_CYTOKINE, 'Induction',
            REL_INDCK, [ENT_TYP_CK_L, ENT_TYP_CT_L]
        ),
        CandidateClass(
            # * Make sure SecretedCytokine gives cytokine + cell type in same order as they 
            # will share rules for labeling functions
            1, REL_CLASS_SECRETED_CYTOKINE, REL_FIELD_SECRETED_CYTOKINE, 'Secretion',
            REL_SECCK, [ENT_TYP_CK_L, ENT_TYP_CT_L]
        ),
        CandidateClass(
            2, REL_CLASS_INDUCING_TRANSCRIPTION_FACTOR, REL_FIELD_INDUCING_TRANSCRIPTION_FACTOR, 'Differentiation',
            REL_INDTF, [ENT_TYP_TF_L, ENT_TYP_CT_L]
        ),
    ])


#########################
# Snorkel Data Utilities
#########################

def get_entity_type(cand, span):
    """Return string type of a span associated with a candidate"""
    # Find index of type for first word in span and use it to lookup type in list 
    # with length equal to number of words in sentence
    return cand.get_parent().entity_types[span.get_word_start()]


def get_cids_query(session, candidate_class, split):
    """Fetch list of candidate ids for a particular class"""
    from snorkel.models import Candidate
    return session.query(Candidate.id)\
        .filter(Candidate.type == candidate_class.field)\
        .filter(Candidate.split == split)


def get_gold_labels_matrix(session, candidate_class, split, **kwargs):
    from snorkel.annotations import csr_LabelMatrix, load_matrix
    from snorkel.models import GoldLabelKey, GoldLabel

    # Set candidates to get labels for
    cids_query = get_cids_query(session, candidate_class, split)

    # Set annotator keys (expect relation name substring)
    key_names = [k[0] for k in session.query(GoldLabelKey.name).all() if candidate_class.field in k[0]]
    if len(key_names) == 0:
        raise ValueError(f'Failed to find annotator names for relation class {candidate_class.field}')

    # Fetch labels for all candidates and keys determined above
    y = load_matrix(
        csr_LabelMatrix, GoldLabelKey, GoldLabel, session,
        key_names=key_names, cids_query=cids_query, split=split, **kwargs)

    # Ensure size of labels array matches expectations
    ct = cids_query.count()
    assert ct == y.shape[0], \
        'Gold labels matrix length ({}) does not match candidate count ({}) for class {}, split {}'\
        .format(y.shape[0], ct, candidate_class.field, split)
    return y


def get_gold_labels(session, candidate_class, split, **kwargs):
    """ Return gold labels for candidates as numpy array (all -1 or 1) with candidate id as index """
    y = get_gold_labels_matrix(session, candidate_class, split, load_as_array=False, **kwargs)
    assert y.ndim == 2, f'Expecting 2D labels array but got shape {y.shape}'

    def get_value(r, cand_id):
        if (r != 0).sum() > 1:
            raise AssertionError(f'Found multiple annotations for candidate id {cand_id}')
        # For all annotators, labels should be present or considered negative when absent
        if (r != 0).sum() == 0:
            return -1
        return r[r != 0][0]

    # Get label_id -> cand_id mapping
    cand_idx = y.row_index
    y = np.array([get_value(r, cand_idx[i]) for i, r in enumerate(y.toarray())])
    assert np.all(np.in1d(y, [-1, 1]))
    return pd.Series(y, index=[cand_idx[i] for i in range(len(y))])


###############################
# Labeling Function Utilities
###############################


def get_cand_span_type(c, span):
    # Return entity type for sentence at position of first word for span
    return c.get_parent().entity_types[span.get_word_range()[0]]


def get_cand_sibling(c, strict=True):
    """Return a candidate of a different type for relation classes specified on identical spans"""
    if c.type not in COMP_CLASSES:
        return None
    entity_types = c.__class__.__argnames__  # ['cytokine', 'immune_cell_type']
    sibl_type = COMP_CLASSES[c.type]  # 'secreted_cytokine'
    cand_spans = sorted(c.get_contexts(), key=lambda v: v.id)
    span_map = {get_cand_span_type(c, span): span for i, span in enumerate(cand_spans)}  # map spans by type
    # Find sibling candidates through backref fields like "secreted_cytokine_cytokines"
    # * only first entity type is necessary for lookup since the sibling candidate will be attached to both
    sibl_cands = getattr(span_map[entity_types[0]], sibl_type + '_' + entity_types[0] + 's')
    # Filter to candidate with same spans
    sibl_cands = [s for s in sibl_cands if sorted(s.get_contexts(), key=lambda v: v.id) == cand_spans]
    if strict and len(sibl_cands) != 1:
        raise ValueError(
            f'Failed to find exactly one sibling candidate for candidate {c} (siblings found = {sibl_cands})')
    return sibl_cands[0] if sibl_cands else None


def is_a_before_b(c):
    """ Return signed result for whether or not first entity type appears before second """
    from snorkel.lf_helpers import get_tagged_text
    text = get_tagged_text(c)
    return 1 if text.index('{{A}}') < text.index('{{B}}') else -1


def has_closer_reference(c, right=True):
    """Determine if there is a closer entity reference in a relation
    
    Args:
        right: If true, this indicates that the right-most entity defined in a two-entity
            relation will be the "source" and the left-most entity will be the "target" 
            (and vise-versa when false).  This is true regardless of the order of the 
            appearance of the entities in a candidate -- i.e. if the relation is defined
            as [cytokine, cell_type], then saying right=True is equivalent to looking
            for a closer cytokine to the cell type than the current one.
    Returns:
        1 or 0 indicating whether or not another target entity exists around the source entity
        within a distance half that on either side of the source entity of the original distance
        between the two (1 if true).
    """
    # Candidates are list of "Context" objects (Spans in this case) which will
    # always be in the order of the entity types defined for the relation class
    source_span, target_span = (c[1], c[0]) if right else (c[0], c[1])
    first_span, last_span = (c[0], c[1]) if is_a_before_b(c) > 0 else (c[1], c[0])
        
    # Get word index start/end using original ordering of entities
    dist = last_span.get_word_start() - first_span.get_word_end()
    assert dist >= 0, \
        'Candidate ({}) has distance ({}) between references < 0'.format(c, dist)
    
    # "source" = entity type to search around, "target" = entity type to search for
    sent = c.get_parent()
    target_typ = sent.entity_types[target_span.get_word_start()]
    assert len(target_typ) > 0, 'Candidate ({}) has empty target type'.format(c)
    target_cid = sent.entity_cids[target_span.get_word_start()].split(':')[-1]
    
    # Search within a dist // 2 window around the target entity for an entity of the 
    # OTHER type, that is not equal to the non-target entity type within this candidate
    start = max(0, source_span.get_word_start() - dist // 2)
    end = min(len(sent.words), source_span.get_word_end() + dist // 2)
    for i in range(start, end):
        if source_span.get_word_start() <= i <= source_span.get_word_end():
            continue
        ent_typ, cid = sent.entity_types[i], sent.entity_cids[i].split(':')[-1]
        if ent_typ == target_typ and cid != target_cid:
            return 1
    return 0


PUNC_TOKENS = list('!.:;?')
PAREN_TOKENS = list('()[]{}')


def num_punctuation_tokens(tokens):
    return len([t for t in tokens if t in PUNC_TOKENS])


def num_paren_tokens(tokens):
    return len([t for t in tokens if t in PAREN_TOKENS])


def num_newlines(text):
    return text.count('\n')


FIG_TOKENS = [
    'p', 'table', 'tables', 'figure', 'figures', 'fig.', 'fig', 'diagram', 'diagrams',
    'plot', 'plots', 'chart', 'charts', 'graph', 'graphs', 'error', 'formula', 'formulas',
    'abbreviation', 'abbreviations'
]


def num_fig_tokens(tokens):
    ct = 0
    for t in tokens:
        tl = t.lower()
        if tl in FIG_TOKENS:
            ct += 1
            continue
    return ct


def candidate_high_fig_keywords(c, thresh=1):
    return num_fig_tokens(c.get_parent().words) >= thresh


def candidate_high_punctuation_tokens(c, thresh=8):
    return num_punctuation_tokens(c.get_parent().words) >= thresh


def candidate_high_paren_tokens(c, thresh=16):
    return num_paren_tokens(c.get_parent().words) >= thresh


def candidate_high_newlines(c, thresh=3):
    return num_newlines(c.get_parent().text) >= thresh


def candidate_high_characters(c, thresh=2000):
    return len(c.get_parent().text) >= thresh


def is_invalid_candidate(c, fig_kw_ct_thresh=1, punc_ct_thresh=8,
                       paren_ct_thresh=16, newline_ct_thresh=3, char_ct_thresh=2000):

    return candidate_high_fig_keywords(c, fig_kw_ct_thresh) or \
        candidate_high_punctuation_tokens(c, punc_ct_thresh) or \
        candidate_high_paren_tokens(c, paren_ct_thresh) or \
        candidate_high_newlines(c, newline_ct_thresh) or \
        candidate_high_characters(c, char_ct_thresh)


HYPOTHESIS_PREFIX = ['to']
HYPOTHESIS_TOKENS = [
    'may', 'maybe', 'whether', 'might', 'wish', 'wished', 'could',
    'analyze', 'analyzed', 'assess', 'assessed',
    'tested', 'intent', 'intended', 'goal', 'aim', 'aimed',
    'objective'
]


def is_hypothesis_candidate(c, prefix_ct_thresh=1, token_ct_thresh=1):
    """Function to detect sentences that propose a relation without confirming it

    Examples:
        - "We designed our study to capture ..."
        - "To determine if ..."
        - "The relation was analzyed using ..."
    """
    words = [w.lower() for w in c.get_parent().words]
    if prefix_ct_thresh:
        ct = len([w for w in HYPOTHESIS_PREFIX if words[0] == w])
        if ct >= prefix_ct_thresh:
            return True
    if token_ct_thresh:
        ws = set(words)
        ct = len(ws) - len(ws - set(HYPOTHESIS_TOKENS))
        if ct >= token_ct_thresh:
            return True
    return False


def is_expressed_protein(c):
    """Rule to catch cytokines and transcription factors mentioned with positive expression modifier (e.g. FoxP3+)"""
    return True if rule_regex_search_tagged_text(c, r'{{A}}\+', 1) > 0 else False


def _get_unique_entity_mentions(typs, null_val='O'):
    """Get unique entity types for a list of token entity types (which may repeat)

    Example: _get_unique_entity_mentions(['e1', 'O', 'O', 'e2', 'e2', 'e3']) -> {0: 'e1', 3: 'e2', 5: 'e3'}
    """
    uniq = np.unique(typs)
    uniq = {v: i for i, v in enumerate(uniq)}
    ind = pd.Series(typs).map(uniq).diff().values
    ind[0] = float(typs[0] != null_val)
    ind = np.argwhere(ind).ravel()
    return {i: typs[i] for i in ind if typs[i] != null_val}


def is_complex_candidate(c, entity_ct_thresh=3, char_ct_thresh=500):
    sent = c.get_parent()
    if entity_ct_thresh:
        typs = _get_unique_entity_mentions(sent.entity_types)
        if len(typs) >= entity_ct_thresh:
            return True
    if char_ct_thresh:
        if len(sent.text) >= char_ct_thresh:
            return True
    return False


#############################
# Dependency Parse Utilities
#############################


VERB_MAP = {
    REL_FIELD_INDUCING_CYTOKINE: [
        'induce', 'drive', 'direct', 'regulate', 'control', 'promote', 'rise',
        'mediate', 'cause', 'depend', 'create', 'generate', 'need', 'require', 'rely',
        'polarize', 'differentiate', 'develop', 'form', 'stabilize'
    ],
    REL_FIELD_SECRETED_CYTOKINE: [
        'secrete', 'express', 'coexpress', 'co-express', 'release',
        'produce', 'exhibit', 'display', 'show'
    ],
    REL_FIELD_INDUCING_TRANSCRIPTION_FACTOR: [
        # Differentiation program verbs
        'induce', 'drive', 'direct', 'regulate', 'control', 'promote', 'rise',
        'mediate', 'cause', 'depend', 'create', 'generate', 'need', 'require', 'rely',
        'polarize', 'differentiate', 'develop', 'form', 'stabilize',
        # Transcription/expression verbs
        'express', 'coexpress', 'co-express', 'transcribe'
    ],
    'negative': [
        # From iX verb lexicon
        'abolish', 'abrogate', 'aggravate', 'alleviate', 'antagonize',
        'arise', 'arrested', 'attenuate', 'augment', 'block', 'cleave',
        'confined', 'counteract', 'damage', 'deactivates', 'decline',
        'decrease', 'degrade', 'delay', 'delete', 'deplete', 'depress',
        'deprive', 'desensitize', 'destabilize', 'destroy', 'detached',
        'diminish', 'disable', 'disappeared', 'disrupt', 'dissect',
        'down-modulate', 'downregulate', 'down-regulate', 'draining',
        'dysregulate', 'eliminate', 'exacerbate', 'excise', 'fail',
        'hinders', 'impair', 'inactivate', 'inhibit', 'interfere',
        'interfering', 'kill', 'lack', 'limit', 'lose', 'lost', 'lower',
        'lyse', 'minimize', 'mitigate', 'obstruct', 'oppose', 'overridden',
        'predominate', 'prevent', 'reduce', 'reject', 'remove', 'repress',
        'restraining', 'restrict', 'reverse', 'spikes', 'stop', 'stress',
        'suppress', 'sustain', 'truncated', 'understand', 'underwent',
        'undifferentiate', 'unstimulate'
    ]
}


def get_parse_tree(candidate):
    """Reconstruct parse tree from token parent indicies stored in DB (from SpaCy initially)"""
    from treelib import Tree
    sent = candidate.get_parent()
    words = sent.words
    tree = Tree()

    # Map token index to token data
    nodes = {
        i: dict(
            token=words[i], dep_label=sent.dep_labels[i],
            # Dep parents are stored with one-based index
            dep_parent=sent.dep_parents[i] - 1,
            index=i, node_id=i
        )
        for i in range(len(words))
    }

    # Recursive method for building tree that ensures parent is always added first
    def add_node(n):
        parent_id = None
        if n['dep_label'] != 'ROOT':
            parent = add_node(nodes[n['dep_parent']])
            parent_id = parent.identifier
        if not tree.contains(n['node_id']):
            tree.create_node(tag=n['token'], identifier=n['node_id'], parent=parent_id, data=n)
        return tree.get_node(n['node_id'])

    # Build tree up from every possible starting token
    for i in range(len(words)):
        add_node(nodes[i])
    assert len(tree) == len(words)
    return tree


class DependencyParseTree(object):

    def get_relation_verbs(self, doc, tree, verbs, entities, between=True):
        res = []
        entities = sorted(entities)
        for t in doc:
            if not (t['pos'].startswith('VB') or t['pos'] == 'VERB'):
                continue
            # Ensure verb lemma is in target set
            if t['lemma'] not in verbs:
                continue
            # Ignore verbs not between the entities, if requested
            if between:
                if t['index'] <= entities[0] or t['index'] >= entities[1]:
                    continue
            # Fitler to verbs with both entities in subtree
            subtree = tree.subtree(t['index'])
            indexes = set([n.data['index'] for n in subtree.all_nodes()])
            for ei in entities:
                if ei not in indexes:
                    continue
            res.append(t)
        return res

    def is_candidate_relation(self, c, typ):
        verbs = VERB_MAP.get(typ)
        if verbs is None:
            raise ValueError(
                'Candidate type "{}" not supported; must be one of {}'
                .format(c.type, VERB_MAP.keys())
            )

        sent = c.get_parent()
        tree = get_parse_tree(c)
        doc = [
            dict(lemma=sent.lemmas[i], pos=sent.pos_tags[i], index=i)
            for i in range(len(sent.words))
        ]

        # Pull 0-based indexes of first tokens for each entity
        entities = [ctx.get_word_start() for ctx in c.get_contexts()]

        # Determine number of verbs implying a positive or negative relation
        pos_verbs = len(self.get_relation_verbs(doc, tree, verbs, entities, between=True))
        neg_verbs = len(self.get_relation_verbs(doc, tree, VERB_MAP['negative'], entities, between=False))

        if pos_verbs > 0 and neg_verbs == 0:
            return True
        return False


###########################
# Regex Function Utilities
###########################


def ltp(x):
    x = [v for v in x if v]
    return '(' + '|'.join(x) + ')'


def get_terms_map():
    terms = {
        'r_diff': [
            # noun, verb (present or 3rd person present), verb (past or participle), gerund
            ('differentiation', 'differentiate', 'differentiated', 'differentiating'), 
            ('formation', 'form', 'formed', 'forming'), 
            ('generation', 'generate', 'generated', 'generating'),
            ('polarization', 'polarize', 'polarized', 'polarizing'),
            ('development', 'develop', 'developed', 'developing'),
            ('induction', None, None, None),
        ],
        'r_push': [
            ('inducer', 'induce', 'induced', 'inducing'),
            ('driver', 'drive', 'drove|driven', 'driving'),
            ('director', 'direct', 'directed', 'directing'),
            ('regulator', 'regulate', 'regulated', 'regulating'),
            ('controller', 'control', 'controlled', 'controlling'),
            ('promoter', 'promote', 'promoted', 'promoting'),
            ('mediator|mediater', 'mediate', 'mediated', 'mediating')
        ],
        'r_prod': [
            ('producer|production', 'produce', 'produced', 'producing'),
            ('generator|generation', 'generate', 'generated', 'generating'),
            ('creator|creation', 'create', 'created', 'creating'),
        ],
        'r_secr': [
            ('secretor|secretion', 'secrete', 'secreted', 'secreting'),
            ('expressor|expression', 'express', 'expressed', 'expressing'),
            ('producer|production', 'produce', 'produced', 'producing'),
            ('releaser|release', 'release', 'released', 'releasing'),
        ],
        'r_oppose': [
            ('inhibitor', 'inhibit', 'inhibited', 'inhibiting'),
            ('suppressor|suppresor', 'suppress|suppres', 'suppressed|suppresed', 'suppressing|suppresing'),
            ('repressor|represor', 'repress|repres', 'repressed|represed', 'repressing|represing'),
            ('antagonizer', 'antagonize', 'antagonized', 'antagonizing'),
            ('impairer', 'impair', 'impaired', 'impairing'),
        ]
    }
    terms_map = {}
    for k, v in terms.items():
        terms_map[k+'_n'] = ltp([r[0] for r in v])
        terms_map[k+'_v'] = ltp([r[1] for r in v] +[r[1]+('es' if r[1].endswith('s') else 's') for r in v if r[1]])
        terms_map[k+'_p'] = ltp([r[2] for r in v])
        terms_map[k+'_g'] = ltp([r[3] for r in v])

    terms_map['n_do'] = '(cannot|can\'t|will not|won\'t|are not|aren\'t|does not|doesn\'t|do not|don\'t|have not|haven\'t|would not|wouldn\'t|should not|shouldn\'t|never)'
    terms_map['n_break'] = '(;|however|whereas|yet|otherwise|although|nonetheless|despite|spite of)'
    
    # Cell actions that may be the intended effect of cytokines/TF's not relevant in these relations
    terms_map['n_action'] = '(stimulation|costimulation|expansion|proliferation|migration)'
    
    # Create varying length "wildcard" terms for substitution matching everything except
    # characters/phrases that typically indicate separate clauses (currently just ';')
    terms_map['wc_sm'] = '[^;]{0,30}'
    terms_map['wc_md'] = '[^;]{0,50}'
    terms_map['wc_lg'] = '[^;]{0,150}'
    terms_map['wc_xl'] = '[^;]{0,250}'
    return terms_map


DEFAULT_NEGATIVE_PATTERNS = [
    [p % t]
    for p in [
        '{{A}}{{wc_md}} {{%s}} {{wc_md}}{{B}}',
        '{{B}}{{wc_md}} {{%s}} {{wc_md}}{{A}}'
    ]
    for t in ['r_oppose_v', 'r_oppose_n', 'r_oppose_p', 'r_oppose_g', 'n_do', 'n_break']
] + [
    # [cell type] cell migration/proliferation/stimulation (i.e. not differentation)
    [r'{{A}}{{wc_lg}}{{B}}{{wc_sm}}{{n_action}}'],
    [r'{{A}}{{wc_lg}}{{n_action}}{{wc_sm}}{{B}}'],
    [r'{{n_action}}{{wc_sm}}{{B}}{{wc_lg}}{{A}}'],
    [r'{{B}}{{wc_sm}}{{n_action}}{{wc_lg}}{{A}}'],
    [r'{{A}}{{wc_md}} negative(ly)? {{wc_md}}{{B}}'],
    [r'{{B}}{{wc_md}} negative(ly)? {{wc_md}}{{A}}']
]

LF_REGEX = {
    REL_CLASS_INDUCING_CYTOKINE: {
        'positive': [
            # predominance of [cytokine] drives [cell type] differentiation
            # [cytokine] regulates [cell type] differentiation
            [r'{{A}}{{wc_md}}{{r_push_v}}{{wc_md}}{{B}}{{wc_md}}{{r_diff_n}}'],

            # Furthermore, a key inducer of both [cell type] and [cell type] cell differentiation, [cytokine] ...
            [r'{{r_push_n}} of {{wc_md}}{{B}}{{wc_md}}{{r_diff_n}}{{wc_md}}{{A}}'],

            # [cytokine] has been shown to induce [cell type] cell differentiation
            # [cytokine] and [cytokine] induce [cell type] cell differentiation
            [r'{{A}}{{wc_md}} {{r_push_v}} {{B}} {{r_diff_n}}'],

            # whereas [cytokine], critical for [cell type] and [cell type] cell induction
            # revealed that [cytokine] was an essential cytokine in mediating [cell type] cell development
            [r'{{A}}{{wc_md}} (critical|essential|important) {{wc_md}}{{B}} {{r_diff_n}}'],

            # The role of [cytokine] ... to its ability to polarize T-helper cells toward the [cell type] type
            [r'{{A}}{{wc_lg}}ability to {{r_diff_v}}{{wc_md}}toward{{wc_md}}{{B}}'],

            # [cell type] … driven by [cytokine],
            [r'{{B}}{{wc_lg}} {{r_push_p}} (by|via|using|through) {{A}}'],

            # [cytokine] regulates [cell type] differentiation
            [r'{{A}}{{wc_md}} (regulate[s]?|control[s]?) {{wc_md}}{{B}} {{r_diff_n}}'],

            # lacking other molecules involved in [cell type] differentiation, such as [cytokine],
            [r'{{B}} {{r_diff_n}}[,]?{{wc_md}}(such as|like|including){{wc_md}}{{A}}'],

            # [cytokine], a component of the [cell type] paradigm
            [r'{{A}}[,]?{{wc_md}}(component|part|constituent) of the {{B}}'],

            # confirms that [cytokine] is a critical cytokine in the commitment to [cell type] 
            [r'{{A}}{{wc_lg}}(critical|essential){{wc_lg}}commitment to {{B}}'],

            # cells exposed to [cytokine] can … develop into [cell type],
            [r'{{A}}{{wc_lg}}{{r_diff_v}} into {{B}}'],

            # [cell type] cells require [cytokine] for their generation
            [r'{{B}}{{wc_lg}} (need[s]?|require[s]?) {{wc_lg}}{{A}}{{wc_lg}}{{r_prod_n}}'],

            # [cytokine] is important for differentiation of [cell type],
            [r'{{A}}{{wc_lg}}(critical|essential|important){{wc_md}}{{r_diff_n}}{{wc_md}}{{B}}'],

            # [cytokine] induce(s) the development of [cell type],
            # [cytokine] promote(s) differentiation into [cell type],
            [r'{{A}} {{r_push_v}}{{wc_sm}}{{r_diff_n}} (of|into) {{B}}'],

            # the receptors for [cytokine] are required for [cell type] differentiation
            [r'receptors for{{wc_md}}{{A}}{{wc_md}}(needed|required|necessary for){{wc_md}}{{B}}{{wc_md}}{{r_diff_n}}'],

            # impaired human [cell type] differentiation when [cytokine] was blocked
            [r'{{B}}{{wc_md}}{{r_diff_n}}{{wc_md}}when {{B}}{{wc_md}}blocked'],

            # [other cytokine] expression is required for [other cell type] differentiation, 
            # [cytokine] for [cell type] differentiation
            [r'{{A}} for {{B}}{{wc_md}}{{r_diff_n}}'],

            # in [cell type] differentiation, a process driven by [cytokine],
            [r'{{B}}{{wc_md}}{{r_diff_n}}{{wc_md}}{{r_push_p}} by {{A}}'],

            # role of [cytokine] in [cell type] differentiation
            [r'role of {{A}}{{wc_md}}{{B}}{{wc_md}}{{r_diff_n}}'],

            # role of [cytokine] in [cell type] differentiation
            # effects of [cytokine] in [cell type] differentiation
            [r'(effects|role) of {{A}}{{wc_md}}{{B}}{{wc_md}}{{r_diff_n}}'],

            # [cytokine] … initiate [cell type] differentiation
            [r'{{A}}{{wc_md}}(initiate|trigger|induce){{wc_md}}{{B}}{{wc_md}}{{r_diff_n}}'],

            # [cytokine] is the most potent factor that causes [other ct] to differentiate to the [cell type] phenotype
            [r'{{A}}{{wc_md}}(cause|lead|force){{wc_md}}{{r_diff_v}}{{wc_md}} (in)?to( the)? {{B}}'],

            # we show that [cytokine], probably secreted by APC cells, is able to polarize naive 
            # CD4+ T cells to [cell type] cells
            [r'{{A}}{{wc_md}}{{r_secr_p}}cells{{wc_md}}{{r_diff_v}}{{wc_md}}{{B}}'],

            # It is clear that the cytokine [cytokine] directs differentiation to a [cell type] phenotype 
            # while IL-4 can drive differentiation to a Th2 phenotype
            [r'{{A}}{{wc_md}}{{r_push_v}}{{wc_md}}{{r_diff_n}}{{wc_md}}{{B}}'],

            # upregulating IL-4 production and inhibiting IFN-γ production, thereby polarizing 
            # the differentiation of Th2 cells.
            [r'{{A}} production{{wc_md}}thereby polarizing{{wc_md}}{{r_diff_n}}{{wc_md}}{{B}}'],

            # we show that IL-6 is able to initiate the polarization of naive CD4+ T cells to effector Th2
            [r'{{A}}{{wc_sm}}{{initiate}}{{wc_sm}}{{r_diff_n}}{{wc_md}}{{B}}', {
            'initiate': '(initiate|start|cause|begin|commence|catalyze)'
            }],

            # the [cytokine]– mediated differentiation of [cell type] cells
            [r'{{A}}.{0,10}mediated {{r_diff_n}}{{wc_md}}{{B}}'],

            # [cytokine] Directs the Differentiation of [cell type] Cells.
            [r'{{A}}{{wc_sm}}{{r_push_v}}{{wc_sm}}{{r_diff_n}}{{wc_sm}}{{B}}'],
        ],
        'negative': DEFAULT_NEGATIVE_PATTERNS + [
            # *References to endogenous cytokines should rarely make sense in the context of polarization
            [r'(endogenous|intracellular|intra-cellular){{wc_sm}}{{A}}'],
        ]
    },
    REL_CLASS_SECRETED_CYTOKINE: {
        'positive': [
            [p % t]
            for p in [
                # [cell type] cells produce [cytokines]
                # [cell type] cells are a producer of [cytokines]
                # [cell type] cells produced [cytokines]
                # [cell type] cells producing [cytokine]
                r'{{B}}{{wc_md}}{{%s}}{{wc_md}}{{A}}',
                
                # **TOO OFTEN INCORRECT**
                # [cytokine] expresses highly in [cell type] cells
                # [cytokine] producer cell, [cell type] cells
                # [cytokine] secreted by [cell type] cells
                # [cytokine] producing [cell type] cells
                # r'{{A}}{{wc_sm}}{{%s}}{{wc_sm}}{{B}}',
                
                # promotes generation of cells producing IL-9 (Th9)
                #r'{{%s}}{{wc_sm}}{{A}}{{wc_sm}}{{B}}',
                
                # key cytokine secreted by Tfh cells, IL-21
                #r'{{%s}}{{wc_sm}}{{B}}{{wc_sm}}{{A}}'
            ]
            for t in ['r_secr_v', 'r_secr_n', 'r_secr_p', 'r_secr_g']
        ] + [
            
            # **TOO OFTEN INCORRECT**
            # induced [cell type] cell expansion and [cytokine] (release|secretion)
            # [cell type]-mediated therapeutic effect critically depended on [cytokine] production
            # [r'{{B}}{{wc_sm}}{{A}}{{wc_sm}}{{r_secr_v}}'],
            # [r'{{B}}{{wc_sm}}{{A}}{{wc_sm}}{{r_secr_n}}'],

            # by inducing the initial (production|release) of [cytokine] in [cell type] cells
            [r'{{r_secr_n}}{{wc_sm}}{{A}}{{wc_sm}}{{B}}'],
            [r'{{r_secr_v}}{{wc_sm}}{{A}}{{wc_sm}}{{B}}'],
            
            # Th-1-type cytokines such as interferon-γ (IFN-γ) and tumor necrosis factor-α (TNF-α)
            # Th2 cytokine, IL-4,
            # IFNγ, a Th1 cytokine
            [r'{{B}}{{wc_sm}}cytokine{{wc_sm}}{{A}}'],
            [r'{{A}}{{wc_sm}}{{B}}{{wc_sm}}cytokine'],
            
            # [cell type] subset predominated among the IL-17+ cell  [*look for expression sign]
            [r'{{A}}(\+|-)'],
            [r'{{A}}(\+|-)?(positive|negative|pos|neg|hi|lo)'],
            
            # [cell type] was strongly biased toward IL-17 rather than toward IFN-γ production
            [r'{{B}}{{wc_sm}}biased toward{{wc_sm}}{{A}}'], 

            # ... regulates [cell type] differentiation, inducing [cytokine] expression
            [r'{{B}}{{wc_md}}{{r_diff_n}}{{wc_md}}{{r_push_g}}{{wc_sm}}{{A}}{{wc_sm}}{{r_secr_n}}'],
            
            # while [cell type] are the main source of [cytokine]
            [r'{{B}}{{wc_sm}}{{primary}}{{wc_sm}}{{provider}}{{wc_sm}}{{A}}', {
              'primary': '(main|primary|typical|conventional|usual|consistent)',
              'provider': '(source|provider|producer|generator|creator|supplier)'
            }],

        ],
        'negative': DEFAULT_NEGATIVE_PATTERNS
    }, 
    REL_CLASS_INDUCING_TRANSCRIPTION_FACTOR: {
        'positive': [
            # Down-regulation of [TF] expression by [cytokine] is important for differentiation of [cell type]
            # [TF] programs the development and function of [cell type]
            # [TF] promote(s) differentiation into [cell type]
            [r'{{A}}{{wc_sm}}{{r_diff_n}}{{wc_sm}}{{B}}'],
            
            # … induce [TF], a master regulator of [cell type]
            # [TF] promotes [cell type] through direct transcriptional activation of [other TF]
            # RORγ was previously shown to regulate TH17 differentiation
            # [TF] is a key regulator of [cell type] cell differentiation
            [r'{{A}}{{wc_md}}{{r_push_n}}{{wc_md}}{{B}}'],
            [r'{{A}}{{wc_md}}{{r_push_v}}{{wc_md}}{{B}}'],
            
            # [cell type] cells differentiated from [TF] cKO mice have a severe defect
            # factors known to interfere with [cell type] differentiation do so by regulation of [TF] expression
            # Enhanced [cell type] formation by deletion of [TF]
            # TH17 lineage differentiation is programmed by orphan nuclear receptors RORα and RORγ
            [r'{{B}}{{wc_md}}{{r_diff_n}}{{wc_md}}{{A}}'],
            
            # [cell type] cell differentiation is mainly controlled by specific master transcription factors [TFs] 
            # [cell type] differentiation is driven by [TF]
            # TH17 lineage differentiation is mediated by both RORα and RORγ
            [r'{{B}}{{wc_md}}{{r_diff_n}}{{wc_md}}{{r_push_p}}{{wc_md}}{{A}}'],
            
            # there is a selective component involving [TF] in [cell type] cell differentiation
            # [TF] controls [cell type] differentiation
            # whereas [cytokine], critical for [cell type] cell induction, down-regulates [TF] expression
            # activation of [TF], a master gene for [cell type] differentiation
            # enhanced [TF]-mediated stimulation of [cell type] differentiation
            # Deletion of [TF] results in high potential for [cell type] differentiation 
            # The [TF] (necessary for [cell type] differentiation) transcription factor …
            # role for STAT3 in human central memory T cell formation
            [r'{{A}}{{wc_md}}{{B}}{{wc_md}}{{r_diff_n}}'],
            
            # [TF] regulates the developmental program resulting in modulation of the potential for [cell type] formation
            [r'{{A}}{{wc_md}} (program|processs|procedure){{wc_md}}{{B}}{{wc_md}}{{r_diff_n}}'],

            
            # suggesting that [TF] is important for sustaining the [cell type] cell phenotype
            [r'{{A}}{{wc_md}} (sustain|maintain|establish){{wc_sm}}{{B}}{{wc_sm}}({{r_diff_n}}|phenotype)'],
            
            # It has been shown that [TF] programs the commitment of the [cell type] lineage
            [r'{{A}}{{wc_md}} commitment{{wc_sm}}{{B}}{{wc_sm}}(lineage|{{r_diff_n}})'],
            
            # differentiation of Tr1 cells was dependent on the presence of the aryl hydrocarbon receptor, c-Maf [TF] and IL-27
            [r'{{r_diff_n}}{{wc_sm}}{{B}}{{wc_md}} (depend|require|necessitate){{wc_md}}{{A}}'],
            
            # can form [cell type] cells once [TF] is provided
            [r'{{B}}{{wc_md}}{{A}}{{wc_sm}} (provide|give|introduce|supplied)'],
            
            # forkhead box p3, a signature transcription factor of regulatory T cells
            [r'{{A}}{{wc_md}} signature {{wc_md}}{{B}}'],
            
            # Th1 subset is defined by expression of the lineage-determining transcription factor T-bet
            [r'{{B}}{{wc_md}} (lineage|phenotype|state)-(determin|decid|regulat|direct|dictat|govern){{wc_md}}{{A}}'],
            
        ], 
       'negative': DEFAULT_NEGATIVE_PATTERNS
    }
}