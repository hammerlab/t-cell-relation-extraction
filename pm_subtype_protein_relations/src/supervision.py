import collections

ENT_TYP_CT = 'IMMUNE_CELL_TYPE'
ENT_TYP_CK = 'CYTOKINE'
ENT_TYP_TF = 'TRANSCRIPTION_FACTOR'
ENT_TYPES = [ENT_TYP_CT, ENT_TYP_CK, ENT_TYP_TF]

REL_CLASS_INDUCING_CYTOKINE = 'InducingCytokine'
REL_CLASS_SECRETED_CYTOKINE = 'SecretedCytokine'
REL_CLASS_INDUCING_TRANSCRIPTION_FACTOR = 'InducingTranscriptionFactor'

SPLIT_TRAIN=0
SPLIT_DEV=1
SPLIT_INFER=2
SPLIT_TEST=3

DISPLACY_ENT_OPTS = {
    "ents": [ENT_TYP_CK, ENT_TYP_CT, ENT_TYP_TF],
    "colors": {
        ENT_TYP_CK: "lightblue", 
        ENT_TYP_CT: "lightgreen", 
        ENT_TYP_TF: "lightred"
    }
}

class CandidateClass(object):
    
    def __init__(self, index, name, field, label, entity_types):
        from snorkel.models import candidate_subclass
        self.index = index # Most often used as tie to snorkel key_group
        self.name = name
        self.field = field
        self.label = label
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
            0, REL_CLASS_INDUCING_CYTOKINE, 'inducing_cytokine', 'Induction', 
            [ENT_TYP_CK.lower(), ENT_TYP_CT.lower()]
        ),
        CandidateClass(
            # * Make sure SecretedCytokine gives cytokine + cell type in same order as they 
            # will share rules for labeling functions
            1, REL_CLASS_SECRETED_CYTOKINE, 'secreted_cytokine', 'Secretion', 
            [ENT_TYP_CK.lower(), ENT_TYP_CT.lower()]
        ),
        CandidateClass(
            2, REL_CLASS_INDUCING_TRANSCRIPTION_FACTOR, 'inducing_transcription_factor', 'Differentiation', 
            [ENT_TYP_TF.lower(), ENT_TYP_CT.lower()]
        ),
    ])


def get_entity_type(cand, span):
    """Return string type of a span associated with a candidate"""
    # Find index of type for first word in span and use it to lookup type in list 
    # with length equal to number of words in sentence
    return cand.get_parent().entity_types[span.get_word_start()]


def get_cids_query(session, candidate_class, split):
    from snorkel.models import Candidate
    return session.query(Candidate.id)\
        .filter(Candidate.type == candidate_class.field)\
        .filter(Candidate.split == split)

###############################
# Labeling Function Utilities
###############################

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