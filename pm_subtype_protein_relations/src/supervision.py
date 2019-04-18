import collections

ENT_TYP_CT = 'IMMUNE_CELL_TYPE'
ENT_TYP_CK = 'CYTOKINE'
ENT_TYP_TF = 'TRANSCRIPTION_FACTOR'
ENT_TYPES = [ENT_TYP_CT, ENT_TYP_CK, ENT_TYP_TF]

REL_CLASS_INDUCING_CYTOKINE = 'InducingCytokine'
REL_CLASS_SECRETED_CYTOKINE = 'SecretedCytokine'
REL_CLASS_INDUCING_TRANSCRIPTION_FACTOR = 'InducingTranscriptionFactor'

class CandidateClass(object):
    
    def __init__(self, name, field, label, entity_types):
        from snorkel.models import candidate_subclass
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

    @property
    def inducing_cytokine(self):
        return self.classes[REL_CLASS_INDUCING_CYTOKINE]
    
    @property
    def secreted_cytokine(self):
        return self.classes[REL_CLASS_SECRETED_CYTOKINE]
    
    @property
    def inducing_transcription_factor(self):
        return self.classes[REL_CLASS_INCUDING_TRANSCRIPTION_FACTOR]
            
def get_candidate_classes():
    return CandidateClasses([
        CandidateClass(
            REL_CLASS_INDUCING_CYTOKINE, 'inducing_cytokine', 'Induction', 
            [ENT_TYP_CK.lower(), ENT_TYP_CT.lower()]
        ),
        CandidateClass(
            # * Make sure SecretedCytokine gives cytokine + cell type in same order as they 
            # will share rules for labeling functions
            REL_CLASS_SECRETED_CYTOKINE, 'secreted_cytokine', 'Secretion', 
            [ENT_TYP_CK.lower(), ENT_TYP_CT.lower()]
        ),
        CandidateClass(
            REL_CLASS_INDUCING_TRANSCRIPTION_FACTOR, 'inducing_transcription_factor', 'Differentiation', 
            [ENT_TYP_TF.lower(), ENT_TYP_CT.lower()]
        ),
    ])

def get_cids_query(session, candidate_class, split):
    from snorkel.models import Candidate
    return session.query(Candidate.id)\
        .filter(Candidate.type == candidate_class.field)\
        .filter(Candidate.split == split)

###############################
# Labeling Function Utilities
###############################

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
            ('promoter', 'promote', 'promoted', 'promoting'),
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
        ]
    }
    terms_map = {}
    for k, v in terms.items():
        terms_map[k+'_n'] = ltp([r[0] for r in v])
        terms_map[k+'_v'] = ltp([r[1] for r in v] +[r[1]+('es' if r[1].endswith('s') else 's') for r in v if r[1]])
        terms_map[k+'_p'] = ltp([r[2] for r in v])
        terms_map[k+'_g'] = ltp([r[3] for r in v])

    terms_map['n_do'] = '(cannot|can\'t|will not|won\'t|does not|doesn\'t|do not|don\'t)'
    
    # Create varying length "wildcard" terms for substitution matching everything except
    # characters/phrases that typically indicate separate clauses (currently just ';')
    terms_map['wc_sm'] = '[^;]{0,30}'
    terms_map['wc_md'] = '[^;]{0,50}'
    terms_map['wc_lg'] = '[^;]{0,150}'
    terms_map['wc_xl'] = '[^;]{0,250}'
    return terms_map


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
        'negative': [
            # [cytokine] cannot produce [cell type] cells de novo from naïve T cells 
            [r'{{wc_md}}{{A}}{{wc_md}}{{n_do}} {{r_prod_v}}{{wc_md}}{{B}}'],

            # [cell type] cells do not respond to [cytokine]
            [r'{{B}}{{wc_md}} {{n_do}} (respond|react) to {{wc_md}}{{A}}'],

            # Vote negative when some kind of contrasting or punctuating clause exists between references
            # expressed high levels of [cytokine]; compared to adult subsets, [cell type]
            # [cytokine] is instrumental in directing [cell type] differentiation, 
            # whereas [cytokine] promotes [cell type] differentiation
            [r'{{A}}{{wc_lg}} (;|:|whereas|however|although) {{wc_lg}}{{B}}'],

            # cells cultured in [cytokine] or low-dose IL-2 never developed into full-fledged [cell type] cells
            [r'{{A}}{{wc_md}}({{n_do}}|(never)){{wc_md}}{{r_diff_v}}{{wc_md}}{{B}}'],

            # *References to endogenous cytokines should rarely make sense in the context of polarization
            [r'(endogenous|intracellular|intra-cellular){{wc_sm}}{{A}}'],

            # [cytokine] also antagonizes the [other cytokine]– mediated differentiation of [cell type] cells
            [r'{{A}}{{wc_lg}}(antagonizes|inhibits){{wc_lg}}{{B}}'],
        ]
    },
    REL_CLASS_SECRETED_CYTOKINE: {
        'positive': [
            # ... regulates [cell type] differentiation, inducing [cytokine] expression
            [r'{{B}}{{wc_md}}{{r_diff_n}}{{wc_md}}{{r_push_g}}{{wc_sm}}{{A}}{{wc_sm}}{{r_secr_n}}'],

            # [cell type] cells produce [cytokines]
            # [cell type] cells, which secrete [cytokine]
            [r'{{B}}{{wc_md}}{{r_secr_v}}{{wc_sm}}{{A}}'],

            # induced [cell type] cell expansion and [cytokine] release
            [r'{{B}}{{wc_md}}{{A}}{{wc_sm}}{{r_secr_n}}'],

            # Considerable amounts of [cytokine] were released by the [cell type] cells
            # significantly higher levels of [cytokine] were secreted by [cell type]
            [r'{{A}}{{wc_md}}{{r_secr_p}}{{wc_md}}{{B}}'],

            # [cell type] cells secreted significantly higher levels of [cytokine]
            [r'{{B}}{{wc_md}}{{r_secr_p}}{{wc_md}}{{A}}'],

            # by inducing the initial production of [cytokine] in [cell type] cells
            [r'{{r_secr_n}} of {{A}}{{wc_sm}}{{B}}'],

            # while [cell type] are the main source of [cytokine]
            [r'{{B}}{{wc_sm}}{{primary}}{{wc_sm}}{{provider}}{{wc_sm}}{{A}}', {
              'primary': '(main|primary|typical|conventional|usual|consistent)',
              'provider': '(source|provider|producer|generator|creator|supplier)'
            }],

        ],
        'negative': [
            
        ]
    }
}