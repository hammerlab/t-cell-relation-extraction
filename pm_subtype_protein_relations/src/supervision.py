import collections

ENT_TYP_CT = 'IMMUNE_CELL_TYPE'
ENT_TYP_CK = 'CYTOKINE'
ENT_TYP_TF = 'TRANSCRIPTION_FACTOR'
ENT_TYPES = [ENT_TYP_CT, ENT_TYP_CK, ENT_TYP_TF]

RELATION_CLASSES = {
    'InducingCytokine': {
        'entity_types': [ENT_TYP_CK.lower(), ENT_TYP_CT.lower()], 
        'field': 'inducing_cytokine',
        # Relation name assigned in annotation
        'label': 'Induction'
    },
    'SecretedCytokine': {
        'entity_types': [ENT_TYP_CK.lower(), ENT_TYP_CT.lower()],
        'field': 'secreted_cytokine',
        'label': 'Secretion'
    },
    'InducingTranscriptionFactor': {
        'entity_types': [ENT_TYP_TF.lower(), ENT_TYP_CT.lower()],
        'field': 'inducing_transcription_factor',
        'label': 'Differentiation'
    }
}

def get_relation_classes():
    from snorkel.models import candidate_subclass
    
    # * Make sure SecretedCytokine gives cytokine + cell type in same order as they 
    # will share rules for labeling functions
    class_names = sorted(RELATION_CLASSES.keys())
    fields = [RELATION_CLASSES[c]['field'] + '_' + v for c in class_names for v in ['class', 'types']]
    Classes = collections.namedtuple('Classes', fields)
    args = []
    for c in class_names:
        args.append(candidate_subclass(c, RELATION_CLASSES[c]['entity_types']))
        args.append(RELATION_CLASSES[c]['entity_types'])
    classes = Classes(*args)
    return classes