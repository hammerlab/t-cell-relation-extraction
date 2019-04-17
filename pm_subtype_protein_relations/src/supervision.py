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
