import os
import copy
import os.path as osp

ANNOTATION_CONF_TEMPLATE = """
[entities]
{entities}

[relations]
{relations}

[events]

[attributes]
"""

VISUAL_CONF_TEMPLATE = """
[drawing]
{drawing}

[labels]
"""

DEFAULT_STYLES = [
    'bgColor:#1f77b4',  # muted blue
    'bgColor:#ff7f0e',  # safety orange
    'bgColor:#2ca02c',  # cooked asparagus green
    'bgColor:#d62728',  # brick red
    'bgColor:#9467bd',  # muted purple
    'bgColor:#8c564b',  # chestnut brown
    'bgColor:#e377c2',  # raspberry yogurt pink
    'bgColor:#7f7f7f',  # middle gray
    'bgColor:#bcbd22',  # curry yellow-green
    'bgColor:#17becf'   # blue-teal
]
    
class BratEntity(object):
    
    def __init__(self, id, type, text, start_char, end_char):
        self.id = id
        self.type = type
        self.text = text
        self.start_char = start_char
        self.end_char = end_char
        
    def to_ann(self, include_text=True):
        """Convert entity to brat stand-off format
        
        See: http://brat.nlplab.org/standoff.html
        Example: T3	Organization 33 41	Ericsson
        """
        if include_text:
            return 'T{}\t{} {} {}\t{}'.format(self.id, self.type, self.start_char, self.end_char, self.text)
        else:
            return 'T{}\t{} {} {}'.format(self.id, self.type, self.start_char, self.end_char)
    

class BratRelation(object):
    
    def __init__(self, id, type, ent1, ent2):
        self.id = id
        self.type = type
        self.ent1 = copy.deepcopy(ent1)
        self.ent2 = copy.deepcopy(ent2)
        
    def set_ids(self, id, ent1_id, ent2_id):
        self.id = id
        self.ent1.id = ent1_id
        self.ent2.id = ent2_id
        return self
        
    def to_ann(self):
        """Convert relation to brat stand-off format
        
        See: http://brat.nlplab.org/standoff.html
        Example: R1	Origin Arg1:T3 Arg2:T4
        """
        return 'R{}\t{} Arg1:T{} Arg2:T{}'.format(self.id, self.type, self.ent1.id, self.ent2.id)


class BratDocument(object):
    
    def __init__(self, id, text):
        self.entities = {}
        self.relations = {}
        self.id = id
        self.text = text
        
    def add_entities(self, entities):
        for ent in entities:
            self.entities[ent.id] = ent
        
    def add_relations(self, relations):
        for rel in relations:
            self.relations[rel.id] = rel
            
    def to_ann(self, include_entity_text=True):
        ann = []
        for k in sorted(self.entities.keys()):
            ann.append(self.entities[k].to_ann(include_text=include_entity_text))
        for k in sorted(self.relations.keys()):
            ann.append(self.relations[k].to_ann())
        return '\n'.join(ann)
    
    def assign_incremental_ids(self):
        """Assign incremental integer ids to all elements contained
        
        This is useful when building documents using ids that preserve uniqueness
        and sorting but are not amenable to brat integer indexing
        """
        # Note: brat entity/relation ids are one-based
        
        # Get mapping of old entity/relation ids to new ids
        entity_id_map = {k: i + 1 for i, k in enumerate(sorted(self.entities.keys()))}
        relation_id_map = {k: i + 1 for i, k in enumerate(sorted(self.relations.keys()))}
        
        # Apply id re-assignment to entities and relations
        for ent in self.entities.values():
            ent.id = entity_id_map[ent.id]
        for rel in self.relations.values():
            rel.id = relation_id_map[rel.id]
            rel.ent1.id = entity_id_map[rel.ent1.id]
            rel.ent2.id = entity_id_map[rel.ent2.id]
        
        # Re-map objects using new ids
        self.entities = {ent.id: ent for ent in self.entities.values()}
        self.relations = {rel.id: rel for rel in self.relations.values()}
        return self
        
    def export(self, collection_dir, include_entity_text=True):
        """Export document as separate text and brat stand-off (ann) files"""
        path = osp.join(collection_dir, self.id + '.txt')
        with open(path, 'w') as fd:
            fd.write(self.text)

        path = osp.join(collection_dir, self.id + '.ann')
        with open(path, 'w') as fd:
            fd.write(self.to_ann(include_entity_text=include_entity_text))
        return self
        
        
class BratCollection(object):
    
    def __init__(self):
        self.docs = {}
        
    def add_documents(self, docs):
        for doc in docs:
            self.docs[doc.id] = doc
        
    def get_configuration_files(self, styles=DEFAULT_STYLES):
        entities = sorted(set([
            (ent.type,)
            for doc in self.docs.values() for ent in doc.entities.values()
        ]))
        relations = sorted(set([
            (rel.type, rel.ent1.type, rel.ent2.type) 
            for doc in self.docs.values() for rel in doc.relations.values()
        ]))
        
        configs = {}
        configs['annotation.conf'] = ANNOTATION_CONF_TEMPLATE.format(
            entities='\n'.join(['{}'.format(*ent) for ent in entities]), 
            relations='\n'.join(['{}\tArg1:{}, Arg2:{}'.format(*rel) for rel in relations])
        )
        
        # Function to convert entity and relation types to styles
        def get_style(i, typ, styles):
            # Retrieve direct assignment for this entity/relation type if possible
            if isinstance(styles, dict):
                if typ in styles:
                    return styles[typ]
                styles = DEFAULT_STYLES
            # Otherwise, choose round-robin from list of styles
            return styles[i % len(styles)]
        
        configs['visual.conf'] = VISUAL_CONF_TEMPLATE.format(
            drawing='\n'.join([
                '{}\t{}'.format(typ, get_style(i, typ, styles))
                for i, typ in enumerate([ent[0] for ent in entities] + [rel[0] for rel in relations])
            ])
        )
        return configs
        
    def export(self, collection_dir, styles=DEFAULT_STYLES, include_entity_text=True):
        # Infer and create configuration files for the collection
        configs = self.get_configuration_files(styles=styles)
        for k, v in configs.items():
            path = osp.join(collection_dir, k)
            with open(path, 'w') as fd:
                fd.write(v)
                
        # Save individual documents as separate text and stand-off annotation files
        for doc_id in sorted(self.docs.keys()):
            self.docs[doc_id].export(collection_dir, include_entity_text=include_entity_text)
        
        
def spacy_doc_to_brat_doc(doc, id):
    """Convert spacy doc (with entities) to brat doc"""
    bdoc = BratDocument(id, doc.text)
    for i, ent in enumerate(doc.ents):
        bdoc.add_entities([BratEntity(i, ent.label_, ent.text, ent.start_char, ent.end_char)])
    return bdoc


def snorkel_candidates_to_brat_collection(candidates, candidate_class, text_loader_fn, relation_type_fn=None):
    bdocs = {}
    
    # Get components of candidates (e.g. ["cytokine", "cell_type"])
    classes = candidate_class.__argnames__
    if len(classes) != 2:
        raise ValueError(
            'Currently only 2 item relations are supported '
            '(candidate class with component types {} is not valid)'
            .format(classes)
        )
        
    # Loop through candidates and build up dict of documents, adding relations
    # to corresponding documents as they are encountered
    for i, c in enumerate(candidates):
        # Get Document for candidate
        sdoc = c.get_parent().get_parent()
        doc_id = sdoc.name
        
        # Fetch or create brat document
        if doc_id not in bdocs:
            # Not yet sure how to rebuild a text from snorkel context objects
            # text = '\n'.join([sent.text for sent in sdoc.sentences])
            text = text_loader_fn(doc_id)
            bdocs[doc_id] = BratDocument(doc_id, text)
        bdoc = bdocs[doc_id]

        # Get character offset in document of first letter in sentence
        sent_offset = c.get_parent().abs_char_offsets[0]
        
        # NOTE: Add one to end_char as snorkel ranges are inclusive and brat requires exclusive ranges
        ents = []
        for cls in classes:
            # Fetch object in canidate relation (e.g. c.cytokine = Cytokine instance)
            obj = getattr(c, cls)
            # Convert to brat relation annotation
            ents.append(BratEntity(
                obj.stable_id, cls, obj.get_span(), 
                sent_offset + obj.char_start, sent_offset + obj.char_end + 1
            ))
        bdoc.add_entities(ents)
        
        # Determine relation type, which may be altered based on characteristics of the relation
        # and candidates (e.g. using prediction probabilities)
        relation_type = c.type
        if relation_type_fn is not None:
            relation_type = relation_type_fn(c, i, ents)
        bdoc.add_relations([BratRelation(len(bdoc.relations) + 1, relation_type, ents[0], ents[1])])
    
    # Create the brat-compitable collection of documents
    bcol = BratCollection()
    bcol.add_documents([bdoc.assign_incremental_ids() for bdoc in bdocs.values()])
    return bcol
