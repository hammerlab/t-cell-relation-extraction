import pandas as pd
from tcre import integration
from tcre import lib
from spacy.tokens import Doc
from spacy.pipeline import EntityRuler
from spacy.tokens import Span
from tcre.lib import CYTOKINES, TRANSCRIPTION_FACTORS, CELL_TYPES
from tcre.supervision import ENT_TYP_CK, ENT_TYP_TF, ENT_TYP_CT

if not Doc.has_extension('pmc_id'):
    Doc.set_extension('pmc_id', default=None)
if not Span.has_extension('meta'):
    Span.set_extension('meta', default=None)


class EntityFilter(object):
    name = 'filter'

    def __init__(self, types):
        self.types = types

    def __call__(self, doc):
        if self.types is None:
            return doc
        doc.ents = [e for e in doc.ents if e.label_ in self.types]
        return doc


def copy_ent(ent):
    return dict(
        type=ent.label_,
        meta=ent._.meta if hasattr(ent._, 'meta') else None,
        start_chr=ent.start_char,
        end_chr=ent.end_char,
        start_wrd=ent.start,
        end_wrd=ent.end,
        text=ent.text
    )


class EntityRelocator(object):

    def __init__(self, attr):
        self.attr = attr
        if not Doc.has_extension(attr):
            Doc.set_extension(attr, default=None)

    def __call__(self, doc):
        setattr(doc._, self.attr, [copy_ent(e) for e in doc.ents])
        doc.ents = []
        return doc


class EntityRefiner(object):
    name = 'refiner'

    def __init__(self, nlp, df):
        self.ruler = EntityRuler(nlp, overwrite_ents=True)
        if not df['id'].is_unique:
            raise ValueError('All entity records must have a unique id')
        self.df = df.copy().set_index('id')
        patterns = []
        for rid, r in self.df.iterrows():
            # TODO: add match types into meta data frames
            tokens = nlp.tokenizer(r['sym'])  # returns Doc
            pattern = [{'lower': t.lower_} for t in tokens]

            # Assign label as record id (e.g. "CKBF6003C60D23BA0D")
            patterns.append({'label': rid, 'pattern': pattern})
        self.ruler.add_patterns(patterns)

    def __call__(self, doc):
        doc = self.ruler(doc)
        ents = []
        for ent in doc.ents:
            if ent.label_ not in self.df.index:
                ents.append(ent)
                continue
            # Convert label of span from record id to entity type
            # (e.g. CKBF6003C60D23BA0D -> CYTOKINE)
            r = self.df.loc[ent.label_]
            rid = ent.label_
            ent = Span(doc, ent.start, ent.end, label=r['type'])
            ent._.meta = dict(id=rid, prefid=r['prefid'], lbl=r['lbl'])
            ents.append(ent)
        doc.ents = ents
        return doc


def get_pipeline(include_jnlpba=True):
    from spacy.tokens import Doc
    from spacy.tokens import Span

    if not Doc.has_extension('pmc_id'):
        Doc.set_extension('pmc_id', default=None)
    if not Span.has_extension('meta'):
        Span.set_extension('meta', default=None)

    nlp = integration.get_scispacy_pipeline(model='en_ner_jnlpba_md')

    df = pd.concat([
        lib.get_entity_meta_data(CYTOKINES).assign(type=ENT_TYP_CK),
        lib.get_entity_meta_data(TRANSCRIPTION_FACTORS).assign(type=ENT_TYP_TF),
        lib.get_entity_meta_data(CELL_TYPES).assign(type=ENT_TYP_CT)
    ], sort=True)
    refiner = EntityRefiner(nlp, df)

    # If using multiple NER sources, add both NER components and corresponding "relocator"
    # steps to support multiple NER attributes for each token
    if include_jnlpba:
        nlp.add_pipe(EntityFilter(['CELL_TYPE', 'CELL_LINE']), after='ner')
        nlp.add_pipe(EntityRelocator('ents_jnlpba'), after=EntityFilter.name, name='reloc_jnlpba')
        nlp.add_pipe(refiner, after='reloc_jnlpba')
        nlp.add_pipe(EntityRelocator('ents_lkp'), after=EntityRefiner.name, name='reloc_lkp')
    # Otherwise, use standard attributes
    else:
        nlp.replace_pipe('ner', refiner)

    return nlp


def snorkel_cid(meta):
    # Combine id for matching entity with preferred match id (for convenience)
    return meta['id'] + ':' + meta['prefid']


def snorkel_ent_fn(ent):
    return dict(
        type=ent.label_.lower(),
        cid=snorkel_cid(ent._.meta)
    )
