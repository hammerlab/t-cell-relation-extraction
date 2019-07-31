import snorkel
from tcre import supervision
from tcre import integration
from snorkel.parser import CorpusParser
import pandas as pd
from snorkel.models import Document

DEFAULT_TEXT_COLS = ['title', 'abstract', 'body']


class DataFrameDocProcessor(object):

    def __init__(self, df, meta=None, id_col='id_pmc', id_prefix='PMC', meta_cols=None, text_cols=DEFAULT_TEXT_COLS,
                 limit=None):
        self.df = df
        self.meta = meta
        self.id_col = id_col
        self.id_prefix = id_prefix
        self.meta_cols = meta_cols
        self.text_cols = text_cols
        self.limit = limit or float('inf')
        self.ct = 0
        if self.meta is None:
            self.meta = {}
        # If meta cols not explicitly given, use every non-textual field
        if self.meta_cols is None:
            self.meta_cols = self.df.columns.difference(self.text_cols).to_list()

    @classmethod
    def get_stable_id(cls, doc_id):
        return "%s::document:0:0" % doc_id

    def generate(self):
        if self.df[self.id_col].isnull().any():
            raise ValueError('Found null id for document in row group {} (file = {})'.format(rg, self.path))
        for i, r in self.df.iterrows():
            if 'text' in r and not pd.isnull(r['text']) and len(r['text']) > 0:
                text = r['text']
            else:
                text = integration.combine_text(*[r[c] for c in self.text_cols])
            doc_id = self.id_prefix + r[self.id_col]
            meta = {**r[self.meta_cols].to_dict(), **self.meta}
            stable_id = self.get_stable_id(doc_id)
            doc = Document(name=doc_id, stable_id=stable_id, meta=meta)
            self.ct += 1
            if self.ct > self.limit:
                return
            yield doc, text

    def __len__(self):
        return min(len(self.df), self.limit)

    def __iter__(self):
        return self.generate()


def from_feather(path):
    return pd.read_feather(path)


class DocLoader(object):

    def __init__(self, path, load_fn=from_feather):
        self.path = path
        self.load_fn = load_fn

    def run(self, limit=None):
        from tcre import tagging
        from tcre import parsing
        df = self.load_fn(self.path)
        docs = DataFrameDocProcessor(df, limit=limit)
        nlp = tagging.get_pipeline(include_jnlpba=False)
        parser = CorpusParser(parser=parsing.SpaCyParser(nlp, ent_fn=tagging.snorkel_ent_fn))
        parser.apply(docs, clear=False)



