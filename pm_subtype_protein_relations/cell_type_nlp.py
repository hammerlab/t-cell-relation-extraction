import re
import spacy
import pandas as pd
nlp = spacy.load("en_scispacy_core_web_sm")

# # *Potentially useful function for recursive subtree search
# # def find_node(tokens, p):
# #     for token in tokens:
# #         if p(token):
# #             return token
# #     else:
# #         children = [t for token in tokens for t in list(token.children)]
# #         if not children:
# #             return None
# #         return find_node(children, p)

def prep(text):
    text = re.sub('<(sup|sub)>([a-z0-9A-Z._+-]*)</(sup|sub)>', '\\2 ', text)
    text = re.sub('[tT]\\-([cC]ell[s]*|[lL]ymphocyte[s]*)', 'T \\1', text)
    return text

def clean(term):
    term = term.strip()
    if term.endswith('s'):
        term = term[:-1]
    term = re.sub('[\\(\\)\\-\\+]', '', term).upper().strip()
    term = term.replace('CELLS', '')
    term = term.replace('CELL', '')
    return term

def tag_tcell_types(text, details=False):
    doc = nlp(prep(text))
    res = []
    for token in doc:
        if token.text.startswith('T') or token.text.endswith('T'): 
            if token.dep_ in ['appos', 'amod', 'conj', 'nmod', 'nounmod']:
                if token.head.lemma_ in ['cell', 'lymphocyte']:
                    type_text = clean(token.text)
                    if type_text != 'T':
                        res.append((type_text, token.idx, len(token.text)))
    df = pd.DataFrame(res, columns=['type', 'offset', 'len'])
    if details:
        return df
    cts = df['type'].value_counts()
    cts.index.name = 'type'
    cts.name = 'count'
    return cts.reset_index()