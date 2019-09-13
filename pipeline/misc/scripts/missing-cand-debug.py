import os
import pandas as pd
os.environ['SNORKELDB'] = 'sqlite:////lab/repos/t-cell-relation-extraction/data/snorkel/snorkel.bkp_20190720.db'

from snorkel import SnorkelSession
from snorkel.models import Document
session = SnorkelSession()

doc = session.query(Document).filter(Document.name == 'PMC4785102').one()
sents = doc.sentences
pd.set_option('display.max_rows', 10000)
df = pd.concat([
    pd.DataFrame(list(zip(sent.entity_types, sent.abs_char_offsets, sent.words)))
    for sent in sents[:15]
])
df.to_csv('/tmp/tokens.csv', index=False)
print(df)