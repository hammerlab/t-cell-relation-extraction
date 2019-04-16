import pandas as pd
import mygene
mg = mygene.MyGeneInfo()


def mygene_batch_query(**kwargs):
    kwargs['as_dataframe'] = True
    start = 0
    dfs = []
    while True:
        kwargs['size'] = 1000
        kwargs['from'] = start
        df = mg.query(**kwargs)
        start += kwargs['size']
        if len(df) == 0:
            break
        dfs.append(df)
    return pd.concat(dfs)
        

def mygene_query(query, species='human', score_threshold=20):
    query_args = dict(
        q=query,
        scopes=["symbol", "retired", "name", "alias"],
        fields='symbol,name,taxid,ensembl.gene,alias', 
        species=species, 
        as_dataframe=True
    )
    df = mygene_batch_query(**query_args)
    print('Original result info:')
    print(df.info())
    
    print('Score histogram:')
    ax = df['_score'].hist(bins=32)
    if score_threshold is not None:
        ax.vlines(x=score_threshold, ymin=0, ymax=100)
        df = df[df['_score'] > score_threshold]
        
    df = df.rename(columns={'_id': 'extid'}).drop('_score', axis=1)
    df['extid'] = df['extid'].astype(str)
    return df
    
def mygene_prep(df, label_fn=None):
    dfs = []
    if label_fn is None:
        label_fn = lambda v: v
    for i, r in df.iterrows():
        lbl = label_fn(r['symbol'])
        dfs.append((lbl, lbl, r['extid']))
        dfs.append((r['name'], lbl, r['extid']))
        if r['symbol'] != lbl:
            dfs.append((r['symbol'], lbl, r['extid']))
        if r['alias'] is None or isinstance(r['alias'], float):
            aliases = []
        else:
            aliases = [r['alias']] if isinstance(r['alias'], str) else r['alias']
        for alias in aliases:
            dfs.append((alias, lbl, r['extid']))
    dfs = pd.DataFrame(dfs, columns=['sym', 'lbl', 'extid'])
    return dfs

def _get_greek_alphabet():
    from itertools import chain
    from unicodedata import name
    greek_codes   = chain(range(0x370, 0x3e2), range(0x3f0, 0x400))
    greek_symbols = (chr(c) for c in greek_codes)
    greek_letters = [c for c in greek_symbols if c.isalpha()]
    alphabet = [l for l in greek_letters if 'GREEK SMALL LETTER' in name(l) and len(name(l).split()) == 4]
    return alphabet

def get_greek_alphabet():
    import unidecode
    from unicodedata import name
    alphabet = _get_greek_alphabet()
    return [
        # Return tuples as (symbol, name, english transliteration) -> e.g. (α, alpha, a)
        (l, name(l).replace('GREEK SMALL LETTER', '').strip().lower(), unidecode.unidecode(l))
        for l in alphabet
        if unidecode.unidecode(l) != '[?]'
    ]