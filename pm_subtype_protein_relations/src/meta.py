import pandas as pd
import numpy as np
import mygene
import hashlib
ID_TYP_CK = 'CK'
ID_TYP_TF = 'TF'
ID_TYP_CT = 'CT'
MIN_CK_SYM_LEN = 3
MIN_TF_SYM_LEN = 4
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

def get_ids(df, typ):
    ids = [':'.join([r['src'], str(r['spid']), r['sym'], r['lbl']]) for _, r in df.iterrows()]
    return [typ + hashlib.md5(v.encode('utf-8')).hexdigest()[:16].upper() for v in ids]

def merge(dfs, typ):
    cols = ['id', 'src', 'sym', 'lbl', 'spid', 'extid']
    dfs = [df.copy() for df in dfs]
    for df in dfs:
        if 'id' in df:
            raise ValueError('ID should not already be assigned')
        df['id'] = get_ids(df, typ)
    df = pd.concat([df[cols] for df in dfs])
    return df

DEFAULT_SRC_PRIORITY = {'manual': 2, 'transform': 1}

def add_preferred_id(g, src_priority=DEFAULT_SRC_PRIORITY):
    """Add preferred id for synonym records grouped by label"""
    # Ensure record group is for one label
    n_label = g['lbl'].nunique()
    assert n_label == 1, 'Expecting one label per group but {} were found'.format(n_label)
    lbl = g['lbl'].iloc[0]
    
    # Ensure all sources are non-null
    assert g['src'].notnull().all()
    
    # Ensure that each source is providing one record with a symbol equal to label
    cts = g.groupby('src').apply(lambda gi: len(gi[gi['sym'] == gi['lbl']]))
    assert np.all(cts == 1),\
        'Found at least one source mulitple or zero records where symbol equals label: counts = {}, lbl = {}'\
        .format(cts, lbl)
    
    # Choose a source to use the preferred id from
    gids = g[g['sym'] == g['lbl']]
    srcs = gids['src'].unique()
    src = srcs[np.argmax([src_priority.get(src, 0) for src in srcs])]
        
    # Set id as id for label record from preferred source
    pid = gids[gids['src'] == src]
    assert len(pid) == 1
    pid = pid.iloc[0]['id']
    return g.assign(prefid=pid)

def add_preferred_ids(df, src_priority=DEFAULT_SRC_PRIORITY):
    return df.groupby(['lbl'], group_keys=False).apply(add_preferred_id, src_priority=src_priority)