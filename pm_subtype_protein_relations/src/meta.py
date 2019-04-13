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
    return df
    
def mygene_prep(df):
    dfs = []
    for i, r in df.iterrows():
        dfs.append((r['name'], r['symbol'], r['extid']))
        dfs.append((r['symbol'], r['symbol'], r['extid']))
        if r['alias'] is None or isinstance(r['alias'], float):
            aliases = []
        else:
            aliases = [r['alias']] if isinstance(r['alias'], str) else r['alias']
        for alias in aliases:
            dfs.append((alias, r['symbol'], r['extid']))
    dfs = pd.DataFrame(dfs, columns=['sym', 'lbl', 'extid'])
    return dfs