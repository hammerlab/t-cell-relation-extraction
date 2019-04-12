import pandas as pd
import mygene
mg = mygene.MyGeneInfo()

def mygene_query(query, species='human', score_threshold=20):
    df = mg.query(
        q=query,
        size=1000,
        scopes=["symbol", "retired", "name", "alias"],
        fields='symbol,name,taxid,ensembl.gene,alias', 
        species=species, 
        as_dataframe=True
    )
    print('Original result info:')
    print(df.info())
    
    print('Score histogram:')
    ax = df['_score'].hist(bins=32)
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