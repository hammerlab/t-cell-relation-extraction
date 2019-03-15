import unicodedata
import pandas as pd
import os.path as osp
from env import META_DATA_DIR
SPECIES_HUMAN_ID = 1
SPECIES_MOUSE_ID = 2

def fix_jupyter_spacy_config():
    # Work-around for https://github.com/explosion/spaCy/issues/3208
    from IPython.core.getipython import get_ipython
    ip = get_ipython()
    ip.config['IPKernelApp']['parent_appname'] = 'notebook'
    
    
def get_entity_meta_data(table):
    path = osp.join(META_DATA_DIR, '{}.csv'.format(table))
    return pd.read_csv(path)