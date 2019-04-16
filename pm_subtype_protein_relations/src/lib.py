import unicodedata
import pandas as pd
import os.path as osp
from interlap import InterLap
from env import META_DATA_DIR

SPECIES_HUMAN_ID = 1
SPECIES_MOUSE_ID = 2

CELL_TYPES = 'cell_types'
CYTOKINES = 'cytokines'
TRANSCRIPTION_FACTORS = 'transcription_factors'


def fix_jupyter_spacy_config():
    # Work-around for https://github.com/explosion/spaCy/issues/3208
    from IPython.core.getipython import get_ipython
    ip = get_ipython()
    ip.config['IPKernelApp']['parent_appname'] = 'notebook'
    
    
def get_entity_meta_data(table):
    path = osp.join(META_DATA_DIR, '{}.csv'.format(table))
    return pd.read_csv(path)


class IntervalMergingDict(object):
    """Dictionary for interval keys with configurable overlap merging"""
    
    def __init__(self, merge_fn):
        self._intervals = InterLap()
        self._result = {}
        self.merge_fn = merge_fn
        
    def add(self, start, end, data=None):
        # Add the new interval and associated data
        self._intervals.add((start, end, data))
        
        # Pull a list of all intervals that overlap (including itself)
        vals = list(self._intervals.find((start, end)))
        assert len(vals) > 0, 'Expecting at least one interval result'
        
        # Merge intervals if there are more than one
        res = vals[0] if len(vals) == 1 else self.merge_fn(vals)
        if not isinstance(res, tuple) or len(res) != 3:
            raise ValueError('Merged results should be 3-item tuples, not {}'.format(res))
            
        # Delete all overlapping ranges in final result before adding merged value
        for e in vals:
            if e[:2] in self._result:
                del self._result[e[:2]]
        self._result[res[:2]] = res[2]
            
    def keys(self):
        return self._result.keys()
    
    def values(self):
        return self._result.values()
    
    def items(self):
        return zip(self.keys(), self.values())
        
        