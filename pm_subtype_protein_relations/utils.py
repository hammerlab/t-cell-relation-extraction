from collections import defaultdict
import re
import os
import os.path as osp

DATA_DIR = os.getenv('PUBMED_NLP_DATA_DIR', 'data')

def fix_jupyter_spacy_config():
    # Work-around for https://github.com/explosion/spaCy/issues/3208
    from IPython.core.getipython import get_ipython
    ip = get_ipython()
    ip.config['IPKernelApp']['parent_appname'] = 'notebook'
    
class Reader:

    def __init__(self, file_name):
        self.file_name = file_name

    def read(self, **kwargs):
        pass


class BioCreativeReader(Reader):

    def __init__(self, file_name):
        super().__init__(file_name)
        with open(file_name, 'r', encoding='utf8') as (f):
            self.lines = f.readlines()

    def read(self):
        """
        :return: dict of abstract's: {<id>: {'t': <string>, 'a': <string>}}
        """
        regex = re.compile('^([\\d]+)\\|([at])\\|(.+)$', re.U | re.I)
        abstracts = defaultdict(dict)
        for line in self.lines:
            matched = regex.match(line)
            if matched:
                data = matched.groups()
                abstracts[data[0]][data[1]] = data[2]

        return abstracts

    def read_entity(self):
        """
        :return: dict of entity's: {<id>: [(pmid, start, end, content, type, id)]}
        """
        regex = re.compile('^(\\d+)\\t(\\d+)\\t(\\d+)\\t([^\\t]+)\\t(\\S+)', re.U | re.I)
        ret = defaultdict(list)
        for line in self.lines:
            matched = regex.search(line)
            if matched:
                data = matched.groups()
                ret[data[0]].append(tuple([data[0], int(data[1]), int(data[2]), data[3], data[4]]))

        return ret