import pandas as pd


class IXDB(object):
    """Model class for determining whether or not relations exist in iX database"""

    def __init__(self, data_file_path, min_papers=None):
        self.data_file_path = data_file_path
        self.min_papers = min_papers
        self.df = None

    def initialize(self):
        df = pd.read_csv(self.data_file_path)

        # Filter to records with known cell/cytokine mappings for internal IDS
        df = df[df['cell_ref_id'].notnull() & df['cytokine_ref_id'].notnull()]

        # Also filter to records with a minimum number of publications
        if self.min_papers is not None:
            df = df[df['num_papers'] >= self.min_papers]

        df = df.set_index(['cell_ref_id', 'cytokine_ref_id']).sort_index()
        self.df = df
        return self

    def is_relation(self, ct_id, ck_id, actor, category):
        if (ct_id, ck_id) in self.df.index:
            df = self.df.loc[(ct_id, ck_id)]
            df = df[(df['actor'] == actor) & (df['category'] == category)]
            return len(df) > 0
        return None

    def is_candidate_relation(self, cand, actor, category):
        # Split ids saved as "<matched id>:<preferred id>" to do lookup on preferred id
        ct_id, ck_id = cand.immune_cell_type_cid, cand.cytokine_cid
        ct_id, ck_id = ct_id.split(':')[1], ck_id.split(':')[1]
        return self.is_relation(ct_id, ck_id, actor, category)

