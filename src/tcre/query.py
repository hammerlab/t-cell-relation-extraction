"""Query utilities for efficient snorkel entity linking"""
import pandas as pd


class DocToCand(object):

    QUERY_TEMPLATE = (
        'SELECT D.id AS doc_id, C.id AS sentence_id, A.id AS cand_id '
        'FROM {} A '
        'INNER JOIN span B ON A.immune_cell_type_id = B.id '
        'INNER JOIN sentence C ON B.sentence_id = C.id '
        'INNER JOIN document D ON C.document_id = D.id'
    )

    @classmethod
    def _get_query(cls, cand_class):
        if 'immune_cell_type' not in cand_class.entity_types:
            raise ValueError(
                'Candidate class {} does not have required entity type "immune_cell_type"'.format(cand_class.field))
        return DocToCand.QUERY_TEMPLATE.format(cand_class.field)

    @classmethod
    def _run_query(cls, con, query):
        rs = con.execute(query)
        return pd.DataFrame([r for r in rs], columns=rs.keys())

    @classmethod
    def all(cls, session, classes):
        con = session.connection()
        df = pd.concat([
            DocToCand._run_query(con, DocToCand._get_query(classes[c])).assign(cand_type=classes[c].field)
            for c in classes
        ])
        return df
