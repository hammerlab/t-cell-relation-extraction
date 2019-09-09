"""Functions for labeling function result integration and analysis"""
from snorkel.models import Candidate, LabelKey
from snorkel.annotations import LabelAnnotator
from snorkel.annotations import load_gold_labels
from tcre import supervision
import logging
logger = logging.getLogger(__name__)


def _add_f1(df, eps=1e-8):
    if 'Empirical Acc.' in df:
        precision = df['TP'] / (df['TP'] + df['FP'])
        recall = df['TP'] / (df['TP'] + df['FN'])
        df['Empirical F1'] = (2 * precision * recall) / (precision + recall + eps)
    return df


def clear_labeling_functions(session):
    from snorkel.models import Label, LabelKey
    acs = [Label, LabelKey]
    cts = {c.__name__: session.query(c).count() for c in acs}
    for c in acs:
        session.query(c).delete()
    session.commit()
    for c in acs:
        assert session.query(c).count() == 0
    return cts


def get_label_key_group(candidate_class):
    # Use index of class as LabelKey.group
    return candidate_class.index


def annotation_keys_exist(session, candidate_class, annotation_key_class, key_names=None):
    """Check to see if label keys (e.g. LabelKey) exist for this candidate type"""
    key_group = get_label_key_group(candidate_class)
    query = session.query(annotation_key_class).filter(annotation_key_class.group == key_group)
    if key_names is not None:
        query = query.filter(annotation_key_class.name.in_(frozenset(key_names)))
    return query.count() > 0


def apply_labeling_functions(session, candidate_class, split, lfs=None, label_generator=None, key_names=None):
    """Persist LF labels to DB

    Returns:
        X, y, stats, labeler: X=Labels matrix, y=gold labels if available or None, stats=data frame with
            coverage and empirical accuracy/F1, labeler=snorkel LabelAnnotator
    """
    # Determine if gold labels are expected on this split
    has_gold_labels = split in supervision.SPLIT_GOLD_LABELS
    # Determine whether or not the keys for the labels already exist (if not, then replace_key_set=True means
    # that they will be created but must be set to false if they exist as snorkel will insert them otherwise
    # raising IntegrityError)
    replace_key_set = not annotation_keys_exist(session, candidate_class, LabelKey, key_names=key_names)
    # Get query for candidate ids associated with this class and split (scopes all other operations)
    cids_query = supervision.get_cids_query(session, candidate_class, split)

    # Fetch gold labels if they are expected for this split
    y = None
    if has_gold_labels:
        y = supervision.get_gold_labels(
            session, candidate_class,
            split=split
        ).values
        if len(y) == 0:
            raise ValueError(f'Failed to find gold labels for class={candidate_class.field}, split={split}')

    logger.info(
        'Running labeling for class %s, split %s (%s)',
        candidate_class.field, split, supervision.SPLIT_MAP[split]
    )
    key_group = get_label_key_group(candidate_class)
    labeler = LabelAnnotator(lfs=lfs, label_generator=label_generator)
    X = labeler.apply(
        split=split, cids_query=cids_query, clear=False,
        replace_key_set=replace_key_set, key_group=key_group
    )
    stats = X.lf_stats(session, y)
    stats = _add_f1(stats)
    return X, y, stats, labeler


def get_labels_keys(session, candidate_class):
    from snorkel.models import LabelKey
    return (
        session.query(LabelKey)
        .filter(LabelKey.group == get_label_key_group(candidate_class))
        # Always order by id to maintain consistency with snorkel.annotations.load_matrix
        .order_by(LabelKey.id)
        .all()
    )


def get_labels_matrix(session, candidate_class, split, key_names=None, **kwargs):
    """ Return gold labels for candidates as numpy array (all -1 or 1) and candidate index as
    label_index -> cand_index dict if requested
    """
    from snorkel.annotations import csr_LabelMatrix, load_matrix
    from snorkel.models import LabelKey, Label
    cids_query = supervision.get_cids_query(session, candidate_class, split)
    # if cids is not None:
    #     cids_query = cids_query.filter(candidate_class.subclass.id.in_(frozenset(cids)))
    key_group = candidate_class.index
    X = load_matrix(
        csr_LabelMatrix, LabelKey, Label, session,
        cids_query=cids_query, key_group=key_group, split=split, key_names=key_names, **kwargs
    )
    # Ensure that length of matrix matches candidate count
    ct = cids_query.count()
    assert X.shape[0] == ct, \
        'Labels matrix length ({}) does not match candidate count ({}) for class {}, split {}'\
        .format(X.shape[0], ct, candidate_class.field, split)
    return X
