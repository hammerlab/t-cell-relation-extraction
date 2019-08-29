import numpy as np
import pandas as pd
from tcre.env import *
from tcre.modeling import features
from tcre.supervision import SPLIT_MAPI, SPLIT_MAP
from snorkel.learning.utils import LabelBalancer
from sklearn.model_selection import train_test_split

DEFAULT_TARGET_SPLIT_MAP = {'dev': 'train', 'val': 'val', 'test': 'test'}
DEFAULT_LABEL_BALANCE = .5


def get_modeling_splits(session, target_split_map=DEFAULT_TARGET_SPLIT_MAP,
                        max_training_examples=3000, balance=DEFAULT_LABEL_BALANCE):

    # Get list of splits to be mapped into (possibly different) target splits
    source_splits = [SPLIT_MAPI[k] for k in target_split_map.keys()]

    # Get all candidates and associated gold labels
    df_cand = features.get_candidate_labels(session, splits=source_splits).rename(columns={'type': 'task'})
    df_cand['split'] = df_cand['split'].map(SPLIT_MAP).map(target_split_map)
    assert df_cand['split'].notnull().all()

    def rand_state():
        return np.random.RandomState(TCRE_SEED)

    res = {}
    for task, g in df_cand.groupby('task'):

        def downsample(df):
            split = df['split'].iloc[0]

            # Return candidates to predict as is
            if split == 'predict':
                return df['id'].unique()

            # Ignore any examples with labels exactly equal to .5 (can happen for examples with all abstaining LFs)
            df = df[(df['label'] - .5).abs() > 1e-6]

            # Re-balance candidates based on label, if requested
            if balance is not None:
                idx = LabelBalancer(df['label'].values).get_train_idxs(rand_state=rand_state(), rebalance=balance)
                if split == 'train' and len(idx) > max_training_examples:
                    idx = rand_state().choice(idx, size=max_training_examples, replace=False)
                df = df.iloc[idx]

            cts = (df['label'] > .5).value_counts()
            max_p = (df['label'] > .5).value_counts(normalize=True).max()
            # Raise an error if balance is more than 10% off target
            if balance is not None and abs(max_p - balance) > .1:
                raise AssertionError(
                    f'Split for task {task} was expected to have balance {balance} but has label distribution {cts}')
            assert df['id'].is_unique
            return df

        res[task] = {k: downsample(gs) for k, gs in g.groupby('split')}

    # Build data frame containing all candidates (plus associated labels) for each task and split
    df_cand = pd.concat([
        res[task][split].assign(task=task, split=split)
        for task in res
        for split in res[task]
    ])

    # Calculate label distribution by task and split
    df_dist = pd.concat([
        (
            df_cand
            .assign(label=lambda df: (df['label'] > .5).astype(int))
            .groupby(['task', 'split'])['label'].value_counts(normalize=normalize).unstack()
            .rename(columns=lambda c: ('percent' if normalize else 'count', c))
            .pipe(lambda df: df * 100 if normalize else df)
            .round(1)
        )
        for normalize in [True, False]
    ], axis=1)
    df_dist.columns = pd.MultiIndex.from_tuples(df_dist.columns)
    df_dist.columns.names = ('statistic', 'label')

    return df_cand, df_dist


def _get_stratified_split(splits, idx, values, proportions):
    # Return immediately if there are fewer than 2 proportions use in splits
    if len(proportions) <= 1:
        splits.append(idx)
        return splits

    # Determine proportion of items to extract for this split and associated
    # indexes for that proportion of the items (and add to final result)
    p = proportions[0]
    idx_split = train_test_split(idx, stratify=values[idx], train_size=p, random_state=TCRE_SEED)[0]
    splits.append(idx_split)

    # Recurse with this split removed from indexes and proportions rescaled
    # to reflect the fraction of items in the subset equivalent to the desired
    # fraction in the original set
    idx = np.setdiff1d(idx, idx_split)
    proportions = (1. / (1 - p)) * proportions[1:]
    return _get_stratified_split(splits, idx, values, proportions)


def get_stratified_split(values, proportions):
    """Split values into arbitrary sets with target size while using stratification

    Args:
        values: Labels to use for stratification
        proportions: Sequence containing desired proportions of resulting splits
    Returns:
        List of numpy arrays with length equal to length of proportions and values
        equivalent to indexes associated with each split (the number of these indexes
        should roughly account for the desired portion of samples and each split
        should be comprised of roughly equal label frequencies)
    Example:

    ```
    # Binary labels array
    values = (np.arange(50) > 10).astype(int)
    # Split into 3 groups where first is largest at 60% of all elements (and next two are smaller)
    proportions = [.6, .3, .1]
    [np.unique(y[s], return_counts=True) for s in get_stratified_split(values, proportions)]
    >> [
    (array([0, 1]), array([ 7, 23])), # 30 items (60% of 50), ~80/20 class balance
    (array([0, 1]), array([ 3, 12])), # 15 items (30% of 50), 80/20 class balance
    (array([0, 1]), array([1, 4]))    # 5 items (10% of 50), 80/20 class balance
    ]
    ```
    """
    values, proportions = np.asarray(values), np.asarray(proportions)
    if values.ndim != 1:
        raise ValueError('Values must be 1D array')
    if proportions.ndim != 1:
        raise ValueError('Proportions must be 1D array')
    if not np.isclose(np.sum(proportions), 1):
        raise ValueError('Proportions must sum to 1')
    idx = np.arange(len(values))
    splits = []
    return _get_stratified_split(splits, idx, values, proportions)