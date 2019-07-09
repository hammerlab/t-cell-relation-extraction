import numpy as np


def _get_primary_entity_indices(r):
    # Expect tag sequence like "O, E:primary:immune_cell_type, , O, E:primary:cytokine, E:primary:cytokine, O, O"
    # Find primary entity tags noting that they may be repeated if they span multiple tokens
    tags = list(set([t for t in r['tags'] if t.startswith('E:primary')]))
    if len(tags) != 2:
        raise ValueError(f'Failed to find two primary tags for record:\n{r}')

    # Find first index of each distinct tag
    itag = [r['tags'].index(t) for t in tags]
    return itag


def _label_by_dist(r):
    # Return probability based on abs distance (prob goes up when distance goes down)
    itag = _get_primary_entity_indices(r)
    dist = abs(itag[1] - itag[0])
    dist = np.clip(dist, 0, 32) / 32.
    logit = 5*(dist - .5)
    prob = 1. / (1 + np.exp(-logit))
    return 1 - prob


def _label_by_secondary_marking(r):
    ptag = _get_primary_entity_indices(r)

    # Get indices of secondary entities (if any)
    stags = [i for i, t in enumerate(r['tags']) if t.startswith('E:secondary')]

    # Return 0 if a secondary exists between primaries, otherwise 1
    for i in stags:
        if ptag[0] <= i <= ptag[1]:
            return 0.
    return 1.


def _rs():
    return np.random.RandomState(1)


def _get_random_labels(n):
    return _rs().choice([0., 1.], size=n, replace=True)


LABEL_SIM_FNS = {
    'position-based': _label_by_dist,
    'secondary-marking': _label_by_secondary_marking
}


def get_simulated_labels(df, strategy):
    if strategy == 'random':
        return _get_random_labels(len(df))
    if strategy.startswith('random-'):
        # Use the strategy suffix to get permuted labels (to preserve class balance)
        fn = strategy.replace('random-', '')
        if fn in LABEL_SIM_FNS:
            y = df.apply(LABEL_SIM_FNS[fn], axis=1).values
            _rs().shuffle(y)
            return y
    if strategy in LABEL_SIM_FNS:
        return df.apply(LABEL_SIM_FNS[strategy], axis=1).values
    raise ValueError(f'Simulation strategy "{strategy}" invalid')


