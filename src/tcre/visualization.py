import pandas as pd

DEFAULT_COLORS = ['red', 'blue', 'green']


def candidate_html(c, colors=None):
    """Get HTML visualization of a candidate"""
    if colors is None:
        colors = {}
    sent = c.get_parent()
    words = [w for w in sent.words]
    types = c.get_parent().entity_types
    cids = c.get_parent().entity_cids
    for ctx in c.get_contexts():
        w1, w2 = ctx.get_word_range()
        cid = cids[w1]
        color = colors.get(types[w1], 'gray')
        words[w1] = f'<div style="color:{color};display:inline" title="{cid}">{words[w1]}'
        words[w2] = f'{words[w2]}</div>'
    return ' '.join(words)


def candidate_df(candidates, colors=None):

    if colors is None:
        # Get entity types for candidate and map to colors arbitrarily
        types = list(set([t for c in candidates for t in c.__class__.__argnames__]))
        colors = {
            t: DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
            for i, t in enumerate(types)
        }

    df = pd.DataFrame([
        dict(
            id=c.id,
            type=c.type,
            split=c.split,
            e1=c.get_contexts()[0].get_span(),
            e2=c.get_contexts()[1].get_span(),
            text=candidate_html(c, colors=colors)
        )
        for c in candidates
    ])
    return df[['id', 'type', 'split', 'e1', 'e2', 'text']]


def candidate_html_table(candidates, colors=None):
    return candidate_df(candidates, colors=colors).to_html(escape=False)
