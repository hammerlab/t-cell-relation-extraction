from tcre import lib
from ptkn.protein_tokenization import ProteinTokenizer

DEFAULT_SYN_BLACKLIST = ["ifi", "dif", "esp", "tc1", "til"]


def load_protein_tokenizer(syn_blacklist=DEFAULT_SYN_BLACKLIST):
    """ Create tokenizer using combined transcription factor, cytokine, and surface protein vocabulary

    Args:
        syn_blacklist: List of synonyms to be ignored globally (i.e. across entity types)
    Returns:
        ProteinTokenizer instance
    """
    df_pr = lib.get_entity_meta_data(lib.SURFACE_PROTEINS)
    df_tf = lib.get_entity_meta_data(lib.TRANSCRIPTION_FACTORS)
    pm_tf = df_tf.set_index('id')[['lbl']].to_dict(orient='index')
    df_ck = lib.get_entity_meta_data(lib.CYTOKINES)
    pm_ck = df_ck.set_index('id')[['lbl']].to_dict(orient='index')
    vocab = {
        **{r['syn']: (r['label'], r['extid'], r['pref_lbl'], r['pref_id'], 'pr')
            for i, r in df_pr.iterrows()},
        **{r['sym']: (r['lbl'], r['id'], pm_tf.get(r['prefid'], {}).get('lbl'), r['prefid'], 'tf')
            for i, r in df_tf.iterrows()},
        **{r['sym']: (r['lbl'], r['id'], pm_ck.get(r['prefid'], {}).get('lbl'), r['prefid'], 'ck')
            for i, r in df_ck.iterrows()}
    }
    for syn in syn_blacklist:
        if syn in vocab:
            del vocab[syn]

    return ProteinTokenizer(vocab)
