import os
import os.path as osp
import pathlib
import pandas as pd
import numpy as np
import json
import shutil
import pprint
import warnings
from collections import defaultdict, Counter
from snorkel import SnorkelSession
from snorkel.models import Candidate, GoldLabel
from tcre.env import *
from tcre.supervision import get_candidate_classes, ENT_TYP_CT_L, ENT_TYP_CK_L, ENT_TYP_TF_L, SPLIT_DEV
from tcre.modeling import utils
from tcre.modeling import features
from tcre.modeling import data as tcre_data
from tcre.modeling.vocab import W2VVocab
from tcre.modeling.training import supervise
from tcre.exec.v1.model import RERNN, DIST_PAD_VAL
from torchtext import data as txd
from torchtext.vocab import Vocab
from torchtext.data import BucketIterator
import click
import logging
import dill
logger = logging.getLogger(__name__)

DEFAULT_SWAPS = {
    ENT_TYP_CT_L: 'CELL',
    ENT_TYP_CK_L: 'CYTOKINE',
    ENT_TYP_TF_L: 'TF'
}
MARKER_LISTS = {
    'doub_01': [('<<', '>>'), ('[[', ']]'), ('##', '##'), ('@@', '@@')],
    'sngl_01': [('<', '>'), ('[', ']'), ('#', '#'), ('@', '@')],
    'mult_01': [('< #', '# >'), ('< @', '@ >'), ('| #', '# |'), ('| @', '@')]
}
MODEL_SIZES = {
    'S': {'hidden_dim': 5, 'wrd_embed_dim': 10, 'pos_embed_dim': 3},
    'M': {'hidden_dim': 10, 'wrd_embed_dim': 30, 'pos_embed_dim': 8},
    'L': {'hidden_dim': 20, 'wrd_embed_dim': 50, 'pos_embed_dim': 10},
    'XL': {'hidden_dim': 30, 'wrd_embed_dim': 100, 'pos_embed_dim': 10},
}


def get_training_config(**kwargs):
    markers = MARKER_LISTS[kwargs['marker_list']]
    if not kwargs['use_secondary']:
        markers[2] = None
        markers[3] = None
    typ0, typ1 = kwargs['entity_types']
    markers = {
        'primary': {typ0: list(markers[0]), typ1: list(markers[1])},
        'secondary': {typ0: list(markers[2] or []), typ1: list(markers[3] or [])}
    }

    res = dict(kwargs)
    res['label'] = ':'.join([f'{k}={v}' for k, v in kwargs.items() if k != 'entity_types'])
    res['markers'] = markers
    res['swaps'] = DEFAULT_SWAPS if kwargs['use_swaps'] else None
    return res


def _splits(splits_file):
    # Expect splits written using something like:
    # pd.Series({'train': [1,2,3], 'test': [3,4]}).to_json('/tmp/splits.json', orient='index')
    # --> {"train":[1,2,3],"test":[3,4]}
    return pd.read_json(splits_file, typ='series', orient='index').to_dict()


def _cands(candidate_class, splits):
    session = SnorkelSession()
    cids = list(set([cid for s, cids in splits.items() for cid in cids]))
    return session.query(candidate_class.subclass) \
        .filter(candidate_class.subclass.id.in_(cids)).all()


def get_model(fields, config):
    model_args = dict(MODEL_SIZES[config['model_size']])
    model_args.update(dict(dropout=0, bidirectional=False, cell_type='LSTM', device=config['device']))
    if config['wrd_embedding_type'] != 'denovo':
        model_args['wrd_embed_dim'] = None
    if config['wrd_embedding_type'] in ['w2v_trained']:
        model_args['train_wrd_embed'] = True
    if not config['use_positions']:
        model_args['pos_embed_dim'] = 0
    model = RERNN(fields, **model_args)
    return model, model_args


def _train(cands, splits, config):
    """Train a single model

    Args:
        cands: List of all candidates
        splits: Dict with keys "train", "val", "test" each having a list of integer candidate ids
        config: Training configuration
    """
    SEQ_LEN = 128

    if set(splits.keys()) != {'train', 'val', 'test'}:
        raise ValueError(f'Splits must contain keys "train", "val", "test", got keys {splits.keys()}')

    logger.info('Collecting features')
    # Filter for candidates used in marking
    if config['use_secondary']:
        predicate = lambda e: e['type'] in config['entity_types']
    else:
        predicate = lambda e: e['is_candidate'] and e['type'] in config['entity_types']

    # Get marked token sequences with distance features (as sequences)
    records = features.candidates_to_records(cands, entity_predicate=predicate)
    df = features.get_record_features(
        records, markers=config['markers'], swaps=config['swaps'],
        lower=config['use_lower'],
        subtokenizer=lambda t: t.split(), assert_unique=False
    )
    df = df.rename(columns={'tokens': 'text'})
    if not df['label'].between(0, 1).all():
        ex = df['label'][~df['label'].between(0, 1)].unique()
        raise AssertionError(f"Found label values outside [0, 1]; Examples: {ex[:10]}")

    logger.info(f'Sample feature records:\n')
    for r in df.head(5).to_dict(orient='records'):
        logger.info(pprint.pformat(r, width=128, compact=True, indent=6))
    logger.info('Label distribution:\n%s',
                pd.concat([df['label'].value_counts(), df['label'].value_counts(normalize=True)], axis=1))

    fields = {
        'text': txd.Field(sequential=True, lower=False, fix_length=SEQ_LEN, include_lengths=True),
        'label': txd.Field(sequential=False, use_vocab=False),
        'e0_dist': txd.Field(sequential=True, use_vocab=False, pad_token=DIST_PAD_VAL, fix_length=SEQ_LEN),
        'e1_dist': txd.Field(sequential=True, use_vocab=False, pad_token=DIST_PAD_VAL, fix_length=SEQ_LEN),
        'id': txd.Field(sequential=False, use_vocab=False)
    }

    # Build dataset for each split (one candidate may exist in multiple splits)
    logger.info('Initializing batch iterators')
    dfi = df.set_index('id', drop=False)
    datasets = pd.DataFrame([
        dfi.loc[cid].append(pd.Series({'split': s}))
        for s, cids in splits.items() for cid in cids
    ]).reset_index(drop=True)
    datasets = {
        k: tcre_data.DataFrameDataset(g.drop('split', axis=1), fields)
        for k, g in datasets.groupby('split')
    }

    # If using w2v, set text field vocabulary on all datasets
    if config['wrd_embedding_type'].startswith('w2v'):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message='.*')
            from gensim.models import KeyedVectors
            logger.info(f"Loading w2v model with vocab limit {config['vocab_limit']}")
            w2v = KeyedVectors.load_word2vec_format(W2V_MODEL_01, binary=True, limit=config['vocab_limit'])
            fields['text'].vocab = W2VVocab(w2v)
    # Build vocab on training dataset text field ONLY and assign to others
    elif config['wrd_embedding_type'] == 'denovo':
        fields['text'].build_vocab(datasets['train'])
    # Otherwise, embedding type is not valid
    else:
        raise ValueError(f"Word embedding type {config['wrd_embedding_type']} not valid")

    # Split the dataset and build iterators
    train_iter, val_iter, test_iter = BucketIterator.splits(
        tuple([datasets[k] for k in ['train', 'val', 'test']]),
        batch_sizes=[config['batch_size']]*3,
        sort_key=lambda x: len(x.text),
        sort=True,
        sort_within_batch=True,
        repeat=False
    )
    logger.info('Split datasets with sizes %s', {k: len(ds) for k, ds in datasets.items()})

    model, model_args = get_model(fields, config)
    config['model_args'] = model_args
    logger.info(f"Built model with arguments: {config['model_args']}")
    lr, decay = config['learning_rate'], config['weight_decay']

    logger.info('Running optimization')
    history, predictions = supervise(
        model, lr, decay, train_iter, val_iter, test_iter=test_iter,
        model_dir=config['model_dir'] if config['use_checkpoints'] else None
    )
    return history, predictions, fields


@click.command()
@click.option('--relation-class', default=None, help='Candidate type class ("inducing_cytokine")')
@click.option('--splits-file', default=None, help='Path to json file containing candidate ids keyed '
                                                  'by split name ("train", "val" required, "test" optional)')
@click.option('--marker-list', default='mult_01', help='Marker list name ("doub_01", "sngl_01")')
@click.option('--use-secondary/--no-secondary', default=True, help='Use secondary markers')
@click.option('--use-swaps/--no-swaps', default=True, help='Use swaps for primary entity text')
@click.option('--use-lower/--no-lower', default=False, help='Whether or not to use only lower case tokens')
@click.option('--use-positions/--no-positions', default=False, help='Whether or not to use positional features')
@click.option('--use-checkpoints/--no-checkpoints', default=False, help='Save checkpoint for best model')
@click.option('--save-predictions/--no-predictions', default=False, help='Save test set predictions')
@click.option('--wrd-embedding-type', default='w2v_frozen', help='One of "w2v_frozen", "w2v_trained", or "denovo"')
@click.option('--vocab-limit', default=50000, help='For pre-trained vectors, max vocab size')
@click.option('--model-size', default='S', help='One of "S", "M", "L"')
@click.option('--weight-decay', default=0.0, help='Weight decay for training')
@click.option('--dropout', default=0.0, help='Dropout rate')
@click.option('--learning-rate', default=.005, help='Learning rate')
@click.option('--device', default='cuda', help='Device to use for training')
@click.option('--batch-size', default=32, help='Batch size used in training and prediction')
@click.option('--log-level', default='info', help='Logging level')
@click.option('--output-dir', default=None, help='Output directory (nothing saved if omitted)')
def train(relation_class, splits_file, marker_list, use_secondary, use_swaps, use_lower, use_positions, use_checkpoints,
        save_predictions, wrd_embedding_type, vocab_limit, model_size,
        weight_decay, dropout, learning_rate, device, batch_size, log_level, output_dir):
    logging.basicConfig(level=log_level.upper())

    classes = get_candidate_classes()
    classes = {classes[c].field: classes[c] for c in classes}
    candidate_class = classes[relation_class]

    logger.info(f'Gathering candidates for splits at "{splits_file}"')
    splits = _splits(splits_file)
    splits_ct = {s: len(cids) for s, cids in splits.items()}
    cands = _cands(candidate_class, splits)
    logger.info(f'Found {len(cands)} candidates (split sizes = {splits_ct}')

    if output_dir is not None:
        logger.info(f'Initializing {output_dir}')
        output_dir = pathlib.Path(output_dir)
        if output_dir.exists():
            # Validate that path is at least 2 levels away from root
            # dir before deleting it
            if len(output_dir.resolve().parts) <= 2:
                raise AssertionError(
                    f'Path "{output_dir}" is too close to root dir to delete '
                    f'(are you sure this is correct)?')
            shutil.rmtree(output_dir)
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

    config = get_training_config(
        relation_class=relation_class,
        splits_file=splits_file,
        entity_types=candidate_class.entity_types,
        marker_list=marker_list,
        use_secondary=use_secondary, use_swaps=use_swaps, use_lower=use_lower,
        use_positions=use_positions, use_checkpoints=use_checkpoints,
        wrd_embedding_type=wrd_embedding_type,
        model_size=model_size, learning_rate=learning_rate,
        weight_decay=weight_decay, dropout=dropout,
        vocab_limit=vocab_limit,
        device=device, batch_size=batch_size,
        model_dir=str(output_dir) if output_dir is not None else None
    )
    logger.info(f'Training config:\n{pprint.pformat(config, compact=True, indent=6, width=128)}')

    history, predictions, fields = _train(cands, splits, config)

    if output_dir is not None:
        with (output_dir / 'fields.pkl').open('wb') as f:
            dill.dump(fields, f)
        with (output_dir / 'history.json').open('w') as f:
            json.dump(history, f)
        with (output_dir / 'config.json').open('w') as f:
            json.dump(config, f)
        if save_predictions:
            predictions.to_json(output_dir / 'predictions.json', orient='records')

    logger.info('Training complete')


if __name__ == '__main__':
    train()
