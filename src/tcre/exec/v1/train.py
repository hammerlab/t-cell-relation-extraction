import os
import os.path as osp
import pathlib
import pandas as pd
import numpy as np
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
from tcre.modeling.metrics import get_f1_metric
from tcre.exec.v1.model import RNN, DIST_PAD_VAL
from torchtext import data as txd
from torchtext.vocab import Vocab
from torchtext.data import BucketIterator
from ignite.contrib.handlers.param_scheduler import LRScheduler
from ignite.engine import Events, Engine, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, Precision, Recall
from ignite.handlers import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import torch.optim as optim
import click
import logging
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


def train(cands, config):
    SEQ_LEN = 128

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

    fields = {
        'text': txd.Field(sequential=True, lower=False, fix_length=SEQ_LEN, include_lengths=True),
        'label': txd.Field(sequential=False),
        'e0_dist': txd.Field(sequential=True, use_vocab=False, pad_token=DIST_PAD_VAL, fix_length=SEQ_LEN),
        'e1_dist': txd.Field(sequential=True, use_vocab=False, pad_token=DIST_PAD_VAL, fix_length=SEQ_LEN),
        'id': txd.Field(sequential=False, use_vocab=False)
    }

    logger.info(f'Sample feature records:\n')
    for r in df.head(5).to_dict(orient='records'):
        logger.info(pprint.pformat(r, width=128, compact=True, indent=6))
    logger.info('Label distribution:\n%s',
                pd.concat([df['label'].value_counts(), df['label'].value_counts(normalize=True)], axis=1))

    logger.info('Initializing batch iterators')
    # Build dataset for training
    ds = tcre_data.DataFrameDataset(df, fields)
    for k, f in fields.items():
        if k in ['label', 'text']:
            fields[k].build_vocab(ds)

    # If using w2v, overwrite text field vocabulary
    if config['wrd_embedding_type'].startswith('w2v'):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message='.*')
            from gensim.models import KeyedVectors
            logger.info(f"Loading w2v model with vocab limit {config['vocab_limit']}")
            w2v = KeyedVectors.load_word2vec_format(W2V_MODEL_01, binary=True, limit=config['vocab_limit'])
            fields['text'].vocab = W2VVocab(w2v)
    # Otherwise, ensure that embedding type is valid
    else:
        if config['wrd_embedding_type'] != 'denovo':
            raise ValueError(f"Word embedding type {config['wrd_embedding_type']} not valid")

    # Split the dataset and build iterators
    ds_train, ds_val = ds.split(split_ratio=.7, stratified=True, strata_field='label')
    train_iter, val_iter = BucketIterator.splits(
        (ds_train, ds_val),
        batch_sizes=(32, 32),
        sort_key=lambda x: len(x.text),
        sort=True,
        sort_within_batch=True,
        repeat=False
    )
    logger.info(f'Split dataset into num examples train = {len(ds_train)}, validation = {len(ds_val)}')

    model_args = dict(MODEL_SIZES[config['model_size']])
    model_args.update(dict(dropout=0, bidirectional=False, cell_type='LSTM', device=config['device']))
    if config['wrd_embedding_type'] != 'denovo':
        model_args['wrd_embed_dim'] = None
    if config['wrd_embedding_type'] in ['w2v_trained']:
        model_args['train_wrd_embed'] = True
    if not config['use_positions']:
        model_args['pos_embed_dim'] = 0
    config['model_args'] = model_args

    logger.info(f"Building model with arguments: {config['model_args']}")
    model = RNN(ds.fields, **config['model_args'])
    lr, decay = config['learning_rate'], config['weight_decay']

    logger.info('Running optimization')
    history = supervise(model, lr, decay, train_iter, val_iter)
    return model_args, history


def supervise(model, lr, decay, train_iter, val_iter):

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=25, threshold=0.01, verbose=True)
    criterion = nn.BCEWithLogitsLoss()

    trainer = create_supervised_trainer(
        model, optimizer, criterion,
        device=model.device, prepare_batch=model.prepare
    )

    def get_metrics():
        metrics = {
            'accuracy': Accuracy(),
            'precision': Precision(average=False),
            'recall': Recall(average=False),
            'loss': Loss(criterion)
        }
        metrics['f1'] = get_f1_metric(metrics['recall'], metrics['precision'])
        return metrics

    def get_evaluator():
        return create_supervised_evaluator(
            model, metrics=get_metrics(), prepare_batch=model.prepare, device=model.device,
            output_transform=lambda x, y, y_pred: (model.classify(model.transform(y_pred)), y)
        )

    train_evaluator = get_evaluator()
    val_evaluator = get_evaluator()

    def score_function(engine):
        return engine.state.metrics['f1']

    handler = EarlyStopping(patience=25, score_function=score_function, trainer=trainer)
    val_evaluator.add_event_handler(Events.EPOCH_COMPLETED, handler)

    history = []

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        if engine.state.iteration % 10 == 0:
            logger.info("Epoch[{}] Iteration[{}] Loss: {:.4f} LR: {}".format(
                engine.state.epoch, engine.state.iteration,
                engine.state.output, optimizer.param_groups[0]['lr']
            ))

    def log_results(engine, iterator, dataset_type, epoch, iteration):
        engine.run(iterator)
        metrics = dict(engine.state.metrics)
        metrics['ct'] = len(iterator.dataset)
        metrics['lr'] = optimizer.param_groups[0]['lr']
        record = {**metrics, **{'type': dataset_type.title(), 'epoch': epoch}}
        history.append(record)
        if iteration % 1 == 0:
            logger.info(
                '{type} Results - Epoch: {epoch}  Count: {ct} Loss: {loss:.2f} Accuracy: {accuracy:.3f} F1: {f1:.3f}'.format(
                    **record))
        return metrics

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        epoch, iteration = engine.state.epoch, engine.state.iteration
        _ = log_results(train_evaluator, train_iter, 'training', epoch, iteration)['loss']
        metric = log_results(val_evaluator, val_iter, 'validation', epoch, iteration)['f1']
        scheduler.step(metric)

    trainer.run(train_iter, max_epochs=250)
    return history


@click.command()
@click.option('--relation-class', default=None, help='Candidate type class ("inducing_cytokine")')
@click.option('--marker-list', default='mult_01', help='Marker list name ("doub_01", "sngl_01")')
@click.option('--use-secondary/--no-secondary', default=True, help='Use secondary markers')
@click.option('--use-swaps/--no-swaps', default=True, help='Use swaps for primary entity text')
@click.option('--use-lower/--no-lower', default=False, help='Whether or not to use only lower case tokens')
@click.option('--use-positions/--no-positions', default=False, help='Whether or not to use positional features')
@click.option('--wrd-embedding-type', default='w2v_frozen', help='One of "w2v_frozen", "w2v_trained", or "denovo"')
@click.option('--vocab-limit', default=50000, help='For pre-trained vectors, max vocab size')
@click.option('--model-size', default='S', help='One of "S", "M", "L"')
@click.option('--weight-decay', default=0.0, help='Weight decay for training')
@click.option('--dropout', default=0.0, help='Dropout rate')
@click.option('--learning-rate', default=.005, help='Learning rate')
@click.option('--device', default='cuda', help='Device to use for training')
@click.option('--log-level', default='info', help='Logging level')
@click.option('--output-dir', default=None, help='Output directory (nothing saved if omitted)')
def run(relation_class, marker_list, use_secondary, use_swaps, use_lower, use_positions,
        wrd_embedding_type, vocab_limit, model_size,
        weight_decay, dropout, learning_rate, device, log_level, output_dir):
    logging.basicConfig(level=log_level.upper())
    session = SnorkelSession()
    classes = get_candidate_classes()
    classes = {classes[c].field: classes[c] for c in classes}
    candidate_class = classes[relation_class]
    cands = session.query(candidate_class.subclass) \
        .filter(candidate_class.subclass.split == SPLIT_DEV).all()
    logger.info(f'Found {len(cands)} candidates')

    config = get_training_config(
        relation_class=relation_class,
        entity_types=candidate_class.entity_types,
        marker_list=marker_list,
        use_secondary=use_secondary,
        use_swaps=use_swaps, use_lower=use_lower, use_positions=use_positions,
        wrd_embedding_type=wrd_embedding_type,
        model_size=model_size, learning_rate=learning_rate,
        weight_decay=weight_decay, dropout=dropout,
        vocab_limit=vocab_limit,
        device=device
    )
    logger.info(f'Training config:\n{pprint.pformat(config, compact=True, indent=6, width=128)}')

    model_args, history = train(cands, config)

    if output_dir:
        logger.info(f'Saving results to path {output_dir}')
        output_dir = pathlib.Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        import json
        with (output_dir / 'history.json').open('w') as f:
            json.dump(history, f)
        with (output_dir / 'config.json').open('w') as f:
            json.dump(config, f)

    logger.info('Training complete')


if __name__ == '__main__':
    run()
