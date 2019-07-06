import glob
import pathlib as pl
import numpy as np
import random
import os
from collections import defaultdict
from ignite.metrics import Accuracy, Loss, Precision, Recall
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.engine import Events, Engine, create_supervised_trainer, create_supervised_evaluator
from ignite.contrib.handlers.param_scheduler import LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tcre.modeling.metrics import get_f1_metric, PredictionAggregator
import torch.optim as optim
import torch.nn as nn
import torch
import logging
logger = logging.getLogger(__name__)


def set_seed(seed):
    # See:
    # - https://pytorch.org/docs/stable/notes/randomness.html
    # - https://www.kaggle.com/protan/lstm-cnn-torchtext-with-ignite
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def supervise(model, lr, decay, train_iter, val_iter,
              test_iter=None, model_dir=None, max_epochs=250, es_patience=25, lr_patience=25,
              log_iter_interval=10, log_epoch_interval=1, seed=0):
    set_seed(seed)

    if test_iter is None:
        test_iter = val_iter
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=lr_patience, threshold=0.01, verbose=True)
    criterion = nn.BCEWithLogitsLoss()

    if model_dir is not None:
        model_dir = pl.Path(model_dir) / 'checkpoints'

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
        metrics['f1'] = get_f1_metric(metrics['precision'], metrics['recall'])
        return metrics

    def get_evaluator():
        return create_supervised_evaluator(
            model, metrics=get_metrics(), prepare_batch=model.prepare, device=model.device,
            output_transform=lambda x, y, y_pred: (model.classify(model.transform(y_pred)), y)
        )

    train_evaluator = get_evaluator()
    val_evaluator = get_evaluator()
    test_evaluator = get_evaluator()

    def score_function(engine):
        return engine.state.metrics['f1']

    val_evaluator.add_event_handler(
        Events.EPOCH_COMPLETED,
        EarlyStopping(patience=es_patience, score_function=score_function, trainer=trainer)
    )
    if model_dir is not None:
        val_evaluator.add_event_handler(
            Events.EPOCH_COMPLETED,
            ModelCheckpoint(
                dirname=model_dir, filename_prefix='model', score_function=score_function, score_name='f1',
                create_dir=True, require_empty=True, n_saved=1
            ),
            {'model': model, 'optimizer': optimizer, 'scheduler': scheduler}
        )

    history = []

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        if engine.state.iteration % log_iter_interval == 0:
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
        history.append({k: v for k, v in record.items() if k != 'predictions'})
        if iteration % log_epoch_interval == 0:
            logger.info(
                '{type} Results - Epoch: {epoch}  Count: {ct} Loss: {loss:.2f} Accuracy: {accuracy:.3f} F1: {f1:.3f}'.format(
                    **record))
        return metrics

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        epoch, iteration = engine.state.epoch, engine.state.iteration
        log_results(train_evaluator, train_iter, 'training', epoch, iteration)
        log_results(test_evaluator, test_iter, 'test', epoch, iteration)
        metric = log_results(val_evaluator, val_iter, 'validation', epoch, iteration)['f1']
        scheduler.step(metric)

    trainer.run(train_iter, max_epochs=max_epochs)
    return history


def load_checkpoint(checkpoint_dir):
    files = glob.glob(str(pl.Path(checkpoint_dir) / '*.pth'))
    comps = defaultdict(lambda: [])
    for f in files:
        comps[pl.Path(f).name.split('_')[1]].append(torch.load(f))
    if any([len(v) > 1 for v in comps.values()]):
        raise ValueError(
            f'Found multiple checkpoint files for the same component '
            f'in dir "{checkpoint_dir}" (keys found = {comps.keys()})'
        )
    return {k: v[0] for k, v in comps.items()}
