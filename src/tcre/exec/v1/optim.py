"""Hyperparameter search utilitis"""
import os
import os.path as osp
import pandas as pd
import numpy as np
from tcre.env import *
from tcre.exec.v1 import cli_client
from skopt import forest_minimize
from skopt.callbacks import CheckpointSaver, TimerCallback
import logging
logger = logging.getLogger(__name__)

DEFAULT_DEVICE = '"cuda:1"'
CMD_FORMAT = "{cmd} > {log_dir}/log.txt 2>&1"


class CheckpointCallback(CheckpointSaver):

    def __init__(self, checkpoint_path, interval=1, **dump_options):
        super().__init__(checkpoint_path, **dump_options)
        self.i = 0
        self.interval = interval

    def __call__(self, res):
        self.i += 1
        if self.i % self.interval == 0:
            logger.info(f'Saving checkpoint to {self.checkpoint_path}')
            super().__call__(res)


class ProgressLogger(object):

    def __init__(self, n, interval=1):
        self.n = n
        self.i = 0
        self.interval = interval

    def __call__(self, res, *args, **kwargs):
        self.i += 1
        if self.i % self.interval == 0:
            logger.info(f'Completed iteration {self.i} of {self.n} (score = {res.func_vals[-1]})')


class ObjectiveFunction(object):

    def __init__(self, fn):
        self.fn = fn
        self.scores = []

    def __call__(self, x):
        res = self.fn(x)
        self.scores.append(res)
        return -res[('f1', 'validation')]


def to_dict(x, space):
    return dict(zip([s.name for s in space], x))


class TaskParameterOptimizer(object):

    def __init__(self, task, space, output_dir, minimizer=forest_minimize, device=DEFAULT_DEVICE, client_args=None):
        self.task = task
        self.space = space
        self.minimizer = minimizer
        self.output_dir = osp.join(output_dir, task)
        self.device = device

        self.client = cli_client.get_default_client()
        self.client_args = client_args or {}

        self.dirs = {}
        for d in ['checkpoints', 'data', 'log', 'splits']:
            self.dirs[d] = osp.join(self.output_dir, d)
            if not osp.exists(self.dirs[d]):
                os.makedirs(self.dirs[d])

    def get_splits_file(self):
        return osp.join(self.dirs['splits'], 'splits.json')

    def get_checkpoints_file(self):
        return osp.join(self.dirs['checkpoints'], 'checkpoint.pkl')

    def write_splits(self, splits):
        splits_file = self.get_splits_file()
        if not osp.exists(osp.dirname(splits_file)):
            os.makedirs(osp.dirname(splits_file))
        splits.to_json(splits_file, orient='index')
        return splits_file

    def cli_args(self):
        return {
            **dict(relation_class=self.task, device=self.device, output_dir=self.dirs['data']),
            **self.client_args.get('cli', {})
        }

    def train_args(self, x):
        splits_file = self.get_splits_file()
        train_opts = to_dict(x, self.space)
        return {
            **dict(splits_file=splits_file, use_checkpoints=False, save_keys='"history"'),
            **self.client_args.get('train', {}),
            **train_opts
        }

    def format(self, cmd):
        return CMD_FORMAT.format(cmd=cmd, log_dir=self.dirs['log'])

    def get_cmd(self, x):
        cli_args = self.cli_args()
        train_args = self.train_args(x)
        cmd = self.client.cmd(cli=cli_args, train=train_args)
        return self.format(cmd)

    def evaluate(self, x):
        # Generate and run CLI command for sampled point in search space (non zero return codes raise automatically)
        cmd = self.get_cmd(x)
        self.client.execute(cmd)

        df = pd.read_json(osp.join(self.dirs['data'], 'history.json'))
        df['type'] = df['type'].str.lower()

        # Pivot to epoch in index and columns like (metric, type) (e.g. ('f1', 'validation'))
        df = df.set_index(['epoch', 'type']).unstack()

        # Return all F1 @ best F1 on validation
        idx = np.argmax(df[('f1', 'validation')].values)
        return {**df.iloc[idx].to_dict(), **{('epoch', ''): df.index.values[idx]}}

    def run(self, n_iterations, progress_interval=1, checkpoint_interval=10, **kwargs):
        timer = TimerCallback()
        plogger = ProgressLogger(n_iterations, interval=progress_interval)

        saver = CheckpointCallback(self.get_checkpoints_file(), interval=checkpoint_interval)
        callbacks = {'saver': saver, 'plogger': plogger, 'timer': timer}

        obj_fn = ObjectiveFunction(self.evaluate)

        logger.info(f'Beginning parameter search with max iterations {n_iterations}')
        res = self.minimizer(
            obj_fn, self.space,
            n_calls=n_iterations,
            callback=list(callbacks.values()),
            random_state=TCRE_SEED,
            **kwargs
        )

        return res, obj_fn.scores, callbacks
