"""Hyperparameter search utilitis"""
from skopt.callbacks import CheckpointSaver, TimerCallback
from skopt import forest_minimize
from tcre.env import *
import logging
import os
import os.path as osp
import pandas as pd
import numpy as np
from tcre.exec.v1 import cli
logger = logging.getLogger(__name__)

DEFAULT_DEVICE = "cuda:1"
CLI_PATH = cli.__file__
CMD_FORMAT = """python {script_file} \\
--relation-class={relation_class} --device={device} --output-dir={output_dir} \\
train \\
--splits-file={splits_file} --save-keys="history" --use-checkpoints=False \\
{args} > {log_dir}/log.txt 2>&1
"""


class CheckpointCallback(CheckpointSaver):

    def __init__(self, checkpoint_path, interval=1, **dump_options):
        super().__init__(checkpoint_path, **dump_options)
        self.i = 0
        self.interval = interval

    def __call__(self, res):
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
        return -res['Validation']


def to_dict(x, space):
    return dict(zip([s.name for s in space], x))


def to_args(x, space):
    args = to_dict(x, space)
    return ' '.join(['--{}={}'.format(k.replace('_', '-'), v) for k, v in args.items()])


class TaskParameterOptimizer(object):

    def __init__(self, task, space, output_dir, minimizer=forest_minimize, device=DEFAULT_DEVICE):
        self.task = task
        self.space = space
        self.minimizer = minimizer
        self.output_dir = osp.join(output_dir, task)
        self.device = device

        self.dirs = {}
        for d in ['checkpoints', 'data', 'log', 'splits']:
            self.dirs[d] = osp.join(self.output_dir, d)
            if not osp.exists(self.dirs[d]):
                os.makedirs(self.dirs[d])

    def get_splits_file(self):
        return osp.join(self.dirs['splits'], f'hopt_{self.task}.json')

    def get_checkpoints_file(self):
        return osp.join(self.dirs['checkpoints'], 'checkpoint.pkl')

    def write_splits(self, splits):
        splits_file = self.get_splits_file()
        if not osp.exists(osp.dirname(splits_file)):
            os.makedirs(osp.dirname(splits_file))
        splits.to_json(splits_file, orient='index')
        return splits_file

    def get_cmd(self, x):
        splits_file = self.get_splits_file()
        args = to_args(x, self.space)
        cmd = CMD_FORMAT.format(
            script_file=CLI_PATH, relation_class=self.task,
            device=self.device, output_dir=self.dirs['data'],
            splits_file=splits_file,
            log_dir=self.dirs['log'], args=args
        )
        return cmd

    def evaluate(self, x):
        cmd = self.get_cmd(x)
        rc = os.system(cmd)
        if rc != 0:
            raise ValueError(f'Return code {rc} (!=0) for command: {cmd}')
        df = pd.read_json(osp.join(self.dirs['data'], 'history.json'))

        # Pivot to split in columns (Train, Validation, Test)
        df = df.set_index(['epoch', 'type'])['f1'].unstack()

        # Return all F1 @ best F1 on validation
        idx = np.argmax(df['Validation'].values)
        return df.iloc[idx].to_dict()

    def run(self, n_iterations, progress_interval=1, checkpoint_interval=10, n_random_starts=10):
        timer = TimerCallback()
        plogger = ProgressLogger(n_iterations, interval=progress_interval)

        saver = CheckpointCallback(self.get_checkpoints_file(), interval=checkpoint_interval)
        callbacks = {'saver': saver, 'plogger': plogger, 'timer': timer}

        obj_fn = ObjectiveFunction(self.evaluate)

        logger.info(f'Beginning parameter search with max iterations {n_iterations}')
        res = self.minimizer(
            obj_fn, self.space,
            n_calls=n_iterations,
            n_random_starts=n_random_starts,
            callback=list(callbacks.values()),
            random_state=TCRE_SEED
        )

        return res, obj_fn.scores, callbacks
