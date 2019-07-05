""" Environment variables common to all tasks (typically directory locations)

This is useful for importing all variables often used at a CLI into
any python script namespace.  E.g.:

from tcre.env import *
print(DATA_DIR)
"""
import os
import sys
import os.path as osp
from dotenv import dotenv_values

TCRE_SEED = int(os.getenv('TCRE_SEED', 3832))

# Root environment variables that should always be set externally
DEFAULT_ENV_VARS = ['DATA_DIR', 'REPO_DATA_DIR', 'REPO_DIR']


def _get_env_vars(default_vars=DEFAULT_ENV_VARS):
    pkg_dir = osp.abspath(osp.dirname(__file__))
    path = osp.normpath(osp.join(pkg_dir, '..', '..', 'env.sh'))
    if not osp.exists(path):
        raise ValueError(f'Environment variable script not found (path = {path})')
    return {**dotenv_values(path), **{v: os.getenv(v) for v in default_vars}}


def _set_env_vars(env_vars):
    module = sys.modules[__name__]
    for k, v in env_vars.items():
        setattr(module, k, v)


# Read environment variables from bash-friendly script and set them as globals
# on this module (this is useful for syncing python environment with auxiliary bash script environment)
_set_env_vars(_get_env_vars())

