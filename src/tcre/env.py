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


def _get_env_vars():
    pkg_dir = osp.abspath(osp.dirname(__file__))
    return dotenv_values(osp.normpath(osp.join(pkg_dir, '..', '..', 'env.sh')))


def _set_env_vars(env_vars):
    module = sys.modules[__name__]
    for k, v in env_vars.items():
        setattr(module, k, v)


# Read environment variables from bash-friendly script and set them as globals
# on this module (this is useful for syncing python environment with auxiliary bash script environment)
_set_env_vars(_get_env_vars())

