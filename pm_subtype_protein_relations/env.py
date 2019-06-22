import os
import os.path as osp
from dotenv import load_dotenv
if osp.exists('env.sh'):
    load_dotenv('env.sh')
else:
    load_dotenv('../env.sh')
DATA_DIR=os.environ['DATA_DIR']
REPO_DATA_DIR=os.environ['REPO_DATA_DIR']
META_DATA_DIR=os.environ['META_DATA_DIR']
SUPERVISION_DATA_DIR=os.environ['SUPERVISION_DATA_DIR']
MODEL_DATA_DIR=os.environ['MODEL_DATA_DIR']
IMPORT_DATA_DIR_01=os.environ['IMPORT_DATA_DIR_01']
IMPORT_DATA_DIR_02=os.environ['IMPORT_DATA_DIR_02']
SEED=39283