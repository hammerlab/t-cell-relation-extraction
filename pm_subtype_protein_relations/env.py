import os
from dotenv import load_dotenv
load_dotenv('env.sh')
DATA_DIR=os.environ['DATA_DIR']
REPO_DATA_DIR=os.environ['REPO_DATA_DIR']
META_DATA_DIR=os.environ['META_DATA_DIR']
SEED=39283