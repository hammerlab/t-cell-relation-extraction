import pandas as pd
import numpy as np
import tempfile
import shutil
import os
import os.path as osp
from snorkel.models import Candidate
from tcre.modeling import sampling
from tcre.modeling import features
import tcre

DEFAULT_CONFIG = dict(max_seq_length=128, learning_rate=2e-5, num_train_epochs=3.0, per_gpu_train_batch_size=8)
DEFAULT_BERT_PATH = osp.join('/lab', 'data', 'scibert', 'scibert_scivocab_uncased')
SCRIPT_PATH = osp.join(tcre.src_dir, 'exec', 'v1', 'bert', 'run_dataset.py')

CMD_TRAIN = """
python {script_path} --task_name tcre --do_train --do_eval \
--do_lower_case --data_dir {data_dir} \
--model_type bert --model_name_or_path {model_name_or_path} \
--max_seq_length {max_seq_length} --learning_rate {learning_rate} --num_train_epochs {num_train_epochs} \
--per_gpu_train_batch_size {per_gpu_train_batch_size} \
--overwrite_output_dir \
--scores_filename {scores_filename} \
--output_dir {output_dir}"""

CMD_EVAL = """
python {script_path} --task_name tcre --do_eval \
--do_lower_case --data_dir {data_dir} \
--model_type bert --model_name_or_path {model_name_or_path} \
--max_seq_length {max_seq_length} --learning_rate {learning_rate} --num_train_epochs {num_train_epochs} \
--per_gpu_train_batch_size {per_gpu_train_batch_size} \
--scores_filename {scores_filename} \
--output_dir {output_dir}"""


def run_transformer_script(cands, config=DEFAULT_CONFIG, data_dir=None, bert_path=DEFAULT_BERT_PATH,
                             print_commands=False):

    # Use temp data dir if one not provided
    if data_dir is None:
        data_dir = tempfile.mkdtemp()

    # Initialize directories:
    # input_dir: To contain csv files with examples for both training and evaluation (organized in subfolders by split)
    # output_dir: To contain modeling checkpoints as well as evaluation score files (as json)
    input_dir = osp.join(data_dir, 'input')
    output_dir = osp.join(data_dir, 'output')
    log_file = osp.join(data_dir, 'log.txt')

    # Delete anything that already exists
    if osp.exists(input_dir):
        shutil.rmtree(input_dir)
    os.makedirs(input_dir, exist_ok=True)
    if osp.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    if osp.exists(log_file):
        os.remove(log_file)

    # Extract records necessary for TcreProcessor in tcre.exec.v1.bert.utils_dataset
    def get_df(cands, split):
        df = pd.DataFrame([
            dict(guid=c.id, text=features.get_scibert_text(c), split=split, label=str(features.get_label(c)))
            for c in cands
        ])
        assert df['label'].isin(['0', '1']).all()
        return df

    # Initialize training and evaluation dataset for each split
    for split in ['train', 'val', 'test']:
        in_dir = osp.join(input_dir, split)
        os.makedirs(in_dir, exist_ok=True)
        # BERT script expects only "train" or "dev" splits, with the latter being for evaluation
        df = pd.concat([
            get_df(cands['train'], 'train'),
            get_df(cands[split], 'dev')
        ])
        df.to_csv(osp.join(in_dir, 'data.csv'), index=False)

    def run_cmd(cmd):
        cmd += " >> {} 2>&1".format(log_file)
        if print_commands:
            print("Command = \n{}\n\n".format(cmd))
        rc = os.system(cmd)
        if rc != 0:
            raise ValueError('Command failed with return code {}:\n{}'.format(rc, cmd))

    # Run model training (evaluate on training data as well)
    cmd = CMD_TRAIN.format(
        script_path=SCRIPT_PATH, data_dir=osp.join(input_dir, 'train'),
        output_dir=output_dir, model_name_or_path=bert_path,
        scores_filename='scores_train', **config
    )
    run_cmd(cmd)

    # Evaluate on validation data
    cmd = CMD_EVAL.format(
        script_path=SCRIPT_PATH, data_dir=osp.join(input_dir, 'val'),
        output_dir=output_dir, model_name_or_path=output_dir,
        scores_filename='scores_val', **config
    )
    run_cmd(cmd)

    # Evaluate on test data
    cmd = CMD_EVAL.format(
        script_path=SCRIPT_PATH, data_dir=osp.join(input_dir, 'test'),
        output_dir=output_dir, model_name_or_path=output_dir,
        scores_filename='scores_test', **config
    )
    run_cmd(cmd)

    def get_score_df(split):
        df = pd.read_json(osp.join(output_dir, 'scores_{}.json'.format(split)), lines=True)
        return df.iloc[0].rename('value').rename_axis('metric').reset_index().assign(split=split)

    return pd.concat([get_score_df(s) for s in ['train', 'val', 'test']])


def run_transformer_modeling(session, candidate_class, config=DEFAULT_CONFIG, data_dir=None,
                             bert_path=DEFAULT_BERT_PATH, clear=True, **kwargs):
    # Pull candidates using common sampling logic (to ensure consistency with other modeling strategies)
    df_cand, df_dist = sampling.get_modeling_splits(
        session, target_split_map={'dev': 'train', 'val': 'val', 'test': 'test'})
    df_cand = df_cand[df_cand['task'] == candidate_class.field]

    # Prepare output directory
    if clear:
        if osp.exists(data_dir):
            shutil.rmtree(data_dir)
    os.makedirs(data_dir, exist_ok=True)

    def get_cands(split):
        # Convert to native int rather than numpy int (query will return nothing otherwise)
        cands = [int(v) for v in df_cand[df_cand['split'] == split]['id'].unique()]
        cands = session.query(Candidate).filter(Candidate.id.in_(frozenset(cands))).all()
        if len(cands) == 0:
            raise ValueError('No candidates found for split "{}"'.format(split))
        return cands

    # Materialize candidates from id and group by split
    cands = {s: get_cands(s) for s in ['train', 'val', 'test']}
    return run_transformer_script(cands, config=config, data_dir=data_dir, bert_path=bert_path, **kwargs)
