import pandas as pd
import numpy as np
import tempfile
import shutil
import os
import os.path as osp

DEFAULT_CONFIG = dict(max_seq_length=128, learning_rate=2e-5, num_train_epochs=3.0)
DEFAULT_BERT_PATH = osp.join('/lab/data/scibert', 'scibert_scivocab_uncased')

CMD_TRAIN = """
python run_dataset.py --task_name tcre --do_train \
--do_lower_case --data_dir {data_dir} \
--model_type bert --model_name_or_path {model_name_or_path} \
--max_seq_length {max_seq_length} --learning_rate {learning_rate} --num_train_epochs {num_train_epochs} \
--overwrite_output_dir \
--output_dir {output_dir}"""

CMD_EVAL = """
python run_dataset.py --task_name tcre --do_eval \
--do_lower_case --data_dir {data_dir} \
--model_type bert --model_name_or_path {model_name_or_path} \
--max_seq_length {max_seq_length} --learning_rate {learning_rate} --num_train_epochs {num_train_epochs} \
--output_dir {output_dir}"""


def run_transformer_training(cands, config=DEFAULT_CONFIG, data_dir=None, bert_path=DEFAULT_BERT_PATH):
    if data_dir is None:
        data_dir = tempfile.mkdtemp()
    input_dir = osp.join(data_dir, 'input')
    output_dir = osp.join(data_dir, 'output')
    log_file = osp.join(data_dir, 'log.txt')
    if osp.exists(input_dir):
        shutil.rmtree(input_dir)
    os.makedirs(input_dir, exist_ok=True)
    if osp.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    if osp.exists(log_file):
        os.remove(log_file)

    def get_df(cands, split):
        df = pd.DataFrame([
            dict(guid=c.id, text=features.get_scibert_text(c), split=split, label=str(features.get_label(c)))
            for c in cands
        ])
        assert df['label'].isin(['0', '1']).all()
        return df

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
        cmd += " >> {}".format(log_file)
        print("Command = \n{}\n\n".format(cmd))
        rc = os.system(cmd)
        print(rc)
        if rc != 0:
            raise ValueError('Command failed with return code {}:\n{}'.format(rc, cmd))

    cmd = CMD_TRAIN.format(data_dir=osp.join(input_dir, 'train'), output_dir=output_dir, model_name_or_path=bert_path,
                           **config)
    run_cmd(cmd)

    cmd = CMD_EVAL.format(data_dir=osp.join(input_dir, 'val'), output_dir=osp.join(output_dir, 'val'),
                          model_name_or_path=output_dir, **config)
    run_cmd(cmd)

    cmd = CMD_EVAL.format(data_dir=osp.join(input_dir, 'test'), output_dir=osp.join(output_dir, 'test'),
                          model_name_or_path=output_dir, **config)
    run_cmd(cmd)

    def get_score_df(split):
        df = pd.read_json(osp.join(output_dir, split, 'scores.json'))
        return df.iloc[0].rename('value').rename_axis('metric').reset_index().assign(split=split)

    return pd.concat([get_score_df(s) for s in ['val', 'test']])


def run_transformer_modeling(session, candidate_class, config=DEFAULT_CONFIG, data_dir=None,
                             bert_path=DEFAULT_BERT_PATH, clear=True):
    df_cand, df_dist = sampling.get_modeling_splits(
        session, target_split_map={'dev': 'train', 'val': 'val', 'test': 'test'})
    df_cand = df_cand[df_cand['task'] == candidate_class.field]

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

    cands = {s: get_cands(s) for s in ['train', 'val', 'test']}
    run_transformer_training(cands, config=config, data_dir=data_dir, bert_path=bert_path)