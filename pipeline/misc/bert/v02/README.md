### Execution

```
python run_dataset.py --task_name imdb --do_train --do_eval \
--do_lower_case --data_dir $IMDB_DIR/ \
--model_type bert --model_name_or_path bert-base-uncased \
--max_seq_length 128 --learning_rate 2e-5 --num_train_epochs 3.0 \
--output_dir /tmp/imdb_output/
```

### Compatibility Notes


New configs for pytorch-transformers: https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json

```
{
  "attention_probs_dropout_prob": 0.1,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "max_position_embeddings": 512,
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "type_vocab_size": 2,
  "vocab_size": 30522
}
```

Config with scibert named bert_config.json that must be renamed to config.json:

```
{
  "attention_probs_dropout_prob": 0.1,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "max_position_embeddings": 512,
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "type_vocab_size": 2,
  "vocab_size": 31090
}
```

### Pytorch-transformers Notes

Default cache location is ```/root/.cache/torch/pytorch_transformers``` and the logic for setting it is in [file_utils.py](https://github.com/huggingface/pytorch-transformers/blob/ed717635ff5c2bd5dfa8fd0266f309e314a3e44f/pytorch_transformers/file_utils.py#L42)

The need to rename config file is also mentioned in https://pypi.org/project/spacy-pytorch-transformers/ (with no other modifications, so presumably this is all that's needed to make pytorch-transformers work with scibert exports)


### Fine Tuning Notes

BERT authors recommend hyperparameter settings for fine tuning: https://mccormickml.com/2019/07/22/BERT-fine-tuning/