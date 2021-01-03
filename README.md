# Sentiment Analysis via Span-Based Extraction and Classification for hotel reviews

This framework consists of three components:
- Multi-target extractor
- Polarity classifier
- Multilabel classifier

All components utilize [BERT](https://github.com/huggingface/pytorch-pretrained-BERT) as backbone network. The multi-target extractor aims to propose one or multiple opinion spans based on the probabilities of the start and end positions. The polarity classifier predicts the sentiment polarity using the span representation. The multilabel classifier classifies the resulting span on a fixed set of categories.

## Requirements
- Python 3.6
- [Pytorch 1.1](https://pytorch.org/)
- [Allennlp](https://allennlp.org/)

Download the uncased [BERT-Base](https://drive.google.com/file/d/13I0Gj7v8lYhW5Hwmp5kxm3CTlzWZuok2/view?usp=sharing) model and unzip it in the current directory.

## Running the scripts

You can use the VSCode configuration for running the scripts using the debugger. Please refer to each script to validate the required arguments.
To transform the input data, please refer to the examples on the data/absa folder and the notebooks for preprocessing.

## Mandatory arguments

  "--bert_config_file", "BERT_DIR/bert_config.json" - Config file for BERT
  "--vocab_file", "BERT_DIR/bert-base-uncased/vocab.txt", - vocabulary file for BERT
  "--output_dir", "out/cls/01", - folder where results will be dumped
  "--do_train", - if present, trains the model
  "--do_predict", - if present, duumps predictions on the out folder
  "--init_checkpoint", "out/extract/01/checkpoint.pth.tar", - if presents, uses a checkpoint for training the model. Otherwise, the training will be done from scratch
  "--train_batch_size", "32", - Size of the training batch
  "--max_seq_length", "60", - max sequence length for a review
  "--distilled", - if present, uses DistillBERT as backbone. Otherwise, BERT-base-uncased will be used
  "--train_file", "hotel_train_polarity.txt",
  "--predict_file",  "hotel_test_polarity.txt",
  "--data_dir", "data/absa"


  for span extraction: use model from bert for training from scratch
  "--init_checkpoint", "BERT_DIR/bert-base-uncased/pytorch_model.bin",

  For running on the terminal:

Run the following commands to set up environments:
```bash
export DATA_DIR=data/absa
export BERT_DIR=bert-base-uncased
```

## Multi-target extractor
Train the multi-target extractor:
```shell
python -m absa.run_extract_span \
  --vocab_file $BERT_DIR/vocab.txt \
  --bert_config_file $BERT_DIR/bert_config.json \
  --init_checkpoint $BERT_DIR/pytorch_model.bin \
  --do_train \
  --do_predict \
  --data_dir $DATA_DIR \
  --train_file rest_total_train.txt \
  --predict_file rest_total_test.txt \
  --train_batch_size 32 \
  --output_dir out/extract/01
```

## Polarity classifier
Train the polarity classifier:
```shell
python -m absa.run_cls_span \
  --vocab_file $BERT_DIR/vocab.txt \
  --bert_config_file $BERT_DIR/bert_config.json \
  --init_checkpoint $BERT_DIR/pytorch_model.bin \
  --do_train \
  --do_predict \
  --data_dir $DATA_DIR \
  --train_file rest_total_train.txt \
  --predict_file rest_total_test.txt \
  --train_batch_size 32 \
  --output_dir out/cls/01
```

## Results

```
The predicted results will be saved into a file called `predictions.json` in the `output_dir`:
```bash
cat out/cls/01/predictions.json
```

The test performance is shown in a file called `performance.txt` in the `output_dir`:
```bash
cat out/cls/01/performance.txt
```
Which should produce an output like this:
```bash
pipeline, step: 210, P: 0.6991, R: 0.7156, F1: 0.7073 (common: 1638.0, retrieved: 2343.0, relevant: 2289.0)
```


## Acknowledgements

This project relies on the code of the following paper for the span extractor:

<i> [Open-Domain Targeted Sentiment Analysis via Span-Based Extraction and Classification](https://arxiv.org/abs/1906.03820). Minghao Hu, Yuxing Peng, Zhen Huang, Dongsheng Li, Yiwei Lv. ACL 2019.</i>

We sincerely thank Hu, Minghao and Peng, Yuxing and Huang, Zhen and Li, Dongsheng and Lv and Yiwei for releasing the project SpanABSA

```
@inproceedings{hu2019open,
  title={Open-Domain Targeted Sentiment Analysis via Span-Based Extraction and Classification},
  author={Hu, Minghao and Peng, Yuxing and Huang, Zhen and Li, Dongsheng and Lv, Yiwei},
  booktitle={Proceedings of ACL},
  year={2019}
}
```
