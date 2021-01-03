# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run BERT on SemEval."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import pickle
import json
import argparse
import collections
import pandas as pd
from sklearn import metrics
from transformers import DistilBertTokenizer
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import bert.tokenization as tokenization
from bert.modeling import BertConfig
from bert.sentiment_modeling import DistillBertForMultilabelClassification
from transformers import BertTokenizer

from squad.squad_evaluate import exact_match_score
from absa.utils import read_absa_data, convert_absa_data, convert_absa_data_polarity, convert_examples_to_features, convert_examples_to_features_polarities, \
    RawFinalResult, RawFinalResultPolarity, wrapped_get_final_text, id_to_label, MultilabelClsDataset
from absa.run_base import copy_optimizer_params_to_model, set_optimizer_params_grad, prepare_optimizer, post_process_loss, bert_load_state_dict

try:
    import xml.etree.ElementTree as ET, getopt, logging, sys, random, re, copy
    from xml.sax.saxutils import escape
except:
    sys.exit('Some package is missing... Perhaps <re>?')

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

# Creating the dataset and dataloader for the neural network
def read_train_data(args, tokenizer, logger):
    df = pd.read_csv(args.data_dir + '/' + args.train_file)
    df['list'] = df[df.columns[0:-1]].values.tolist()
    new_df = df[['match', 'list']].copy()

    train_size = 0.8
    train_dataset=new_df.sample(frac=train_size,random_state=200)
    test_dataset=new_df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    print("FULL Dataset: {}".format(new_df.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))

    training_set = MultilabelClsDataset(train_dataset, tokenizer, args.max_seq_length)
    testing_set = MultilabelClsDataset(test_dataset, tokenizer, args.max_seq_length)
    train_params = {'batch_size': args.train_batch_size,
            'shuffle': True,
            'num_workers': 0
            }

    test_params = {'batch_size': args.train_batch_size,
                    'shuffle': True,
                    'num_workers': 0
                    }

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)

    return training_loader, testing_loader

def validation(args, model, device, testing_loader, write_pred=False, predictions_path=''):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            # bert
            #outputs = model(ids, mask, token_type_ids)
            # distilledbert
            outputs = model(ids, mask)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
            
            if write_pred and predictions_path:
                f = open(predictions_path, "a")
                print("pred: {}, gold: {}"
                    .format(outputs, targets), file=f)
                print(" ", file=f)
                f.close()

    return fin_outputs, fin_targets

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--bert_config_file", default=None, type=str, required=True,
                        help="The config json file corresponding to the pre-trained BERT model. "
                             "This specifies the model architecture.")
    parser.add_argument("--vocab_file", default=None, type=str, required=True,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--debug", default=False, action='store_true', help="Whether to run in debug mode.")
    parser.add_argument("--data_dir", default='data/semeval_14', type=str, help="SemEval data dir")
    parser.add_argument("--train_file", default=None, type=str, help="SemEval xml for training")
    parser.add_argument("--predict_file", default=None, type=str, help="SemEval csv for prediction")
    parser.add_argument("--extraction_file", default=None, type=str, help="pkl file for extraction")
    parser.add_argument("--init_checkpoint", default=None, type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model).")
    parser.add_argument("--do_lower_case", default=True, action='store_true',
                        help="Whether to lower case the input text. Should be True for uncased "
                             "models and False for cased models.")
    parser.add_argument("--max_seq_length", default=96, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--do_train", default=False, action='store_true', help="Whether to run training.")
    parser.add_argument("--do_predict", default=False, action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_pipeline", default=False, action='store_true', help="Whether to run pipeline on the dev set.")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size", default=32, type=int, help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=8, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% "
                             "of training.")
    parser.add_argument("--save_proportion", default=0.5, type=float,
                        help="Proportion of steps to save models for. E.g., 0.5 = 50% "
                             "of training.")
    parser.add_argument("--verbose_logging", default=False, action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--optimize_on_cpu',
                        default=False,
                        action='store_true',
                        help="Whether to perform optimization and keep the optimizer averages on CPU")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=128,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')

    args = parser.parse_args()
    if not args.do_train and not args.do_predict and not args.do_pipeline:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if args.do_train and not args.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")
    if args.do_predict and not args.predict_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_file` must be specified.")
    save_path = os.path.join(args.output_dir, 'checkpoint_mlcls.pth.tar')
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        if args.fp16:
            logger.info("16-bits training currently not supported in distributed training")
            args.fp16 = False # (see https://github.com/pytorch/pytorch/pull/13496)
    logger.info("torch_version: {} device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        torch.__version__, device, n_gpu, bool(args.local_rank != -1), args.fp16))

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    logger.info('output_dir: {}'.format(args.output_dir))
    save_path = os.path.join(args.output_dir, 'checkpoint_ml_cls.pth.tar')
    log_path = os.path.join(args.output_dir, 'performance_ml_cls.txt')
    network_path = os.path.join(args.output_dir, 'network_ml_cls.txt')
    parameter_path = os.path.join(args.output_dir, 'parameter_ml_cls.txt')
    predictions_path = os.path.join(args.output_dir, 'predictions_ml_cls.txt')

    logger.info("***** Preparing model *****")
    model = DistillBertForMultilabelClassification()
    model.to(device)

    if args.init_checkpoint is not None and not os.path.isfile(save_path):
    
        checkpoint = torch.load(save_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer = torch.optim.Adam()
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])        
        logger.info("Loading model from pretrained checkpoint: {}".format(save_path))
    else:
        optimizer = torch.optim.Adam(params =  model.parameters(), lr=args.learning_rate)

    logger.info("***** Preparing data *****")
    training_loader, testing_loader = read_train_data(args, tokenizer, logger)

    if args.do_train:
        logger.info("***** Preparing training *****")

        #bert
        def loss_fn(outputs, targets):
            return torch.nn.BCEWithLogitsLoss()(outputs, targets)
        def calcuate_accu(big_idx, targets):
            n_correct = (big_idx==targets).sum().item()
            return n_correct
        def train(epoch):
            tr_loss = 0
            n_correct = 0
            nb_tr_steps = 0
            nb_tr_examples = 0
            #bert
            # loss = loss_fn(outputs, targets)
            #distilled
            loss_function = torch.nn.CrossEntropyLoss()
            model.train()
            for _,data in enumerate(training_loader, 0):
                ids = data['ids'].to(device, dtype = torch.long)
                mask = data['mask'].to(device, dtype = torch.long)
                #bert 
                #token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
                targets = data['targets'].to(device, dtype = torch.float)

                #outputs = model(ids, mask, token_type_ids)
                outputs = model(ids, mask)
                loss = loss_fn(outputs, targets)
                tr_loss += loss.item()
                big_val, big_idx = torch.max(outputs.data, dim=1)
                # n_correct += calcuate_accu(big_idx, targets)

                nb_tr_steps += 1
                nb_tr_examples+=targets.size(0)

                optimizer.zero_grad()

                if _%5000==0:
                    loss_step = tr_loss/nb_tr_steps
                    accu_step = (n_correct*100)/nb_tr_examples 
                    print(f"Training Loss per 5000 steps: {loss_step}")
                    print(f"Training Accuracy per 5000 steps: {accu_step}")

                #print(f'Epoch: {epoch + 1}, Loss:  {loss.item()}')
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f'The Total Accuracy for Epoch {epoch}: {(n_correct*100)/nb_tr_examples}')
            epoch_loss = tr_loss/nb_tr_steps
            epoch_accu = (n_correct*100)/nb_tr_examples
            print(f"Training Loss Epoch: {epoch_loss}")
            print(f"Training Accuracy Epoch: {epoch_accu}")
            return

        for epoch in range(args.num_train_epochs):
            train(epoch)

        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'step': 0
        }, save_path)

    logger.info("***** Running validation *****")
    f = open(log_path, "a")
    for epoch in range(3):
        outputs, targets = validation(args, model, device, testing_loader)
        outputs = np.array(outputs) >= 0.5
        from sklearn import metrics
        accuracy = metrics.accuracy_score(targets, outputs)
        recall = metrics.recall_score(targets, outputs,average='samples')
        f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
        f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
        f1 = metrics.f1_score(targets, outputs, average='samples')
        print("epoch: {}, P: {:.4f}, R: {:.4f}, F1: {:.4f}"
              .format(epoch + 1, accuracy, recall,f1), file=f)
        print(" ", file=f)
    f.close()
    print("epoch: {}, P: {:.4f}, R: {:.4f}, F1: {:.4f}"
              .format(epoch + 1, accuracy, recall,f1))
    if args.do_predict:
        logger.info("***** Running prediction *****")

        # restore from best checkpoint
        if save_path and os.path.isfile(save_path) and args.do_train:
            checkpoint = torch.load(save_path)
            model.load_state_dict(checkpoint['model'])
            logger.info("Loading model from finetuned checkpoint: '{}'"
                        .format(save_path))

        model.eval()
        results =  validation(args, model, device, testing_loader, write_pred=True, predictions_path=predictions_path)


    

if __name__=='__main__':
    main()
