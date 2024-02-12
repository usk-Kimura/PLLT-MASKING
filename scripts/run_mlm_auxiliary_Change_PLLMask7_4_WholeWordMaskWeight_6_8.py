### THIS FILE IS COPIED FROM THE HUGGINGFACE REPOSITORY FOR CONVENIENCE.

# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
import pdb
from typing import Dict, List, Tuple
import gc
import torch.backends.cudnn as cudnn

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    WEIGHTS_NAME,
    AutoTokenizer,
    AdamW,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)
from minicons import scorer
import spacy
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import random

from .endtask_aware import TartanModel, add_tartan_args
from .endtask_auxtasks import get_auxtask_files
from .alpha_generator import add_config_args
from .utils import *

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)



'''
Original
'''
nlp = spacy.load("en_core_web_sm")

pretrained_model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
mlm_model = scorer.MaskedLMScorer(pretrained_model_name, 'cuda')



def tokenize_with_offsets(text, LENGTH,tokenizer):

    inputs = tokenizer.encode_plus(text, return_offsets_mapping=True, add_special_tokens=True, truncation=True, max_length=LENGTH)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"])
    # tokens_ = []
    # for t in tokens:
    #     if t not in ["<pad>", "</s>", "<s>"]:
    #         tokens_.append(t)
    offsets = inputs["offset_mapping"]
    return tokens, offsets

def calculate_word_level_scores_forWeightWord(tokens, scores): #実装終了
    word_scores, ws = [], []
    words, w = [],[]

    token_to_word = []
    t_to_w = []
    
    subword_cnt = 0
    #token は単語かサブワードかもしれない
    for i, token in enumerate(tokens):
        # print(token)
        if (not(token.startswith('Ġ')) or token.startswith('##')) and i != 0 and token not in [",", "."]: #サブワード！
            subword_cnt += 1 
            if subword_cnt == 1:
                word_scores += ws[:-1]
                words += w[:-1]#サブワード前までのスコアと単語を格納
                ws = ws[-1:]
                w = w[-1:]

                token_to_word+=t_to_w[:-1]
                t_to_w = t_to_w[-1:]

            ws.append(scores[i])
            w.append(token)

            t_to_w.append(i)
            
            # print("if1",w)
            # print("if1",t_to_w)
            

            if len(tokens)-1 == i and  token not in [",", "."]:#new
                word_scores += ws
                words += w
                token_to_word += t_to_w
                

        elif subword_cnt == 0:
            # print("if2",token)
            ws.append(scores[i])
            w.append(token) #単語かサブワードかこの時点では分からない
            t_to_w.append(i)
            
            if i == len(tokens) - 1:
                word_scores += ws
                words += w
                token_to_word+=t_to_w
                
        else: #前回のループでサブワード判定で，今回は別単語なので初期化
            # print("if3",token)
            word_scores.append(sum(ws)/len(ws)) #word Mask 用　ドメインが異なると単語分割され，その単語を認識したいので，平均値を算出したくない
            words.append("".join(w))
            token_to_word.append(t_to_w)
            
            ws, w = [],[]
            subword_cnt = 0
            t_to_w = []
            
            ws.append(scores[i])
            w.append(token)
            t_to_w.append(i)
            
            if i == len(tokens) - 1:
                word_scores += ws
                words += w
                token_to_word += t_to_w

    
    return words, word_scores, token_to_word


def anything_masking(tokenizer: PreTrainedTokenizer, token_to_word, labels_, labels, masked_indices, indices_replaced, inputs,mask_probability=0.2):
    #word,phrase_masking (Our Method)
    indices_replaced_tokentoword = [False]#<s>
    for n, ttw in enumerate(token_to_word):
        if type(ttw) == list:
            for t in ttw:
                indices_replaced_tokentoword.append(True)
        else:
            indices_replaced_tokentoword.append(False)
    indices_replaced_tokentoword.append(False)#<\s>
    indices_replaced_tokentoword = torch.tensor([indices_replaced_tokentoword])

    cnt = 0
    our_method_act = False
    
    # Maskが単語内にあれば単語全体をマスクする
    for n, (i_r, i_r_t, label) in enumerate(zip(indices_replaced[0], indices_replaced_tokentoword[0], labels_[0])):
        # マスク位置が単語内か
        if i_r_t == False:
            cnt = 0#単語内のトークンでなければ初期化

        if i_r_t and cnt == 0:#最初に実行されるのは単語の頭トークン
            masktoword_size = 0
            now = n
            
            for i in range(n, n+10):
                if indices_replaced_tokentoword[0][i] == True:
                    masktoword_size += 1
                else:
                    break
            cnt += 1
            
            if True in indices_replaced[0][now:now + masktoword_size]:#最初のマスクが単語内にあるか？
                our_method_act = random.random() > 1 - mask_probability #True #提案手法実行
            else:
                our_method_act = False
            # print(n, "Tokenの最初．実行するか", our_method_act,"何回？",masktoword_size)
            # print(inputs)
            # print(labels)
        
        if i_r_t and our_method_act:#Wordマスキング True  
            # print("実行する")
            labels[0][n] = label
            masked_indices[0][n] = True
            indices_replaced[0][n] = True
            inputs[0][n] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
            # print(inputs)
            # print(labels)
        
        else:
            pass
            # print("実行しない")


    return labels, masked_indices, indices_replaced, inputs


def masking_probability(score, allscore):
    # 例: スコアが平均より低い場合、確率を高める v5
    prob = score/allscore   # 0.15はデフォルトのマスキング確率


    return prob

'''
    Core MLM functionality １文書ずつしか対応していない
'''

def input_pad(inputs,labels,tokenizer,LENGTH):
    if len(inputs[0]) < LENGTH:
        # for i in range(LENGTH-len(inputs[0])):
            # inputs[0].append(tokenizer.pad_token_id)
            # labels[0].append(-100)
        # print("input_pad")
        # print(inputs[0].size())
        # print(labels[0].size())
        # print(torch.tensor([tokenizer.pad_token_id]))
        # print(torch.tensor([tokenizer.pad_token_id] * (LENGTH-len(inputs[0]))))
        # print(torch.tensor([-100] * (LENGTH-len(inputs[0]))))
        # inputs[0] = torch.cat((inputs[0], torch.tensor([tokenizer.pad_token_id] * (LENGTH-len(inputs[0])))),dim=0)
        # labels[0] = torch.cat((labels[0], torch.tensor([-100] * (LENGTH-len(inputs[0])))),dim=0)
        inputs = inputs.tolist()
        inputs[0] += [tokenizer.pad_token_id] * (LENGTH-len(inputs[0]))
        labels = labels.tolist()
        labels[0] += [-100] * (LENGTH-len(labels[0]))
        # print("INP","LAB",len(inputs[0]),len(labels[0]))

    return torch.tensor(inputs), torch.tensor(labels)

def mask_tokens_PRP(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, task_info, task_name, token_to_word_to_phrase_, token_to_phrase,token_to_word, mlm_list, wordscores, args) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    mlm_mask = task_info[task_name]["mlm_mask"]# default 0.8
    mlm_random = task_info[task_name]["mlm_random"]# default 0.5
    mlm_probability = task_info[task_name]["mlm_probability"]#args.word_mask_probability[task_index]
    word_mask_probability = task_info[task_name]["word_mask_probability"]#args.word_mask_probability[task_index]
    phrase_mask_probability = task_info[task_name]["phrase_mask_probability"]#args.phrase_mask_probability[task_index]

    token_to_word_to_phrase = token_to_word_to_phrase_.copy()

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked"
            " language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()
    labels_ = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training
    # (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)

    # print("inputs")
    # print(inputs.size())
    # print(inputs)


    m_score = []
    for i in mlm_list[0]:
        m_score.append(list(i)[1] * -1)


    
    m_score.insert(0, 0)#<s>
    m_score.append(0)

    # print("m_score",len(m_score))
    # print(m_score)

    # probability_matrix = torch.full(labels.shape, mlm_probability) #change Our Method
    c = 0
    # print("token_to_phrase")
    # print(token_to_phrase)

    token_to_word_to_phrase.insert(0, 0)#<s>
    token_to_word_to_phrase.append(0)

    token_to_phrase.insert(0, 0)#<s>
    token_to_phrase.append(0)
    token_to_word.insert(0, 0)#<s>
    token_to_word.append(0)

    wordscores.insert(0, 0)#<s>
    wordscores.append(0)


    wordscores_ = []
    for i in wordscores:
        wordscores_.append(i * -1)

    # min_score = min(wordscores_)
    # max_score = max(wordscores_)
    # mean_score = sum(wordscores_)/len(wordscores_)
    allscore = sum(wordscores)

    probability_matrix = []
    for score in wordscores_:
        probability_matrix.append(masking_probability(score, allscore))

    for l in range(len(labels.tolist()[0])-len(probability_matrix)):
        probability_matrix.append(0)


    probability_matrix = torch.tensor([probability_matrix])


    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)





    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()

    masked_indices_ = []
    for token, mask in zip(token_to_word, masked_indices[0]):
        if type(token)==list:
            masks = [mask] * len(token)
            masked_indices_.extend(masks)
        else:
            masked_indices_.append(mask)


    masked_indices_ = torch.tensor([masked_indices_])


    labels[~masked_indices] = -100  # We only compute loss on masked tokens
    
    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, mlm_mask)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # #Word_masking (Our Method)#どうやって複数のタスクを作る？ v6.5
    # labels, masked_indices, indices_replaced, inputs = anything_masking(tokenizer, token_to_word, labels_, labels, masked_indices, indices_replaced, inputs, mask_probability=word_mask_probability)
    # #Phrase_masking (Our Method)
    # labels, masked_indices, indices_replaced, inputs = anything_masking(tokenizer, token_to_phrase, labels_, labels, masked_indices, indices_replaced, inputs, mask_probability=phrase_mask_probability)
    #外れ値のフレーズみマスク
    
    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, mlm_random)).bool() & masked_indices & ~indices_replaced

    # # #Word_random (Our Method) v6.5
    # _, _, indices_random, _ = anything_masking(tokenizer, token_to_word, labels_, labels, masked_indices, indices_random, inputs, mask_probability=word_mask_probability)
    # #Phrase_random (Our Method)
    # _, _, indices_random, _ = anything_masking(tokenizer, token_to_phrase, labels_, labels, masked_indices, indices_random, inputs, mask_probability=phrase_mask_probability)
  


    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels

def contains_list(lst):
    # isinstanceを使ってリスト内の各要素がリストかどうかをチェック
    for element in lst:
        if isinstance(element, list):
            return True
    return False

# def mask_tokens_PRP_ph(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, token_to_word, args) -> Tuple[torch.Tensor, torch.Tensor]:
#     """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
#     word_mask_probability=args.word_mask_probability
#     phrase_mask_probability=args.phrase_mask_probability

#     if tokenizer.mask_token is None:
#         raise ValueError(
#             "This tokenizer does not have a mask token which is necessary for masked"
#             " language modeling. Remove the --mlm flag if you want to use this tokenizer."
#         )

#     labels = inputs.clone()
#     labels_ = inputs.clone()
#     # We sample a few tokens in each sequence for masked-LM training
#     # (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
#     probability_matrix = torch.full(labels.shape, args.mlm_probability)
#     special_tokens_mask = [
#         tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
#     ]
#     probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
#     if tokenizer._pad_token is not None:
#         padding_mask = labels.eq(tokenizer.pad_token_id)
#         probability_matrix.masked_fill_(padding_mask, value=0.0)
#     masked_indices = torch.bernoulli(probability_matrix).bool()
#     labels[~masked_indices] = -100  # We only compute loss on masked tokens
    
#     # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
#     indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
#     inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

#     #Phrase_masking (Our Method)
#     labels, masked_indices, indices_replaced, inputs = anything_masking(tokenizer, token_to_phrase, labels_, labels, masked_indices, indices_replaced, inputs, mask_probability=phrase_mask_probability)
    
#     # 10% of the time, we replace masked input tokens with random word
#     indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
#     random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
#     inputs[indices_random] = random_words[indices_random]

#     # The rest of the time (10% of the time) we keep the masked input tokens unchanged
#     return inputs, labels


def extract_noun_phrases(text):
    doc = nlp(text)
    noun_phrases = []
    phrase_offsets = []

    for chunk in doc.noun_chunks:
        if chunk[0].pos_ == "DET":
            tokens = chunk[1:]
        else:
            tokens = chunk

        if len(tokens) > 1:
            noun_phrases.append(' '.join(token.text for token in tokens))
            #phrase_offsets.append((tokens[0].idx, tokens[-1].idx + len(tokens[-1])))

    return noun_phrases

def find_phrase_positions(text, phrase_list):
    # Split the text into words
    words = text.split()

    # Function to find the start and end index of a phrase
    def find_indices(phrase):
        phrase_words = phrase.split()
        for i in range(len(words) - len(phrase_words) + 1):
            if words[i:i+len(phrase_words)] == phrase_words:
                return i, i + len(phrase_words) #End の位置はフレーズ最後の単語位置+1
        return -1, -1  # If phrase not found

    # Find indices for each phrase in the list
    positions = {phrase: find_indices(phrase) for phrase in phrase_list}
    return positions


# 包含関係を見つける関数
def find_overlaps(phraseidx):
    overlaps = {}
    for key1, (start1, end1) in phraseidx.items():
        for key2, (start2, end2) in phraseidx.items():
            if key1 != key2 and start1 <= start2 and end1 >= end2:
                if key1 in overlaps:
                    overlaps[key1].add(key2)
                else:
                    overlaps[key1] = {key2}
    return overlaps

# 包含グループを作成する関数
def create_inclusion_groups(overlaps):
    groups = []
    for parent, children in overlaps.items():
        found = False
        for group in groups:
            if parent in group or not children.isdisjoint(group):
                group.update(children)
                group.add(parent)
                found = True
                break
        if not found:
            new_group = set(children)
            new_group.add(parent)
            groups.append(new_group)
    return groups

# ランダムにフレーズを削除する関数
def remove_phrases_randomly(phraseidx, groups):
    for group in groups:
        to_keep = random.choice(list(group))
        for phrase in group:
            if phrase != to_keep:
                del phraseidx[phrase]





def WS_PS(words, wordscores, phrase_dic, token_to_word):

    phidxlist = []
    for phidx in list(phrase_dic.values()):
        phidxlist.extend(list(range(phidx[0], phidx[1])))
    
    phrase_idx = list(phrase_dic.values())
    # Sum the PLL scores for each phrase
    phrase_pll_max = []
    phl = []

    phrases = []
    phs = []
    c = 0
    for i,score in enumerate(wordscores):
        # print(phrases)
        if i in phidxlist:
            
            for pidx in phrase_idx:
                if i in list(range(pidx[0],pidx[1])):
                    phl.append(score)
                    phs.append(words[i])
                    if type(token_to_word[i]) == list:
                        c += len(token_to_word[i])
                    else:
                        c += 1
                    
                if i+1 == pidx[1]:
                    phsum = sum(phl)/c #トークン数の平均であるべき 修正した
                    phrase_pll_max.append(phsum)
                    phl = []
                    phrases.append(phs)
                    phs = []
                    c = 0


         
        
        else:
            phrase_pll_max.append(score)
            phrases.append(words[i])
    
    return phrases, phrase_pll_max        





def func_tau(x, alpha):
    n = x.size
    t = stats.t.isf((alpha/n), n-2) #片側
    tau = (n-1)/np.sqrt(n) * np.sqrt(t**2/(n-2 + t**2))
    return tau

def Smirnov_Grubbs(x, alpha):
    x_ini = x.copy()
    while(True):
        mean = np.mean(x)
        std = np.std(x, ddof=1)
        imax, imin = np.argmax(x), np.argmin(x)
        itest = imax if np.abs(x[imax] - mean) > np.abs(x[imin] - mean) else imin

        test =  np.abs(x[itest] -mean)/std
        tau = func_tau(x, alpha)

        if(test > tau):
            x = np.delete(x, itest)
        else:
            break

    x_outlier = np.setdiff1d(x_ini,x) 

    x_ini_idx = []
    for xo in x_outlier:
        x_ini_idx.append(list(x_ini).index(xo))
    
    return x_outlier, x_ini_idx
    #https://kochi-labo.com/outlier_tests_smirnov_grubbs_python/






'''
    Misc setup code. 
'''
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


'''
    Dataset wrangling related code
'''
class TextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path: str, block_size=512):
        assert os.path.isfile(file_path)

        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, args.model_type + "_cached_lm_" + str(block_size) + "_" + filename
        )

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)

            self.examples = []
            with open(file_path, encoding="utf-8") as f:
                text = f.read()

            tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

            for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
                self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i: i + block_size]))
            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should loook for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)

def get_tokenized_file(file_path:str, tokenizer: PreTrainedTokenizer, block_size=512, shuffle=False, lazy=False):
    logger.info("Creating features from dataset file at %s", file_path)
    logger.info("Reading Line by Line")
    lines = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            if len(line) > 0 and not line.isspace():
                lines.append(line)
    logger.info("Done Reading Line By Line. About to pass through the tokenize")
    if lazy:
        return lines
    return tokenizer.batch_encode_plus(lines, truncation=True, add_special_tokens=True, max_length=block_size)["input_ids"]


class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, lazy:bool, file_path: str, block_size=512):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        self.lazy = lazy
        self.block_size = block_size

        self.tokenizer = tokenizer
        self.examples = get_tokenized_file(file_path, tokenizer, block_size, lazy=lazy)


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        tokenized = self.examples[i]
        if self.lazy:
            tokenized = self.tokenizer.encode_plus(tokenized, truncation=True, add_special_tokens=True, max_length=self.block_size)["input_ids"]
        return torch.tensor(tokenized, dtype=torch.long)


def load_and_cache_examples(args, tokenizer):
    file_paths = args.train_data_file
    assert len(args.train_data_file) == len(args.aux_task_names), 'Mismatch between the number of train files for MLM and the number of aux task names'
    datasets = {}
    for idx, file_path in enumerate(file_paths):
        task_name = args.aux_task_names[idx]
        if args.line_by_line:
            datasets[task_name] = LineByLineTextDataset(tokenizer, args, lazy=args.lazy_dataset, file_path=file_path, block_size=args.block_size)
        else:
            datasets[task_name] = TextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size)
    return datasets


'''
    Checkpointing related code. Inherited from Huggingface.
'''
def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []
    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))
    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)

def save_chkpt(args, id_, model, tokenizer, optimizer, scheduler, rotate_chkpt=True):
    checkpoint_prefix = "checkpoint"
    output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, id_))
    os.makedirs(output_dir, exist_ok=True)
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    torch.save(args, os.path.join(output_dir, "training_args.bin"))
    logger.info("Saving model checkpoint to %s", output_dir)

    if rotate_chkpt:
        _rotate_checkpoints(args, checkpoint_prefix)

    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
    logger.info("Saving optimizer and scheduler states to %s", output_dir)

'''
    Core MLM functionality
'''
def mask_tokens(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, args) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked"
            " language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()
    # labels = torch.tensor([labels.tolist()])
    # We sample a few tokens in each sequence for masked-LM training
    # (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, 0.15)#args.mask_probability
    

    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


# Run a batch of data through the model whilst checking for out of memory errors
def run_batch(model, batch, tokenizer, args, task_name, try_again=True):
    # try :

    pretrained_model_name = args.model_name_or_path#'roberta-base'#'bert-base-uncased'

    
    inputs_ = []
    labels_ = []
    # print(batch)
    for sent in batch:
        # sent_ = sent.clone()
        LENGTH = args.block_size
        # tokens = tokenizer.convert_ids_to_tokens(sent[:LENGTH-2])
        # print("tokens")
        # print(tokens)
        # token_list = tokens[1:-1]
        text =  tokenizer.decode(sent, add_special_tokens=False).replace("\n","").replace("<pad>","").replace("</s>","").replace("<s>","")
        # print("text_bf")
        # print(text)
        text = text.replace(","," , ").replace("."," . ").replace("-"," - ").replace("  "," ") #change!!



        # print("text_af")

        # print(text)

        mlm_list = [mlm_model.token_score(text, PLL_metric='within_word_l2r')[0][:LENGTH-2]]
        # print("score")
        # print(mlm_list)

        doc = [text]

        doc_score = []
        for sent in mlm_list:
            sent_score = []
            for w in sent:
                sent_score.append(w[1])
            doc_score.append(sent_score)


        
        for sentence,score in zip(doc, doc_score):
            # tokens = self.tokenizer.tokenize(sentence)
            tokens, offsets = tokenize_with_offsets(sentence, LENGTH,tokenizer) # offsets いらない
            # print("tokens")
            # print(tokens)
            tokens = tokens[1:-1]# <s>対策
            # print(tokens)
            scores =  score # Use actual MLM scores here
            words,wordscores,token_to_word = calculate_word_level_scores_forWeightWord(tokens,scores)


        # print("token_to_word")
        # print(token_to_word)#上手く言ってる

        phrase_list = extract_noun_phrases(text)
        phrase_idx = find_phrase_positions(text,phrase_list)
        # print(words)
        # phrase_idx


        # 包含関係の検出
        overlaps = find_overlaps(phrase_idx)

        # 包含グループの作成
        inclusion_groups = create_inclusion_groups(overlaps)

        # ランダムにフレーズを削除
        remove_phrases_randomly(phrase_idx, inclusion_groups)

        # 最初のインデックスで並び替え
        phrase_idx = dict(sorted(phrase_idx.items(), key=lambda item: item[1][0]))


        phrases, phrase_scores = WS_PS(words, wordscores, phrase_idx,token_to_word) #トークン数の平均

        ps = np.array(phrase_scores) * -1
        phrase_outlier, phrase_Mask_idx = Smirnov_Grubbs(ps,alpha=0.05)
        # ws = np.array(wordscores) * -1
        # word_outlier, word_Mask_idx = Smirnov_Grubbs(ws,alpha=0.05)


        phrase_idx_ = {}
        #外れ値のフレーズ
        for p in phrase_Mask_idx:
            candidate = phrases[p]
            if type(candidate)==list:
                candidate = [c.replace('Ġ', '') for c in candidate]
                candidate = " ".join(candidate)
                try:
                    phrase_idx_[candidate] = phrase_idx[candidate]
                except:
                    pass
        p_idx = list(phrase_idx_.values())

        # print(phrase_idx_)
        try:
            a = p_idx[0][0]
        except:
            p_idx=[(-100, -98)]


        word_to_phrase = []
        twp = []
        #どの単語がどのフレーズに含まれているか

        cnt = 0
        phrase_length = 100





        ttw_dic = {}
        for n,i in enumerate(token_to_word):
            ttw_dic[n] = i

        # print("ttw_dic")
        # print(ttw_dic)

        for k,v in ttw_dic.items():
            # print("k",k)
            for p_v in p_idx:
                if k in list(range(p_v[0],p_v[1])):
                    if k == p_v[0]:
                        cnt = 0
                        phrase_length = 100
                        twp = []
                    twp.append(k)
                    cnt += 1
                    phrase_length = p_v[1] - p_v[0]
                    if k == p_v[1] - 1:
                        p_idx = p_idx[1:]
                    break

            # print("twp",twp)
                    
            if cnt == phrase_length:
                # print("a",k)
                word_to_phrase.append(twp)
                cnt = 0
                phrase_length = 100
                twp = []
            elif k not in twp:
                # print("b",k)

                word_to_phrase.append(k)
           
        # print("doc")
        # print(doc)
        # print("word_to_phrase")
        # print(word_to_phrase)

        #どのトークンがどのフレーズに含まれているか
        token_to_word_to_phrase = word_to_phrase.copy()

        for n, tok in enumerate(token_to_word_to_phrase):
            if type(tok) == list:
                for m, t in enumerate(tok):
                    token_to_word_to_phrase[n][m] = ttw_dic[token_to_word_to_phrase[n][m]]
            else:
                token_to_word_to_phrase[n] = ttw_dic[token_to_word_to_phrase[n]]

        # print("token_to_word_to_phrase")
        # print(token_to_word_to_phrase)

        #下記のMLMの単語マスクを流用するために，単語の階層をなくす
        token_to_phrase = []
        for twp in token_to_word_to_phrase:
            if type(twp) == list:
                p = []
                for m, t in enumerate(twp):
                    if type(t) == list:
                        p.extend(t)
                    else:
                        p.append(t)
                token_to_phrase.append(p)
            else:
                token_to_phrase.append(twp)
        # print(sent_)
        # sent = torch.tensor(sent)

        sent_ = torch.tensor(tokenizer(doc, max_length=LENGTH, truncation=True)["input_ids"])
        # print("sent_")
        # print(sent_)

        # if "word" in task_name:
        task_info = {}

        for auxtask, mask, wmask, pmask, m_mask, m_random in zip(args.aux_task_names, args.mlm_probability, args.word_mask_probability, args.phrase_mask_probability, args.mlm_mask, args.mlm_random):
            task_info[auxtask] = {}
            task_info[auxtask]["mlm_probability"] = mask
            task_info[auxtask]["word_mask_probability"] = wmask
            task_info[auxtask]["phrase_mask_probability"] = pmask
            task_info[auxtask]["mlm_mask"] = m_mask
            task_info[auxtask]["mlm_random"] = m_random






        # try:
        #     print("Good!")
        try:
            input_, label = mask_tokens_PRP(sent_, tokenizer, task_info, task_name, token_to_word_to_phrase, token_to_phrase, token_to_word, mlm_list, wordscores, args)# if args.mlm else (batch, batch)
        except Exception as e:
            print(e)
            input_, label = mask_tokens(sent_, tokenizer, args)
        # elif "phrase" in task_name:
        #     input_, label = mask_tokens_PRP(sent_, tokenizer,token_to_phrase, args)# if args.mlm else (batch, batch)
        # else:
        # except Exception as e:
        #     print(e)
        #     input_, label = mask_tokens(sent_, tokenizer, args)
        
        del mlm_list
        torch.cuda.empty_cache()

        input_, label = input_pad(input_,label,tokenizer,LENGTH)

        # print("input_")
        # print(input_[0])
        # print("label")
        # print(label[0])
        inputs_.append(input_[0])
        labels_.append(label[0])
    # print(inputs_)
    # print(labels_)
    inputs = torch.stack(inputs_)
    labels = torch.stack(labels_)
    inputs = inputs.to(args.device)
    labels = labels.to(args.device)
    model.train()
    outputs = model(inputs, labels=labels)
    # except RuntimeError as e:
    #     gc.collect()
    #     if 'out of memory' in str(e):
    #         if try_again:
    #             print('| WARNING: ran out of memory during forward. Trying batch again')
    #         else:
    #             print('| WARNING: ran out of memory during forward. Skipping batch')
    #     else:
    #         print('Run into this new error : ', str(e))
    #     torch.cuda.empty_cache()
    #     if not try_again:
    #         return None
    #     else:
    #         outputs = run_batch(model, batch, tokenizer, args, task_name, try_again=False)
    return outputs

# Process a batch of data for a particular auxiliary task task
def process_task_batch(tartan_model, mlm_model, batch, tokenizer, args, task_name):
    outputs = run_batch(mlm_model, batch, tokenizer, args, task_name)

    # This could return none if we aren't able to process the batch even after clearing
    # the cuda cache after an out of memory erorr and re-trying
    loss_ = 0
    if outputs is not None:
        loss = outputs[0]  # mlm_model outputs are always tuple in transformers (see doc)

        # Store the gradients for the meta-learner
        scale = tartan_model.alpha_generator_algo[task_name] / args.gradient_accumulation_steps
        # print("loss")
        # print(loss)
        # loss = torch.mean(loss)

        if tartan_model.alpha_generator_algo.is_meta:
            # print("tartan_model.alpha_generator_algo.is_meta")

            gradients = torch.autograd.grad(loss, mlm_model.parameters(), allow_unused=True)
            tartan_model.set_aux_grads(gradients, aux_task_name=task_name)
            # print("tartan_model")
            # print(gradients)

            # Update the parameter gradients with the computed gradients
            with torch.no_grad():
                for (p, g) in zip(mlm_model.parameters(), gradients):
                    if g is None:
                        continue
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    p.grad.add_(g * scale)  # scaling included here because we are weighting the tasks which influences the gradients
                    del g

            loss_ = (loss * scale).item()
        else:
            # We are doing mt-tartan. No need to store the gradients - can just back-prop
            loss = loss * scale
            loss.backward()
            loss_ = loss.item()
    return loss_


# Core Training Procedure.
def train(
            args, train_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer,
            auxTaskModel: TartanModel = None) -> Tuple[int, float]:
    """ Train the model """

    # Setup data iterators
    train_dataloader = {}
    max_dataset_len, largest_dataset_name = -1, None
    for task_name, dataset in train_dataset.items():
        train_sampler = RandomSampler(dataset)
        bsz_ = args.per_gpu_train_batch_size
        if args.tapt_primsize:
            # Make the batch size for tapt the same as the batch size for the primary task.
            bsz_ = args.classf_iter_batchsz if 'TAPT' in task_name else args.per_gpu_train_batch_size
            # print("classf_iter_batchsz=TRUE, per_gpu_train_batch_size=FALSE")
            # if 'TAPT' in task_name:
            #     print("TRUE")
            # else:
            #     print("FALSE")

        train_dataloader[task_name] = DataLoader(
            dataset, sampler=train_sampler, batch_size=bsz_, collate_fn=collate_fn(tokenizer.pad_token_id), drop_last=True
        )
        if max_dataset_len < len(dataset):
            max_dataset_len = len(dataset)
            largest_dataset_name = task_name

    # Setup training duration details
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (max_dataset_len // (args.gradient_accumulation_steps * args.per_gpu_train_batch_size)) + 1
        logger.info('The number of epochs is : {}'.format(args.num_train_epochs))
    else:
        t_total = (max_dataset_len // (args.gradient_accumulation_steps * args.per_gpu_train_batch_size)) * args.num_train_epochs


    # Pre-trained model housekeeping
    model.resize_token_embeddings(len(tokenizer))

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    # Setup the optimizer for the base model
    optimizer = AdamW(
                        optimizer_grouped_parameters, betas=eval(args.classf_betas),
                        lr=args.learning_rate, eps=args.adam_epsilon, weight_decay=args.base_wd
                    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(args.classf_warmup_frac * t_total), num_training_steps=t_total
    )

    # Setup an optimizer for the parameters of the classifier heads
    classifier_params = auxTaskModel.get_classifier_params(withbase=False)
    classifier_optim = AdamW(
                                classifier_params, betas=eval(args.classf_betas),
                                weight_decay=args.classf_wd, lr=args.classf_lr
                            )

    classifier_scheduler = get_linear_schedule_with_warmup(
        classifier_optim, num_warmup_steps=int(args.classf_warmup_frac * t_total), num_training_steps=t_total
    )

    # Setup the auxiliary task weight generator
    args.train_epochs = t_total
    auxTaskModel.setup_alpha_generator(args)


    # Log info before beginning training.
    logger.info("***** Running training *****")
    for k, v in train_dataset.items():
        logger.info(" Task= {} Num examples = {}".format(k, len(v)))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Num Warmup Steps = %d", int(args.classf_warmup_frac * t_total))
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.per_gpu_train_batch_size
        * args.gradient_accumulation_steps,
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = {}. Will eval every {}".format(t_total, args.eval_every))

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if args.model_name_or_path and os.path.exists(args.model_name_or_path):
        # try:
        # set global_step to gobal_step of last saved checkpoint from model path
        checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
        global_step = int(checkpoint_suffix)
        epochs_trained = global_step // (max_dataset_len // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (max_dataset_len // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        # except ValueError:
        #     logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0




    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch"
    )
    set_seed(args)  # Added here for reproducibility
    classifier_dev_perfs = [] # For saving the dev set performance so we can checkpoint best.
    auxTaskModel.alpha_generator_algo.prep_epoch_start(global_step)
    early_stop = False # Indicator for early stopping

    for epoch in train_iterator:
        gc.collect()
        if early_stop:
            break

        # Log the current task weightings
        weights = {k: auxTaskModel.alpha_generator_algo[k] for k in auxTaskModel.alpha_generator_algo.weights.keys()}
        print('\nGStep = {} Weights : '.format(global_step), weights, '\n')

        # Setup iterator for task with the most amount of data
        epoch_iterator = tqdm(train_dataloader[largest_dataset_name], desc="Iteration")
        
        # Setup Iterators for the other tasks.
        aux_task_iterators = {}
        for task_id, task_data in train_dataloader.items():
            if task_id == largest_dataset_name:
                continue
            aux_task_iterators[task_id] = iter(task_data)

        # Loop through the auxiliary datasets 
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            # try:
                # Forward mode for the task with the largest dataset size
            this_loss = process_task_batch(auxTaskModel, model, batch, tokenizer, args, largest_dataset_name)
            tr_loss += this_loss / len(args.aux_task_names)
            # except RuntimeError as e:
            #     print(e)
            #     gc.collect()
            #     torch.cuda.empty_cache()
            #     print('Run into error when process_task_batch. Skipping')

            # Perform forward mode for the other tasks
            other_tasks = list(set(args.aux_task_names) - set([largest_dataset_name]))
            for task_id, task_name in enumerate(other_tasks):
                other_batch = next(aux_task_iterators[task_name], None)
                if other_batch is None:
                    aux_task_iterators[task_name] = iter(train_dataloader[task_name])
                    other_batch = next(aux_task_iterators[task_name], None)

                assert other_batch is not None, 'We should have more data for {} since we have reset the iterator'.format(task_name)
                # try:
                this_loss = process_task_batch(auxTaskModel, model, other_batch, tokenizer, args, task_name)
                tr_loss += this_loss / len(args.aux_task_names)
                # except:
                #     gc.collect()
                #     torch.cuda.empty_cache()
                #     print('Run into error when process_task_batch. Skipping')

            # We have run the auxiliary tasks. Now run the primary task and update the gradietns of the task weightings
            auxTaskModel.classifier_sample_grad()


            if auxTaskModel.alpha_generator_algo.is_meta:
                # Zero-out the aux grads because we are done with them at the moment
                for task_name in args.aux_task_names:
                    auxTaskModel.set_aux_grads(None, aux_task_name=task_name)


            # If we have accumulated enough gradients, we can now do a gradient descent step.
            if (step + 1) % args.gradient_accumulation_steps == 0:

                # Clip the gradients of the pre-trained model base and the classifier parameters
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(auxTaskModel.get_classifier_params(), args.max_grad_norm)

                # Gradient descent on the classifier parameters
                classifier_optim.step()
                classifier_scheduler.step()
                classifier_optim.zero_grad()

                # Gradient descent on the pre-trained model base parameters
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()

                global_step += 1

                # Housekeeping for the task weight generator
                auxTaskModel.alpha_generator_algo.prep_epoch_start(global_step)
                if auxTaskModel.alpha_generator_algo.is_meta:
                    # Reset the dev head so it is not based on stale pre-trained base parameters.
                    auxTaskModel.reset_dev_head()

                    # Update the task weightings
                    auxTaskModel.alpha_generator_algo.update_meta_weights()

                # House keeping.
                torch.cuda.empty_cache()
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    save_chkpt(args, str(global_step), model, tokenizer, optimizer, scheduler, rotate_chkpt=True)

                # Log the current task weightings
                if global_step % (args.eval_every // 2) == 0:
                    weights = {k: auxTaskModel.alpha_generator_algo[k] for k in auxTaskModel.alpha_generator_algo.weights.keys()}
                    print('\nGStep = {} Weights : '.format(global_step), weights, '\n')

                # We will evaluate the model on the primary task and save if it is the current validation performance so far.
                if global_step % args.eval_every == 0:
                    train_metrics = auxTaskModel.get_metrics(reset=True)
                    dev_metrics = auxTaskModel.evaluate_classifier(set_='dev')
                    test_metrics = auxTaskModel.evaluate_classifier(set_='test')
                    for k, v in train_metrics.items():
                        print_out = "[{}] | Train : {:.3f} | Dev Set : {:.3f} | Test Set : {:.3f}".format(k, v, dev_metrics[k], test_metrics[k])
                        logger.info(print_out)

                    classifier_dev_perfs.append(dev_metrics[args.classf_metric])
                    if dev_metrics[args.classf_metric] >= max(classifier_dev_perfs):
                        # We want to save the best model here
                        print('Current best dev f1 = {} achieved. Saving model'.format(dev_metrics[args.classf_metric]))
                        logger.info('Now Saving the Classifier Model')
                        auxTaskModel.save()
                        logger.info('Saving Base Model')
                        save_chkpt(args, 'best', model, tokenizer, optimizer, scheduler, rotate_chkpt=False)

                    # Record the metrics for the alpha generator
                    auxTaskModel.alpha_generator_algo.record_epoch_end(global_step, dev_metrics[args.classf_metric], test_metrics[args.classf_metric])
                    if len(classifier_dev_perfs) > args.classf_patience:
                        # If we have not seen any improvement in args.classf_patience, then we will early stop
                        max_ = max(classifier_dev_perfs)
                        recent_max = max(classifier_dev_perfs[-args.classf_patience:])
                        if recent_max < max_:
                            print('Stopping Early at Epoch {} because No Improvement in Dev Set Accuracy'.format(epoch))
                            train_iterator.close()
                            early_stop = True
                            break

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    return global_step, tr_loss / global_step

# Perform finetuning using only the primary task
def finetune_primary_task(args, auxTaskModel):
    # Load the primary task classifier parameters
    auxTaskModel.load_primary(args.device)

    # Evaluate the refreshed model
    test_metrics = auxTaskModel.evaluate_classifier(set_='test')
    dev_metrics = auxTaskModel.evaluate_classifier(set_='dev')
    print('Before Training. Dev  (F1={:.3f}, Accuracy={:.3f})'.format(dev_metrics['f1'], dev_metrics['accuracy']))
    print('Before Training. Test (F1={:.3f}, Accuracy={:.3f})'.format(test_metrics['f1'], test_metrics['accuracy']))


    # Setup fine-tuning optimizer and learning rate scheduler
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight", "layer_norm.weight"]
    classifier_params = auxTaskModel.get_classifier_params(keys=[auxTaskModel.primary_task_id], withbase=True)
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in classifier_params if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in classifier_params if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    this_optim = AdamW(
                                optimizer_grouped_parameters, betas=eval(args.classf_betas),
                                weight_decay=args.classf_wd, lr=args.classf_ft_lr
                            )
    this_lr_scheduler = None

    best_f1, best_acc, perfs, dev_perfs  = auxTaskModel.train_primary(
                                                                        args.classf_ft_iters, this_optim, this_lr_scheduler, 
                                                                        args.max_grad_norm, patience=args.classf_ft_patience,
                                                                        metric=args.classf_metric
                                                                    )

    # Caching the best performance based on the chosen metric
    save_model = False
    if args.classf_metric == 'f1':
        save_model = dev_perfs[0] > dev_metrics['f1']
        best_f1 = best_f1 if dev_perfs[0] > dev_metrics['f1'] else test_metrics['f1']
        best_acc = best_acc if dev_perfs[0] > dev_metrics['f1'] else test_metrics['accuracy']
    else:
        save_model = dev_perfs[1] > dev_metrics['accuracy']
        best_f1 = best_f1 if dev_perfs[1] > dev_metrics['accuracy'] else test_metrics['f1']
        best_acc = best_acc if dev_perfs[1] > dev_metrics['accuracy'] else test_metrics['accuracy']
    
    # Do the saving of the best model here
    if save_model:
        logger.info('Now Saving the Classifier Model')
        auxTaskModel.save()

    print('Final Test (F1={:.3f}, Accuracy={:.3f})'.format(best_f1, best_acc))
    pickle.dump(perfs, open(os.path.join(args.output_dir, 'ftmodel.perf.pkl'), 'wb'))
    return save_model


def get_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--train_data_file", nargs='+', default=None, type=str, required=True, 
        help="The input training data file(s). Number of files specified must match number of aux-task-names"
    )
    parser.add_argument(
        "--aux-task-names", nargs='+', default=None, type=str, help="The names of the auxiliary tasks"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--model_type", type=str, default='roberta-base', required=True, help="The model architecture to be trained or fine-tuned.",
    )

    parser.add_argument(
        "--line_by_line",
        action="store_true",
        help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )
    parser.add_argument(
        "--should_continue", action="store_true", help="Whether to continue from latest checkpoint in output_dir"
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
    )

    parser.add_argument(
        "--mlm", action="store_true", help="Train with masked-language modeling loss instead of language modeling."
    )
    # parser.add_argument(
    #     "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
    # )

    parser.add_argument(
        "--config_name",
        default=None,
        type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path. If both are None, initialize a new config.",
    )
    parser.add_argument(
        "--tokenizer_name",
        default='roberta-base',
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
    )
    parser.add_argument(
        "--block_size",
        default=512,
        type=int,
        help="Optional input sequence length after tokenization."
        "The training dataset will be truncated in block of this size for training."
        "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")

    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=1.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")


    parser.add_argument(
        "--base_task_dataset_file",
        type=str,
        help='Name of file for master task'
    )
    
    parser.add_argument(
        "--lazy-dataset",
        action='store_true',
    )

    parser.add_argument(
        "--eval_every",
        type=int,
        default=30,
        help="Frequency with which to evaluate the model"
    )

    parser.add_argument(
        "--no_final_finetuning",
        action='store_true',
        help='turns off further task-specific finetuing'
    )

    parser.add_argument("--classf_warmup_frac", type=float, default=0.06)
    parser.add_argument("--classf_betas", type=str, default="(0.9,0.98)")
    parser.add_argument("--from-scratch", action='store_true')
    parser.add_argument("--classf_wd", type=float, default=0.1)
    parser.add_argument("--base_wd", type=float, default=0.01)
    parser.add_argument("--classf_patience", type=int, default=5)
    parser.add_argument("--classf_lr", type=float, default=2e-5, help="Learning rate of classifier")
    parser.add_argument("--classf_ft_lr", type=float, default=2e-6, help="Learning rate of classifier for finetuning")
    parser.add_argument("--classf_ft_iters", type=int, default=10, help='Number of finetuning iterations')
    parser.add_argument("--classf_ft_patience", type=int, default=3, help='finetuning patience iterations')
    parser.add_argument("--classf-metric", type=str, default='f1', choices=['f1', 'accuracy'])
    parser.add_argument("--tapt-primsize", action='store_true', help='Make tapt batch size the size of primary task')
    # parser.add_argument("--word_mask_probability", type=float, default=0.2)
    # parser.add_argument("--phrase_mask_probability", type=float, default=0.2)

    parser.add_argument('--word_mask_probability', required=True, nargs="*", type=float, help='a list of word_mask_probability') 
    parser.add_argument('--phrase_mask_probability', required=True, nargs="*", type=float, help='a list of phrase_mask_probability') 
    parser.add_argument("--mlm_probability", required=True, nargs="*", type=float, help='a list of mask_probability') 
    parser.add_argument("--mlm_mask", required=True, nargs="*", type=float, help='a list of mlm_mask_probability') 
    parser.add_argument("--mlm_random", required=True, nargs="*", type=float, help='a list of mlm_random_probability') 


    add_config_args(parser)
    add_tartan_args(parser)
    args = parser.parse_args()
    return args

def main():
    args = get_args()



    if args.model_type in ["bert", "roberta", "distilbert", "camembert"] and not args.mlm:
        raise ValueError(
            "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the --mlm "
            "flag (masked language modeling)."
        )

    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
        else:
            args.model_name_or_path = sorted_checkpoints[-1]
            print('Used Should Continue and model found is : ', args.model_name_or_path)
    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
        and not args.should_continue
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()


    # Set seed
    set_seed(args)

    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, output_hidden_states=True, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, output_hidden_states=True, cache_dir=args.cache_dir)
    else:
        # When we release a pip version exposing CONFIG_MAPPING,
        # we can do `config = CONFIG_MAPPING[args.model_type]()`.
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --config_name"
        )

    # Setting up tokenizer and pre-trained model
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )

    if args.model_name_or_path and not args.from_scratch:
        model = AutoModelWithLMHead.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelWithLMHead.from_config(config)


    model_name = args.model_name_or_path
    assert model_name, 'The name of the model is not Set. Maybe use roberta-base as the default'
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setting up the dataset
    train_dataset = load_and_cache_examples(args, tokenizer)
    if args.block_size <= 0:
        args.block_size = tokenizer.model_max_length
        # Our input block size will be the max possible for the model
    else:
        args.block_size = min(args.block_size, tokenizer.model_max_length)


    # Instantiate the model with the auxiliary tasks
    logger.info("Instantiating AuxTaskModel")
    base_task_dataset_files = get_auxtask_files(args.primary_task_id)
    auxTaskModel = TartanModel(
                                        model_name,
                                        model,
                                        base_task_dataset_files,

                                        max_seq_len=args.classf_max_seq_len,
                                        dropout=args.classifier_dropout,

                                        batch_sz=args.classf_iter_batchsz,
                                        save_path=os.path.join(args.output_dir, 'modelWAuxTasks.pth'),
                                        primary_task_id=args.primary_task_id,
                                        grad_accum_factor=args.gradient_accumulation_steps,
                                        dev_batch_sz=args.dev_batch_sz
                                    )


    # # 複数GPU使用宣言
    # Parallel_Frag = True #使わない時は False
    # if Parallel_Frag:
    #     model = torch.nn.DataParallel(model) # make parallel
    #     cudnn.benchmark = True

    # Move the model to the appropriate device
    model.to(args.device)
    auxTaskModel.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    # Do end-task aware training
    if args.do_train:
        global_step, tr_loss = train(args, train_dataset, model, tokenizer, auxTaskModel=auxTaskModel)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        # Saving the visualization
        group_aux = args.weight_strgy == 'meta'
        auxTaskModel.alpha_generator_algo.viz_results(args.output_dir, group_aux=group_aux)

    # We want to now do finetuning the model on only the primary task
    if not args.no_final_finetuning:
        improved_fom_finetuning = finetune_primary_task(args, auxTaskModel)

    # Saving best-practices: if you use save_pretrained for the model and tokenizer,
    # you can reload them using from_pretrained()
    if args.do_train and improved_fom_finetuning:
        # Create output directory if needed
        os.makedirs(args.output_dir, exist_ok=True)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
                            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = AutoModelWithLMHead.from_pretrained(args.output_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
        model.to(args.device)


if __name__ == "__main__":
    main()
