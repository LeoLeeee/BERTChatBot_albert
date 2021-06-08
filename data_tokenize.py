# coding=utf-8

import os
import getConfig
import jieba  #结巴是国内的一个分词python库，分词效果非常不错。pip3 install jieba安装
from zhon.hanzi import punctuation
import re
import collections
import numpy as np


def get_convs():
    # 从xiaohuangji50w_nofenci.conv中获取所有对话
    gConfig = {}
    gConfig=getConfig.get_config()
    conv_path = gConfig['resource_data']
    convs = []  # 用于存储对话的列表
    with open(conv_path, encoding='utf-8') as f:
        one_conv = []        # 存储一次完整对话
        for line in f:
            line = line.strip('\n').replace('?', '')#去除换行符，并将原文件中已经分词的标记去掉，重新用结巴分词.
            line=re.sub(r"[%s]+" %punctuation, "",line)
            if line == '':
                continue
            if line[0] == gConfig['e']:
                if one_conv:
                    convs.append(one_conv)
                one_conv = []
            elif line[0] == gConfig['m']:
                tmp = line.split(' ')[1]
                if '=' not in tmp:
                    # print(one_conv)
                    one_conv.append(tmp)#将一次完整的对话存储下来
                else:
                    continue
    return convs

def tokenize(convs, token='word'):  # "word": 单个词，"char": 单个字  
    # 把所有对话分词化，每个对话包括两个list
    if token == 'word':
        print("Using 'word' as tokens")
        seq = []        
        for conv in convs:
            if len(conv) == 1:
                continue
            if len(conv) % 2 != 0:  # 因为默认是一问一答的，所以需要进行数据的粗裁剪，对话行数要是偶数的
                conv = conv[:-1]
            for i in range(len(conv)):
                if i % 2 == 0:
                    ask=[word for word in jieba.cut(conv[i])]#使用jieba分词器进行分词
                    answer=[word for word in jieba.cut(conv[i+1])]
                    seq.append([ask, answer])#因为i是从0开始的，因此偶数行为发问的语句，奇数行为回答的语
        return seq
    elif token == 'char':
        print("Using 'char' as tokens")
        seq = []        
        for conv in convs:
            if len(conv) == 1:
                continue
            if len(conv) % 2 != 0:  # 因为默认是一问一答的，所以需要进行数据的粗裁剪，对话行数要是偶数的
                conv = conv[:-1]
            for i in range(len(conv)):
                if i % 2 == 0:
                    ask=[char for char in conv[i]] #使用单个磁
                    answer=[char for char in conv[i+1]]
                    seq.append([ask, answer])#因为i是从0开始的，因此偶数行为发问的语句，奇数行为回答的语
        return seq
    else:
        print('ERROR: unknown token type: ' + token)

def count_corpus(tokens):  #@save
    """Count token frequencies."""
    # Here `tokens` is a 1D list or 2D list
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # Flatten a list of token lists into a list of tokens
        tokens = [token for lines in tokens for line in lines for token in line]
    return collections.Counter(tokens)

class Vocab:  # @save
    # 将分词化后的数据分析得到一个词表示例。该词表将每个分词对应一个数字（下标），该词表可以通过分词得到该下标，也可以通过下标获得分词。

    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
        counter = count_corpus(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x:x[1], reverse=True)
        self.unk, uniq_tokens = 0, reserved_tokens
        uniq_tokens += [token for token, freq in self.token_freqs if freq >= min_freq and token not in uniq_tokens]
        self.idx_to_token, self.token_to_idx = [], dict()
        for token in uniq_tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple, np.ndarray)):
            try:
                indices = int(indices)
            except:
                print(indices.dtype)
            return self.idx_to_token[indices]
        return [self.to_tokens(idx) for idx in indices]

def load_corpus_xiaohuangji50w_nofenci(max_tokens=-1):  # @save
	# 读取xiaohuangji50w_nofenci.conv里的对话，做成词表。并返回这些对话分词的下标行成的数组和词表。
    convs = get_convs()
    tokens = tokenize(convs, token = "word")
    vocab = Vocab(tokens)
    corpus = [vocab[token] for lines in tokens for line in lines for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

