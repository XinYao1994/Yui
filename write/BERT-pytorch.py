# https://github.com/google-research/bert.git
# Recent research shows that BERT is an MRF language model. 
# Thus you can begin with a sentence of [MASK] tokens and 
# generate words one by one in arbitrary order (instead
# of the common left-to-right chain decomposition).
# https://colab.research.google.com/drive/1MxKZGtQ9SSBjTK5ArsZ5LKhkztzg52RV#scrollTo=8BR0JVmlTvEQ
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import codecs
import collections
import json
import re
import os
import pprint
import tensorflow as tf

sys.path.insert(0, "./write/model/bert")
try:
    import modeling
    import tokenization
except ImportError:
    print('No import')

import numpy as np
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

model_version = 'bert-base-uncased'
model = BertForMaskedLM.from_pretrained(model_version)
model.eval()

if torch.cuda.is_available():
    model = model.cuda()

tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=model_version.endswith("uncased"))

def tokenize_batch(batch):
    return [tokenizer.convert_tokens_to_ids(sent) for sent in batch]

def untokenize_batch(batch):
    return [tokenizer.convert_ids_to_tokens(sent) for sent in batch]

def detokenize(sent):
    """ Roughly detokenizes (mainly undoes wordpiece) """
    new_sent = []
    for i, tok in enumerate(sent):
        if tok.startswith("##"):
            new_sent[len(new_sent) - 1] = new_sent[len(new_sent) - 1] + tok[2:]
        else:
            new_sent.append(tok)
    return new_sent

CLS = '[CLS]'
SEP = '[SEP]'
MASK = '[MASK]'
mask_id = tokenizer.convert_tokens_to_ids([MASK])[0]
sep_id = tokenizer.convert_tokens_to_ids([SEP])[0]
cls_id = tokenizer.convert_tokens_to_ids([CLS])[0]

def generate_step(out, gen_idx, temperature=None, top_k=0, sample=False, return_list=True):
    """ Generate a word from from out[gen_idx]
    
    args:
        - out (torch.Tensor): tensor of logits of size batch_size x seq_len x vocab_size
        - gen_idx (int): location for which to generate for
        - top_k (int): if >0, only sample from the top k most probable words
        - sample (Bool): if True, sample from full distribution. Overridden by top_k 
    """
    logits = out[:, gen_idx]
    if temperature is not None:
        logits = logits / temperature
    if top_k > 0:
        kth_vals, kth_idx = logits.topk(top_k, dim=-1)
        dist = torch.distributions.categorical.Categorical(logits=kth_vals)
        idx = kth_idx.gather(dim=1, index=dist.sample().unsqueeze(-1)).squeeze(-1)
    elif sample:
        dist = torch.distributions.categorical.Categorical(logits=logits)
        idx = dist.sample().squeeze(-1)
    else:
        idx = torch.argmax(logits, dim=-1)
    return idx.tolist() if return_list else idx

def get_init_text(seed_text, max_len, batch_size = 1, rand_init=False):
    """ Get initial sentence by padding seed_text with either masks or random words to max_len """
    batch = [seed_text + [MASK] * max_len + [SEP] for _ in range(batch_size)]
    #if rand_init:
    #    for ii in range(max_len):
    #        init_idx[seed_len+ii] = np.random.randint(0, len(tokenizer.vocab))
    
    return tokenize_batch(batch)

def printer(sent, should_detokenize=True):
    if should_detokenize:
        sent = detokenize(sent)[1:-1]
    print(" ".join(sent))

import math
import time

def parallel_sequential_generation(seed_text, batch_size=10, max_len=15, top_k=0, temperature=None, max_iter=300, burnin=200,
                                   cuda=False, print_every=10, verbose=True):
    """ Generate for one random position at a timestep
    
    args:
        - burnin: during burn-in period, sample from full distribution; afterwards take argmax
    """
    seed_len = len(seed_text)
    batch = get_init_text(seed_text, max_len, batch_size)
    
    for ii in range(max_iter):
        kk = np.random.randint(0, max_len)
        for jj in range(batch_size):
            batch[jj][seed_len+kk] = mask_id
        inp = torch.tensor(batch).cuda() if cuda else torch.tensor(batch)
        out = model(inp)
        topk = top_k if (ii >= burnin) else 0
        idxs = generate_step(out, gen_idx=seed_len+kk, top_k=topk, temperature=temperature, sample=(ii < burnin))
        for jj in range(batch_size):
            batch[jj][seed_len+kk] = idxs[jj]
            
        if verbose and np.mod(ii+1, print_every) == 0:
            for_print = tokenizer.convert_ids_to_tokens(batch[0])
            for_print = for_print[:seed_len+kk+1] + ['(*)'] + for_print[seed_len+kk+1:]
            print("iter", ii+1, " ".join(for_print))
            
    return untokenize_batch(batch)

def parallel_generation(seed_text, batch_size=10, max_len=15, top_k=0, temperature=None, max_iter=300, sample=True, 
                        cuda=False, print_every=10, verbose=True):
    """ Generate for all positions at each time step """
    seed_len = len(seed_text)
    batch = get_init_text(seed_text, max_len, batch_size)
    
    for ii in range(max_iter):
        inp = torch.tensor(batch).cuda() if cuda else torch.tensor(batch)
        out = model(inp)
        for kk in range(max_len):
            idxs = generate_step(out, gen_idx=seed_len+kk, top_k=top_k, temperature=temperature, sample=sample)
            for jj in range(batch_size):
                batch[jj][seed_len+kk] = idxs[jj]
            
        if verbose and np.mod(ii, print_every) == 0:
            print("iter", ii+1, " ".join(tokenizer.convert_ids_to_tokens(batch[0])))
    
    return untokenize_batch(batch)
            
def sequential_generation(seed_text, batch_size=10, max_len=15, leed_out_len=15, 
                          top_k=0, temperature=None, sample=True, cuda=False):
    """ Generate one word at a time, in L->R order """
    seed_len = len(seed_text)
    batch = get_init_text(seed_text, max_len, batch_size)
    
    for ii in range(max_len):
        inp = [sent[:seed_len+ii+leed_out_len]+[sep_id] for sent in batch]
        inp = torch.tensor(batch).cuda() if cuda else torch.tensor(batch)
        out = model(inp)
        idxs = generate_step(out, gen_idx=seed_len+ii, top_k=top_k, temperature=temperature, sample=sample)
        for jj in range(batch_size):
            batch[jj][seed_len+ii] = idxs[jj]
        
    return untokenize_batch(batch)


def generate(n_samples, seed_text="[CLS]", batch_size=10, max_len=25, 
             generation_mode="parallel-sequential", 
             sample=True, top_k=100, temperature=1.0, burnin=200, max_iter=500,
             cuda=False, print_every=1):
    # main generation function to call
    sentences = []
    n_batches = math.ceil(n_samples / batch_size)
    start_time = time.time()
    for batch_n in range(n_batches):
        if generation_mode == "parallel-sequential":
            batch = parallel_sequential_generation(seed_text, batch_size=batch_size, max_len=max_len, top_k=top_k,
                                                   temperature=temperature, burnin=burnin, max_iter=max_iter, 
                                                   cuda=cuda, verbose=False)
        elif generation_mode == "sequential":
            batch = sequential_generation(seed_text, batch_size=batch_size, max_len=max_len, top_k=top_k, 
                                          temperature=temperature, leed_out_len=leed_out_len, sample=sample,
                                          cuda=cuda)
        elif generation_mode == "parallel":
            batch = parallel_generation(seed_text, batch_size=batch_size,
                                        max_len=max_len, top_k=top_k, temperature=temperature, 
                                        sample=sample, max_iter=max_iter, 
                                        cuda=cuda, verbose=False)
        
        if (batch_n + 1) % print_every == 0:
            print("Finished batch %d in %.3fs" % (batch_n + 1, time.time() - start_time))
            start_time = time.time()
        
        sentences += batch
    return sentences

n_samples = 5
batch_size = 5
max_len = 40
top_k = 100
temperature = 1.0
generation_mode = "parallel-sequential"
leed_out_len = 5 # max_len
burnin = 250
sample = True
max_iter = 500

seed_text = "[CLS]".split()
bert_sents = generate(n_samples, seed_text=seed_text, batch_size=batch_size, max_len=max_len,
                      generation_mode=generation_mode,
                      sample=sample, top_k=top_k, temperature=temperature, burnin=burnin, max_iter=max_iter,
                      cuda=torch.cuda.is_available())

for sent in bert_sents:
  printer(sent, should_detokenize=True)

'''

'''
# Load pre-trained model tokenizer (vocabulary)
modelpath = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(modelpath)

text = "dummy. although he had already eaten a large meal, he was still very hungry."
target = "hungry"
tokenized_text = tokenizer.tokenize(text)

# Mask a token that we will try to predict back with `BertForMaskedLM`
masked_index = tokenized_text.index(target)
tokenized_text[masked_index] = '[MASK]'

# Convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
segments_ids = [1] * len(tokenized_text)
# this is for the dummy first sentence. 
segments_ids[0] = 0
segments_ids[1] = 0

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])
# Load pre-trained model (weights)
model = BertForMaskedLM.from_pretrained(modelpath)
model.eval()

# Predict all tokens
predictions = model(tokens_tensor, segments_tensors)
predicted_index = torch.argmax(predictions[0, masked_index]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])

print("Original:", text)
print("Masked:", " ".join(tokenized_text))

print("Predicted token:", predicted_token)
print("Other options:")
# just curious about what the next few options look like.
for i in range(10):
    predictions[0,masked_index,predicted_index] = -11100000
    predicted_index = torch.argmax(predictions[0, masked_index]).item()
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])
    print(predicted_token)
'''



