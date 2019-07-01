import sys

sys.path.insert(0, "write/model")
try:
    import seqgan
except ImportError:
    print('No import')

import os.path
import numpy as np
import tensorflow as tf
import random
import subprocess

import gzip
import codecs

#### parameter settings ####
NUM_EMB = 4
EMB_DIM = 20
HIDDEN_DIM = 25
SEQ_LENGTH = 10
START_TOKEN = 0

EPOCH_ITER = 1000
CURRICULUM_RATE = 0.02  # how quickly to move from supervised training to unsupervised
TRAIN_ITER = 100000  # generator/discriminator alternating
D_STEPS = 3  # how many times to train the discriminator per generator step
LEARNING_RATE = 0.01 * SEQ_LENGTH
SEED = 88

"""Char-based Seq-GAN on data from a book."""
DATA_FILE = 'write/data/book.txt'

def tokenize(s):
    return [c for c in ' '.join(s.split())]

def get_data(download=not os.path.exists(DATA_FILE)):
    """Downloads and parses Moby Dick."""
    if download:
        subprocess.check_output(
            ['wget', 'http://www.gutenberg.org/cache/epub/2701/pg2701.txt',
             '-O', DATA_FILE])
    token_stream = []
    is_gzip = False
    try:
        open(DATA_FILE).read(2)
    except UnicodeDecodeError:
        is_gzip = True
    with gzip.open(DATA_FILE) if is_gzip else codecs.open(DATA_FILE, 'r', 'utf-8',errors='ignore') as f:
        for line in f:
            line = line if not is_gzip else line.decode('utf-8')
            if ('Call me Ishmael.' in line or token_stream) and line.strip():
                token_stream.extend(tokenize(line.strip().lower()))
                token_stream.append(' ')
            if len(token_stream) > 10000 * SEQ_LENGTH:  # enough data
                break
    return token_stream

class BookGRU(seqgan.GRU):
    def d_optimizer(self, *args, **kwargs):
        return tf.train.AdamOptimizer()  # ignore learning rate

    def g_optimizer(self, *args, **kwargs):
        return tf.train.AdamOptimizer()  # ignore learning rate

def get_random_sequence(token_stream, word2idx):
    """Returns random subsequence."""
    start_idx = random.randint(0, len(token_stream) - SEQ_LENGTH)
    return [word2idx[tok] for tok in token_stream[start_idx:start_idx + SEQ_LENGTH]]


def verify_sequence(three_grams, seq):
    """Not a true verification; only checks 3-grams."""
    for i in range(len(seq) - 3):
        if tuple(seq[i:i + 3]) not in three_grams:
            return False
    return True

def main():
    random.seed(SEED)
    np.random.seed(SEED)

    token_stream = get_data()
    words = ['_START'] + list(set(token_stream))
    word2idx = dict((word, i) for i, word in enumerate(words))
    num_words = len(words)

    three_grams = dict((tuple(word2idx[w] for w in token_stream[i:i + 3]), True)
                       for i in range(len(token_stream) - 3))
    print('num words', num_words)
    print('stream length', len(token_stream))
    print('distinct 3-grams', len(three_grams))

    trainable_model = BookGRU(num_words, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print('training')
    for epoch in range(TRAIN_ITER // EPOCH_ITER):
        print('epoch', epoch)
        proportion_supervised = max(0.0, 1.0 - CURRICULUM_RATE * epoch)
        seqgan.train_epoch(
            sess, trainable_model, EPOCH_ITER,
            proportion_supervised=proportion_supervised,
            g_steps=1, d_steps=D_STEPS,
            next_sequence=lambda: get_random_sequence(token_stream, word2idx),
            verify_sequence=lambda seq: verify_sequence(three_grams, seq),
            words=words)

if __name__ == '__main__':
    main()
