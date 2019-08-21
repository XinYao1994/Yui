# https://github.com/openai/gpt-2
# https://medium.com/@ngwaifoong92/beginners-guide-to-retrain-gpt-2-117m-to-generate-custom-text-content-8bb5363d8b7f


import fire
import json
import os
import numpy as np
import tensorflow as tf

import sys

sys.path.insert(0, "write/gpt-2/src")
try:
    import model, sample, encoder
except ImportError:
    print('No import')

model_name='774M'
seed=None
nsamples=1
batch_size=1
length=None
temperature=1
top_k=0
models_dir='write/gpt-2/models'

models_dir = os.path.expanduser(os.path.expandvars(models_dir))
print(os.path.join(models_dir, model_name))

if batch_size is None:
    batch_size = 1
assert nsamples % batch_size == 0

enc = encoder.get_encoder(model_name, models_dir)
hparams = model.default_hparams()

# load gpt-2 model
with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
    hparams.override_from_dict(json.load(f))

length = 600

if length is None:
    length = hparams.n_ctx // 2
elif length > hparams.n_ctx:
    raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

with tf.Session(graph=tf.Graph()) as sess:
    context = tf.placeholder(tf.int32, [batch_size, None])
    np.random.seed(seed)
    tf.set_random_seed(seed)
    output = sample.sample_sequence(
        hparams=hparams, length=length,
        context=context,
        batch_size=batch_size,
        temperature=temperature, top_k=top_k
    )

    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
    saver.restore(sess, ckpt)

    raw_text = input("hint >>> ")
    all_text = raw_text
    b_len = len(enc.encode(all_text))
    index = 0
    while index <= length:
        # header = all_text[-b_len:]
        # context_tokens = enc.encode(header)
        context_tokens = enc.encode(all_text)
        generated = 0
        for _ in range(nsamples // batch_size):
            out = sess.run(output, feed_dict={
                context: [context_tokens for _ in range(batch_size)]
            })[:, len(context_tokens):]
            for i in range(batch_size):
                generated += 1
                text = enc.decode(out[i])
                all_text = all_text + text
                index = index + len(text)
    
    print(all_text)


'''
if length is None:
    length = hparams.n_ctx
elif length > hparams.n_ctx:
    raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)


with tf.Session(graph=tf.Graph()) as sess:
    np.random.seed(seed)
    tf.set_random_seed(seed)

    output = sample.sample_sequence(
        hparams=hparams, length=length,
        start_token=enc.encoder['<|endoftext|>'],
        batch_size=batch_size,
        temperature=temperature, top_k=top_k
    )[:, 1:]

    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
    saver.restore(sess, ckpt)

    generated = 0
    while nsamples == 0 or generated < nsamples:
        out = sess.run(output)
        for i in range(batch_size):
            generated += batch_size
            text = enc.decode(out[i])
            print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
            print(text)
'''








