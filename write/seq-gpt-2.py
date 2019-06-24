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

model_name='345M',
seed=None,
nsamples=0,
batch_size=1,
length=None,
temperature=1,
top_k=0,
models_dir='write/gpt-2/models'

models_dir = os.path.expanduser(os.path.expandvars(models_dir))
print(os.path.join(models_dir, model_name))
enc = encoder.get_encoder(model_name, models_dir)
hparams = model.default_hparams()

# load gpt-2 model
with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))








