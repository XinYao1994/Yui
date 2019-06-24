# https://github.com/zihangdai/xlnet
# https://storage.googleapis.com/xlnet/released_models/cased_L-24_H-1024_A-16.zip
# A TensorFlow checkpoint (xlnet_model.ckpt) containing the pre-trained weights (which is actually 3 files).
# A Sentence Piece model (spiece.model) used for (de)tokenization.
# A config file (xlnet_config.json) which specifies the hyperparameters of the model.

import sys
from absl import flags
#sys.path.insert(0, "c:/SAVE/Project/Yui/play/brain")
sys.path.insert(0, "./write/model/xlnet")
try:
    import xlnet
except ImportError:
    print('No import')

# some code omitted here...
# initialize FLAGS
# initialize instances of tf.Tensor, including input_ids, seg_ids, and input_mask
flags.DEFINE_string("model_config_path", default=None,
      help="Model config path.")
flags.DEFINE_float("dropout", default=0.1,
      help="Dropout rate.")
flags.DEFINE_float("dropatt", default=0.1,
      help="Attention dropout rate.")
flags.DEFINE_integer("clamp_len", default=-1,
      help="Clamp length.")
flags.DEFINE_string("summary_type", default="last",
      help="Method used to summarize a sequence into a vector.")
flags.DEFINE_bool("use_bfloat16", default=False,
      help="Whether to use bfloat16.")

FLAGS = flags.FLAGS
'''
# XLNetConfig contains hyperparameters that are specific to a model checkpoint.
xlnet_config = xlnet.XLNetConfig(json_path=FLAGS.model_config_path)

# RunConfig contains hyperparameters that could be different between pretraining and finetuning.
run_config = xlnet.create_run_config(is_training=True, is_finetune=True, FLAGS=FLAGS)

# Construct an XLNet model
xlnet_model = xlnet.XLNetModel(
    xlnet_config=xlnet_config,
    run_config=run_config,
    input_ids=input_ids,
    seg_ids=seg_ids,
    input_mask=input_mask)

# Get a summary of the sequence using the last hidden state
summary = xlnet_model.get_pooled_out(summary_type="last")

# Get a sequence output
seq_out = xlnet_model.get_sequence_output()
'''
# build your applications based on `summary` or `seq_out`
