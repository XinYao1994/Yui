https://github.com/OpenNMT/OpenNMT-tf/blob/master/scripts/wmt/README.md

http://data.statmt.org/news-commentary/v14/training/

### Installing SentencePiece

NMT models perform better if words are represented as sub-words, since this helps the out-of-vocabulary problem. [SentencePiece](https://arxiv.org/pdf/1808.06226.pdf) is a powerful end-to-end tokenizer that allows the learning of subword units from raw data. [We will install SentencePiece from source](https://github.com/google/sentencepiece#c-from-source) rather than via `pip install`, since the `spm_train` command used for training a SentencePiece model is not installed via pip but has to be built from the C++.

Installation instructions are available [here](https://github.com/google/sentencepiece#c-from-source). 








