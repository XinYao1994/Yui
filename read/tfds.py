import tensorflow as tf
print(tf.__version__)
import tensorflow_datasets as tfds

# subwords8k is supported in tf2.0
# token from https://www.tensorflow.org/datasets/api_docs/python/tfds/features/text/SubwordTextEncoder

imdb, info = tfds.load("imdb_reviews/subwords8k", with_info=True, as_supervised=True)
train_data, test_data = imdb['train'], imdb['test']

tokenizer = info.features['text'].encoder
#print(tokenizer.subwords)

sample_string = "TensorFlow, from basics to mastery"
tokenized_string = tokenizer.encode(sample_string)
print("tok : {}".format(tokenized_string))

ori_string = tokenizer.decode(tokenized_string)
print("ori : {}".format(ori_string))

embedding_dim = 64

import keras

model = keras.Sequential([
    keras.layers.Embedding(tokenizer.vocab_size, embedding_dim),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(6, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model.summary()

num_epochs = 10

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

# test with 2.0 please, 'DatasetV1Adapter' object has no attribute 'ndim' in this version
history = model.fit(train_data, epochs=num_epochs, validation_data=test_data)


