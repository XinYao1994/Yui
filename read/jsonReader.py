'''
import os

with open('./read/data/Sarcasm_Headlines_Dataset.json', 'r') as lines:
     with open('./read/data/sarcasm.json', 'w') as outfile:
        for line in lines:
            line = line.replace(os.linesep, "") + ','
            outfile.write(line)

exit()
'''
import json

with open("./read/data/sarcasm.json") as f:
    datastore = json.load(f)

sentences = []
labels = []
urls = []

for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)

padded = pad_sequences(sequences, padding='post')

print(sentences[2])
print(padded[2])
print(padded.shape)


