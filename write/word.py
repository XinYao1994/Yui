import tensorflow as tf

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Bidirectional
from keras.models import Sequential
from keras.optimizers import Adam

import numpy as np
tokenizer = Tokenizer()

'''
In the town of Athy one Jeremy Lanigan \n Battered away til he hadn't a pound. \n His father he died and made him a man again \n Left him a farm and ten acres of ground. \n He gave a grand party to friends and relations \n Who didnt forget him when it comes to the will, \n If youll but listen Ill make your eyes glisten \n Of the rows and the ructions of Lanigan's Ball. \n Myself to be sure got free invitation, \n For all the nice girls and boys I might ask, \n And just in a minute both friends and relations \n Were dancing round merry as bees round a cask. \n Judy ODaly, that nice little milliner, \n She tipped me a wink for to give her a call, \n And I soon arrived with Peggy McGilligan \n Just in time for Lanigans Ball. \n There were lashings of punch and wine for the ladies, \n Potatoes and cakes; there was bacon and tea, \n There were the Nolans, Dolans, OGradys \n Courting the girls and dancing away. \n Songs they went round as plenty as water, \n The harp that once sounded in Tara's old hall, \n Sweet Nelly Gray and "The Rat Catcher's Daughter," \n All singing together at Lanigan's Ball. \n They were doing all kinds of nonsensical polkas \n All round the room in a whirligig. \n Julia and I, we banished their nonsense \n And tipped them the twist of a reel and a jig. \n Och mavrone, how the girls got all mad at me \n Danced til youd think the ceiling would fall. \n For I spent three weeks at Brooks' Academy \n Learning new steps for Lanigan's Ball. \n Boys were all merry and the girls they were hearty \n And danced all around in couples and groups, \n Til an accident happened, young Terrance McCarthy \n Put his right leg through Miss Finnerty's hoops. \n Poor creature fainted and cried, Meelia murther, \n Called for her brothers and gathered them all. \n Carmody swore that hed go no further \n Til he had satisfaction at Lanigan's Ball. \n In the midst of the row miss Kerrigan fainted, \n Her cheeks at the same time as red as a rose. \n Some of the lads declared she was painted, \n She took a small drop too much, I suppose. \n Her sweetheart, Ned Morgan, so powerful and able, \n When he saw his fair colleen stretched out by the wall, \n Tore the left leg from under the table \n And smashed all the Chaneys at Lanigan's Ball. \n Boys, oh boys, twas then there were runctions. \n Myself got a lick from big Phelim McHugh. \n I soon replied to his introduction \n And kicked up a terrible hullabaloo. \n Ould Casey, the piper, was near being strangled. \n They squeezed up his pipes, bellows, chanters and all. \n The girls, in their ribbons, they got all entangled \n And that put an end to Lanigan's Ball.
'''

# data = "In the town of Athy one Jeremy Lanigan \n Battered away til he hadn't a pound. \n His father he died and made him a man again \n Left him a farm and ten acres of ground. \n He gave a grand party to friends and relations \n Who didnt forget him when it comes to the will, \n If youll but listen Ill make your eyes glisten \n Of the rows and the ructions of Lanigan's Ball. \n Myself to be sure got free invitation, \n For all the nice girls and boys I might ask, \n And just in a minute both friends and relations \n Were dancing round merry as bees round a cask. \n Judy ODaly, that nice little milliner, \n She tipped me a wink for to give her a call, \n And I soon arrived with Peggy McGilligan \n Just in time for Lanigans Ball. \n There were lashings of punch and wine for the ladies, \n Potatoes and cakes; there was bacon and tea, \n There were the Nolans, Dolans, OGradys \n Courting the girls and dancing away. \n Songs they went round as plenty as water, \n The harp that once sounded in Tara's old hall, \n Sweet Nelly Gray and The Rat Catcher's Daughter, \n All singing together at Lanigan's Ball. \n They were doing all kinds of nonsensical polkas \n All round the room in a whirligig. \n Julia and I, we banished their nonsense \n And tipped them the twist of a reel and a jig. \n Och mavrone, how the girls got all mad at me \n Danced til youd think the ceiling would fall. \n For I spent three weeks at Brooks' Academy \n Learning new steps for Lanigan's Ball. \n Boys were all merry and the girls they were hearty \n And danced all around in couples and groups, \n Til an accident happened, young Terrance McCarthy \n Put his right leg through Miss Finnerty's hoops. \n Poor creature fainted and cried, Meelia murther, \n Called for her brothers and gathered them all. \n Carmody swore that hed go no further \n Til he had satisfaction at Lanigan's Ball. \n In the midst of the row miss Kerrigan fainted, \n Her cheeks at the same time as red as a rose. \n Some of the lads declared she was painted, \n She took a small drop too much, I suppose. \n Her sweetheart, Ned Morgan, so powerful and able, \n When he saw his fair colleen stretched out by the wall, \n Tore the left leg from under the table \n And smashed all the Chaneys at Lanigan's Ball. \n Boys, oh boys, twas then there were runctions. \n Myself got a lick from big Phelim McHugh. \n I soon replied to his introduction \n And kicked up a terrible hullabaloo. \n Ould Casey, the piper, was near being strangled. \n They squeezed up his pipes, bellows, chanters and all. \n The girls, in their ribbons, they got all entangled \n And that put an end to Lanigan's Ball."
# data = "With pursuing high accuracy on big datasets, current research prefers designing complex neural networks, which need to maximize data parallelism for short training time. Many distributed deep learning systems, such as MxNet and Petuum, widely use parameter server framework with relaxed synchronization models. Although these models could cost less on each synchronization, its frequency is still high among many workers, e.g., the soft barrier introduced by Stale Synchronous Parallel (SSP) model. In this paper, we introduce our parameter server design, namely FluentPS, which can reduce frequent synchronization and optimize communication overhead in a large-scale cluster. Different from using a single scheduler to manage all parameters' synchronization in some previous designs, our system allows each server to independently adjust schemes for synchronizing its own parameter shard and overlaps the push and pull processes of different servers. We also explore two methods to improve SSP model: (1) lazy execution of buffered pull requests to reduce the synchronization frequency and (2) a probability-based strategy to pause the fast worker at a probability under SSP condition, which avoids unnecessary waiting of fast workers. We evaluate ResNet-56 with the same large batch size at different cluster scales. While guaranteeing robust convergence, FluentPS gains up to 6x speedup and reduce 93.7% communication costs than PS-Lite. The raw SSP model causes up to 131x delayed pull requests than our improved synchronization model, which can provide fine-tuned staleness controls and achieve higher accuracy."

data = open("./write/data/paper.txt").read()

corpus = data.lower().split("\n")

tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

print(tokenizer.word_index)
print(total_words)

input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)
max_sequence_len = max(len(x) for x in input_sequences)
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

xs, labels = input_sequences[:, :-1], input_sequences[:, -1]
import keras
ys = keras.utils.to_categorical(labels, num_classes=total_words)

print(xs[6])
print(labels[6])
print(ys[6])
'''
XLNet
BERT

model = Sequential()
model.add(Embedding(total_words, 64, input_length=max_sequence_len-1))
model.add(Bidirectional(LSTM(20)))
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
'''
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
model.add(Bidirectional(LSTM(150)))
model.add(Dense(total_words, activation='softmax'))
adam = Adam(lr=0.01)

model.compile(loss='categorical_crossentropy',optimizer=adam, metrics=['accuracy'])
history = model.fit(xs, ys, epochs=100, verbose=1)

import matplotlib.pyplot as plt

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.show()

# plot_graphs(history, 'acc')

seed_test = "We have a idea to optimize the quorum based system."
next_words = 100

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_test])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict_classes(token_list, verbose=0)
    output_word = ""
    print(predicted)
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            print(word)
            seed_test = seed_test + " " + word
            break
        #seed_test += " " + output_word

print(seed_test)

