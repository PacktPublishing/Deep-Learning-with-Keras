# -*- coding: utf-8 -*-
from __future__ import division, print_function
from keras.layers.core import Activation, Dense, Dropout, RepeatVector, SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU, LSTM
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.utils import np_utils
import collections
import matplotlib.pyplot as plt
import numpy as np
import os

def explore_data(datadir, datafiles):
    counter = collections.Counter()
    maxlen = 0
    for datafile in datafiles:
        fdata = open(os.path.join(datadir, datafile), "rb")
        for line in fdata:
            words = line.strip().split()
            if len(words) > maxlen:
                maxlen = len(words)
            for word in words:
                counter[word] += 1
        fdata.close()
    return maxlen, counter
    
def build_tensor(filename, numrecs, word2index, maxlen, 
                 make_categorical=False):
    data = np.empty((numrecs, ), dtype=list)
    fin = open(filename, "rb")
    i = 0
    for line in fin:
        wids = []
        for word in line.strip().split():
            if word2index.has_key(word):
                wids.append(word2index[word])
            else:
                wids.append(word2index["UNK"])
        if make_categorical:
            data[i] = np_utils.to_categorical(
                wids, num_classes=len(word2index))
        else:
            data[i] = wids
        i += 1
    fin.close()
    pdata = sequence.pad_sequences(data, maxlen=maxlen)
    return pdata
    
def evaluate_model(model, Xtest, Ytest, batch_size):
    pass

DATA_DIR = "../data"

s_maxlen, s_counter = explore_data(DATA_DIR, ["babi-sent-train.txt", 
                                              "babi-sent-test.txt"])
t_maxlen, t_counter = explore_data(DATA_DIR, ["babi-pos-train.txt", 
                                              "babi-pos-test.txt"])

print(s_maxlen, len(s_counter), t_maxlen, len(t_counter))
# 7 21 7 9
# maxlen: 7
# size of source vocab: 21
# size of target vocab: 9

# lookup tables
s_word2id = {k:v+1 for v, (k, _) in enumerate(s_counter.most_common())}
s_word2id["PAD"] = 0
s_id2word = {v:k for k, v in s_word2id.items()}
t_pos2id = {k:v+1 for v, (k, _) in enumerate(t_counter.most_common())}
t_pos2id["PAD"] = 0
t_id2pos = {v:k for k, v in t_pos2id.items()}

# vectorize data
MAX_SEQLEN = 10

Xtrain = build_tensor(os.path.join(DATA_DIR, "babi-sent-train.txt"),
                      30000, s_word2id, MAX_SEQLEN)
Xtest = build_tensor(os.path.join(DATA_DIR, "babi-sent-test.txt"),
                     3000, s_word2id, MAX_SEQLEN)
Ytrain = build_tensor(os.path.join(DATA_DIR, "babi-pos-train.txt"),
                      30000, t_pos2id, MAX_SEQLEN, make_categorical=True)
Ytest = build_tensor(os.path.join(DATA_DIR, "babi-pos-test.txt"),
                     3000, t_pos2id, MAX_SEQLEN, make_categorical=True)
print(Xtrain.shape, Xtest.shape, Ytrain.shape, Ytest.shape)

# define network
EMBED_SIZE = 32
HIDDEN_SIZE = 32

BATCH_SIZE = 32
NUM_EPOCHS = 5

model = Sequential()
model.add(Embedding(len(s_word2id), EMBED_SIZE,
                    input_length=MAX_SEQLEN))
model.add(SpatialDropout1D(Dropout(0.2)))
model.add(LSTM(HIDDEN_SIZE, dropout=0.2, recurrent_dropout=0.2))
#model.add(GRU(HIDDEN_SIZE, dropout=0.2, recurrent_dropout=0.2))
#model.add(Bidirectional(LSTM(HIDDEN_SIZE, dropout=0.2, recurrent_dropout=0.2)))
model.add(RepeatVector(MAX_SEQLEN))
model.add(LSTM(HIDDEN_SIZE, return_sequences=True))
#model.add(GRU(HIDDEN_SIZE, return_sequences=True))
#model.add(Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True)))
model.add(TimeDistributed(Dense(len(t_pos2id))))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam",
             metrics=["accuracy"])
             
history = model.fit(Xtrain, Ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,
                    validation_data=[Xtest, Ytest])

# plot loss and accuracy
plt.subplot(211)
plt.title("Accuracy")
plt.plot(history.history["acc"], color="g", label="Train")
plt.plot(history.history["val_acc"], color="b", label="Validation")
plt.legend(loc="best")

plt.subplot(212)
plt.title("Loss")
plt.plot(history.history["loss"], color="g", label="Train")
plt.plot(history.history["val_loss"], color="b", label="Validation")
plt.legend(loc="best")

plt.tight_layout()
plt.show()
                    
# evaluate model
score, acc = model.evaluate(Xtest, Ytest, batch_size=BATCH_SIZE)
print("Test score: %.3f, accuracy: %.3f" % (score, acc))

# custom evaluate
hit_rates = []
num_iters = Xtest.shape[0] // BATCH_SIZE
for i in range(num_iters - 1):
    xtest = Xtest[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
    ytest = np.argmax(Ytest[i * BATCH_SIZE : (i + 1) * BATCH_SIZE], axis=2)
    ytest_ = np.argmax(model.predict(xtest), axis=2)
#    print(ytest.shape, ytest_.shape)
    for j in range(BATCH_SIZE):
#        print("sentence:  " + " ".join([s_id2word[x] for x in xtest[j].tolist()]))
#        print("predicted: " + " ".join([t_id2pos[y] for y in ytest_[j].tolist()]))
#        print("label:     " + " ".join([t_id2pos[y] for y in ytest[j].tolist()]))
        word_indices = np.nonzero(xtest[j])
        pos_labels = ytest[j][word_indices]
        pos_pred = ytest_[j][word_indices]
        hit_rates.append(np.sum(pos_labels == pos_pred) / len(pos_pred))
    break

accuracy = sum(hit_rates) / len(hit_rates)
print("accuracy: {:.3f}".format(accuracy))        

# prediction
pred_ids = np.random.randint(0, 3000, 5)
for pred_id in pred_ids:
    xtest = Xtest[pred_id].reshape(1, 10)
    ytest_ = np.argmax(model.predict(xtest), axis=2)
    ytest = np.argmax(Ytest[pred_id], axis=1)
    print("sentence:  " + " ".join([s_id2word[x] for x in xtest[0].tolist()]))
    print("predicted: " + " ".join([t_id2pos[y] for y in ytest_[0].tolist()]))
    print("label:     " + " ".join([t_id2pos[y] for y in ytest.tolist()]))
    word_indices = np.nonzero(xtest)[1]
    ypred_tags = ytest_[0][word_indices]
    ytrue_tags = ytest[word_indices]
    hit_rate = np.sum(ypred_tags == ytrue_tags) / len(ypred_tags)
    print("hit rate: {:.3f}".format(hit_rate))
    print()
