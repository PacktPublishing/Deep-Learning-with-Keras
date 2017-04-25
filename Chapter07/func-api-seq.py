# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers.core import Dense, Activation

model = Sequential([
    Dense(32, input_dim=784),
    Activation("sigmoid"),
    Dense(10),
    Activation("softmax"),
])
model.compile(loss="categorical_crossentropy", optimizer="adam")
