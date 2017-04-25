# -*- coding: utf-8 -*-
from keras.layers import Input
from keras.layers.core import Dense
from keras.models import Model
from keras.layers.core import Activation

inputs = Input(shape=(784,))

x = Dense(32)(inputs)
x = Activation("sigmoid")(x)
x = Dense(10)(x)
predictions = Activation("softmax")(x)

model = Model(inputs=inputs, outputs=predictions)

model.compile(loss="categorical_crossentropy", optimizer="adam")
