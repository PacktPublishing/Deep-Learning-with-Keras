# -*- coding: utf-8 -*-
from keras.layers import Input
from keras.layers.core import Dense
from keras.models import Model
from keras.layers.core import Activation

input1 = Input(shape=(784,))
input2 = Input(shape=(784,))

x = Dense(32)(input1)
x = Activation("sigmoid")(x)
x = Dense(10)(x)
output1 = Activation("softmax")(x)

x = Dense(32)(input2)
x = Activation("sigmoid")(x)
x = Dense(10)(x)
output2 = Activation("softmax")(x)

model = Model(inputs=[input1, input2], outputs=[output1, output2])

model.compile(loss="categorical_crossentropy", optimizer="adam")
