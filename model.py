from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import *

class CharCNN(tf.keras.models.Model):
    def __init__(self, feature, units, embedding_size, vocab_size, input_length, num_classes):
        super(CharCNN, self).__init__()
        self.embedding=Embedding(vocab_size, embedding_size, input_length=input_length)
        self.convnets=Sequential([
            Conv1D(feature, 7, activation="relu"),
            MaxPooling1D(3),
            Conv1D(feature, 7, activation="relu"),
            MaxPooling1D(3),
            Conv1D(feature, 3, activation="relu"),
            Conv1D(feature, 3, activation="relu"),
            Conv1D(feature, 3, activation="relu"),
            Conv1D(feature, 3, activation="relu"),
            MaxPooling1D(3),
            Flatten()
        ])
      
        self.fully_connected_layers=Sequential([
            Dense(units, activation="relu"),
            Dropout(0.5),
            Dense(units, activation="relu"),
            Dropout(0.5),
            Dense(num_classes, activation="softmax")
        ])

    def call(self, sentence):
        embedded_sentence=self.embedding(sentence)
        convnets_result=self.convnets(embedded_sentence)
        return self.fully_connected_layers(convnets_result)
