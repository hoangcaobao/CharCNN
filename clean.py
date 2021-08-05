from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.model_selection import train_test_split

class CleanData: 
    def __init__ (self, sentences, labels, test_size):
        self.sentences=sentences
        self.labels=np.array(labels)
        self.test_size=test_size
    
    def find_max_len(self):
        self.len_sentences=[len(i) for i in self.sentences]
        self.max_len=max(self.len_sentences)
    
    def split_data(self):
        self.x_train, self.x_valid, self.y_train, self.y_valid=train_test_split(self.sentences, self.labels, test_size=self.test_size, shuffle=False)
        self.vocab_size=len(self.x_train)
    
    def tokenize(self):
        self.token=Tokenizer(self.vocab_size, oov_token="<OOV>")
        self.token.fit_on_texts(self.x_train)

    def pad_data(self):
        self.x_train=self.token.texts_to_sequences(self.x_train)
        self.x_valid=self.token.texts_to_sequences(self.x_valid)
        self.x_train=pad_sequences(self.x_train, maxlen=self.max_len, padding="post")
        self.x_valid=pad_sequences(self.x_valid, maxlen=self.max_len, padding="post")
        self.x_train=np.array(self.x_train)
        self.x_valid=np.array(self.x_valid)

    def preprocessing_data(self):
        self.find_max_len()
        self.split_data()
        self.tokenize()
        self.pad_data()
       
