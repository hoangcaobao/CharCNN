from model import *
from clean import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import json
from argparse import ArgumentParser

if __name__=="__main__":
    
    parser=ArgumentParser()
    parser.add_argument("--data-path", default="data/sarcasm.json", type=str)
    parser.add_argument("--learning-rate", default=0.0001, type=float)
    parser.add_argument("--test-size", default=0.2, type=float)
    parser.add_argument("--embedding-size", default=100, type=int)
    parser.add_argument("--num-classes", default=2, type=int)
    parser.add_argument("--small-model", default="model/small", type=str)
    parser.add_argument("--large-model", default="model/large", type=str)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    args=parser.parse_args()

    print('---------------------Welcome to CharCNN-------------------')
    print('Github: hoangcaobao')
    print('Email: caobaohoang03@gmail.com')
    print('----------------------------------------------------------')
    print('Training CharCNN model with hyper-params:')
    print('===========================')
    for i, arg in enumerate(vars(args)):
        print('{}.{}: {}'.format(i, arg, vars(args)[arg]))
    print('===========================')
    
    #Preprocessing data
    sentences=[]
    labels=[]
    
    with open(args.data_path, "r") as r:
        data_store=json.load(r)
    for data in data_store:
        sentences.append(data["headline"])
        labels.append(data["is_sarcastic"])
    
    clean_data=CleanData(sentences, labels, args.test_size)
    clean_data.preprocessing_data() 
    
    #Train Small CharCNN
    small_CharCNN=CharCNN(256, 1024, args.embedding_size, clean_data.vocab_size, clean_data.max_len, args.num_classes)
    small_CharCNN.compile(optimizer=Adam(args.learning_rate), loss="sparse_categorical_crossentropy", metrics=["acc"])
    small_CharCNN.fit(clean_data.x_train, clean_data.y_train, validation_data=(clean_data.x_valid, clean_data.y_valid), epochs=args.epochs)
    small_CharCNN.save(args.small_model)
    
    #Train Large CharCNN
    large_CharCNN=CharCNN(1024, 2048, args.embedding_size, clean_data.vocab_size, clean_data.max_len, args.num_classes)
    large_CharCNN.compile(optimizer=Adam(args.learning_rate), loss="sparse_categorical_crossentropy", metrics=["acc"])
    large_CharCNN.fit(clean_data.x_train, clean_data.y_train, validation_data=(clean_data.x_valid, clean_data.y_valid), epochs=args.epochs, batch_size=args.batch_size)
    large_CharCNN.save(args.large_model)
