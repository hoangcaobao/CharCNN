from train import *
from argparse import ArgumentParser
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import numpy as np
import tensorflow as tf

if __name__=="__main__":
    parser=ArgumentParser()
    parser.add_argument("--data-path", default="data/sarcasm.json", type=str)
    parser.add_argument("--small-model", default="model/small", type=str)
    parser.add_argument("--large-model", default="model/large", type=str)
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--test-size", default=0.2, type=float)
    parser.add_argument("--test-path", default="test/sentences.json", type=str)
    parser.add_argument("--result-file-path", default="result/result.json", type=str)
    args=parser.parse_args()

    print('---------------------Welcome to CharCNN-------------------')
    print('Github: hoangcaobao')
    print('Email: caobaohoang03@gmail.com')
    print('----------------------------------------------------------')
    print("""
    What model do you want to use to predict
    1. Small CharCNN
    2. Large CharCNN
    """)
    option=input("Your option (1/2): ")
    check=0
    if option=="1":
        type_model="Small CharCNN"
        check=1
    if option=="2":
        type_model="Large CharCNN"
        check=1
    if check==1:
        print('Predict using {} for image path: {}'.format(type_model, args.test_path))
        print('===========================')
        
        
        #Get token from train dataset
        sentences=[]
        labels=[]
        with open(args.data_path, "r") as r:
            data_store=json.load(r)
        for data in data_store:
            sentences.append(data["headline"])
            labels.append(data["is_sarcastic"])
        
        clean_data=CleanData(sentences, labels, args.test_size)
        clean_data.preprocessing_data()

        #Preprocess data
        sentences=[]
        with open(args.test_path, "r") as r:
            test_store=json.load(r)
        for test in test_store:
            sentences.append(test)
        print(sentences)
        sequences=clean_data.token.texts_to_sequences(sentences)
        sequences=np.array(sequences)
        sequences=pad_sequences(sequences, clean_data.max_len, padding="post")
        
        #Choose model
        if(option=="1"):
            model=tf.keras.models.load_model(args.small_model)
        if(option=="2"):
            model=tf.keras.models.load_model(args.large_model)

        #Saving
        result=np.argmax(model.predict(sequences), axis=1)
        result=result.astype("str")
        result=[int(i) for i in result]
        jsonString=json.dumps(result)
        jsonFile=open(args.result_file_path, "w")
        jsonFile.write(jsonString)
        jsonFile.close()
        print("Results have been saved in path: {}".format(args.result_file_path))
    
    else:
        print("You can only choose 1 or 2")

