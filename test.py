import math
import nltk
import random
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

with open("C:/Users/Adithya/Desktop/CD_PROJECT/en_US.twitter.txt", "r", encoding="utf8") as f:
    data = f.read()


def preprocess_pipeline(data) -> 'list':

    # Split by newline character
    sentences = data.split('\n')
    
    # Remove leading and trailing spaces
    sentences = [s.strip() for s in sentences]
    
    # Drop Empty Sentences
    sentences = [s for s in sentences if len(s) > 0]
    
    # Empty List to hold Tokenized Sentences
    tokenized = []
# Iterate through sentences
'''for sentence in sentences:
        
        # Convert to lowercase
        sentence = sentence.lower()
        
        # Convert to a list of words
        token = nltk.word_tokenize(sentence)
        
        # Append to list
        tokenized.append(token)
        print(token)
    return tokenized'''
#Pass our data to this function    
tokenized_sentences = preprocess_pipeline(data)

#print(tokenized_sentences)