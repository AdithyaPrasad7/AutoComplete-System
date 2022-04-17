
import math
import nltk
import random
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

with open("C:/Users/Adithya/Desktop/CD_PROJECT/en_US.twitter.txt", "r", encoding="utf8") as f:
    data = f.read()

tokenized = []
def preprocess_pipeline(data) -> 'list':

    # Split by newline character
    sentences = data.split('\n')
    
    # Remove leading and trailing spaces
    sentences = [s.strip() for s in sentences]
    
    # Drop Empty Sentences
    sentences = [s for s in sentences if len(s) > 0]
    
    # Empty List to hold Tokenized Sentences
    #tokenized = []
# Iterate through sentences
    for sentence in sentences:
        
        # Convert to lowercase
        sentence = sentence.lower()
        
        # Convert to a list of words
        token = nltk.word_tokenize(sentence)
        
        # Append to list
        tokenized.append(token)
    return tokenized

len(data)

tokenized_sentences = preprocess_pipeline(data)
#print(len(tokenized_sentences))
#print(tokenized_sentences)

print(len(tokenized_sentences))

train, test = train_test_split(tokenized_sentences, test_size=0.2, random_state=42)

train, val = train_test_split(train, test_size=0.25, random_state=42)

print(len(train))
print(len(test))

def count_the_words(sentences) -> 'dict':
    
  # Creating a Dictionary of counts
  word_counts = {}

  # Iterating over sentences
  for sentence in sentences:
    
    # Iterating over Tokens
    for token in sentence:
    
      # Add count for new word
      if token not in word_counts.keys():
        word_counts[token] = 1
        
      # Increase count by one
      else:
        word_counts[token] += 1
        
  return word_counts

def handling_oov(tokenized_sentences, count_threshold) -> 'list':

  # Empty list for closed vocabulary
  closed_vocabulary = []

  # Obtain frequency dictionary using previously defined function
  words_count = count_the_words(tokenized_sentences)
    
  # Iterate over words and counts 
  for word, count in words_count.items():
    
    # Append if it's more(or equal) to the threshold 
    if count >= count_threshold :
      closed_vocabulary.append(word)

  return closed_vocabulary

def unk_tokenize(tokenized_sentences, vocabulary, unknown_token = "<unk>") -> 'list':

  # Convert Vocabulary into a set
  vocabulary = set(vocabulary)

  # Create empty list for sentences
  new_tokenized_sentences = []
  
  # Iterate over sentences
  for sentence in tokenized_sentences:

    # Iterate over sentence and add <unk> 
    # if the token is absent from the vocabulary
    new_sentence = []
    for token in sentence:
      if token in vocabulary:
        new_sentence.append(token)
      else:
        new_sentence.append(unknown_token)
    
    # Append sentece to the new list
    new_tokenized_sentences.append(new_sentence)

  return new_tokenized_sentences

def cleansing(train_data, test_data, count_threshold):
    
  # Get closed Vocabulary
  vocabulary = handling_oov(train_data, count_threshold)
    
  # Updated Training Dataset
  new_train_data = unk_tokenize(train_data, vocabulary)
    
  # Updated Test Dataset
  new_test_data = unk_tokenize(test_data, vocabulary)

  return new_train_data, new_test_data, vocabulary

min_freq = 4
final_train, final_test, vocabulary = cleansing(train, test, min_freq)

def count_n_grams(data, n, start_token = "<s>", end_token = "<e>") -> 'dict':

  # Empty dict for n-grams
  n_grams = {}
 
  # Iterate over all sentences in the dataset
  for sentence in data:
        
    # Append n start tokens and a single end token to the sentence
    sentence = [start_token]*n + sentence + [end_token]
    
    # Convert the sentence into a tuple
    sentence = tuple(sentence)

    # Temp var to store length from start of n-gram to end
    m = len(sentence) if n==1 else len(sentence)-1
    
    # Iterate over this length
    for i in range(m):
        
      # Get the n-gram
      n_gram = sentence[i:i+n]
    
      # Add the count of n-gram as value to our dictionary
      # IF n-gram is already present
      if n_gram in n_grams.keys():
        n_grams[n_gram] += 1
      # Add n-gram count
      else:
        n_grams[n_gram] = 1
        
  return n_grams

def prob_for_single_word(word, previous_n_gram, n_gram_counts, nplus1_gram_counts, vocabulary_size, k = 1.0) -> 'float':

  # Convert the previous_n_gram into a tuple 
  previous_n_gram = tuple(previous_n_gram)
    
  # Calculating the count, if exists from our freq dictionary otherwise zero
  previous_n_gram_count = n_gram_counts[previous_n_gram] if previous_n_gram in n_gram_counts else 0
  
  # The Denominator
  denom = previous_n_gram_count + k * vocabulary_size

  # previous n-gram plus the current word as a tuple
  nplus1_gram = previous_n_gram + (word,)

  # Calculating the nplus1 count, if exists from our freq dictionary otherwise zero 
  nplus1_gram_count = nplus1_gram_counts[nplus1_gram] if nplus1_gram in nplus1_gram_counts else 0

  # Numerator
  num = nplus1_gram_count + k

  # Final Fraction
  prob = num / denom
  return prob

def probs(previous_n_gram, n_gram_counts, nplus1_gram_counts, vocabulary, k=1.0) -> 'dict':

  # Convert to Tuple
  previous_n_gram = tuple(previous_n_gram)

  # Add end and unknown tokens to the vocabulary
  vocabulary = vocabulary + ["<e>", "<unk>"]

  # Calculate the size of the vocabulary
  vocabulary_size = len(vocabulary)

  # Empty dict for probabilites
  probabilities = {}

  # Iterate over words 
  for word in vocabulary:
    
    # Calculate probability
    probability = prob_for_single_word(word, previous_n_gram, 
                                           n_gram_counts, nplus1_gram_counts, 
                                           vocabulary_size, k=k)
    # Create mapping: word -> probability
    probabilities[word] = probability

  return probabilities

def auto_complete(previous_tokens, n_gram_counts, nplus1_gram_counts, vocabulary, k=1.0, start_with=None):

    
    # length of previous words
    n = len(list(n_gram_counts.keys())[0]) 
    
    # most recent 'n' words
    previous_n_gram = previous_tokens[-n:]
    
    # Calculate probabilty for all words
    probabilities = probs(previous_n_gram,n_gram_counts, nplus1_gram_counts,vocabulary, k=k)

    # Intialize the suggestion and max probability
    suggestion = None
    max_prob = 0

    # Iterate over all words and probabilites, returning the max.
    # We also add a check if the start_with parameter is provided
    for word, prob in probabilities.items():
        
        if start_with != None: 
            
            if not word.startswith(start_with):
                continue 

        if prob > max_prob: 

            suggestion = word
            max_prob = prob

    return suggestion, max_prob

def get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, k=1.0, start_with=None):

    # See how many models we have
    count = len(n_gram_counts_list)
    
    # Empty list for suggestions
    suggestions = []
    
    # IMP: Earlier "-1"
    
    # Loop over counts
    for i in range(count-1):
        
        # get n and nplus1 counts
        n_gram_counts = n_gram_counts_list[i]
        nplus1_gram_counts = n_gram_counts_list[i+1]
        
        # get suggestions 
        suggestion = auto_complete(previous_tokens, n_gram_counts,
                                    nplus1_gram_counts, vocabulary,
                                    k=k, start_with=start_with)
        # Append to list
        suggestions.append(suggestion)
        
    return suggestions

n_gram_counts_list = []
for n in range(1, 6):
    n_model_counts = count_n_grams(final_train, n)
    n_gram_counts_list.append(n_model_counts)

previous_tokens = ["Hello", "how", "are"]
suggestion = get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, k=0.1)

display(suggestion)

[('you', 0.09292085657969465),
 ('you', 0.03495401751364819),
 ('lol', 1.92130341223486e-05),
 ('lol', 1.92130341223486e-05)]

d=set(suggestion)
print(d)



