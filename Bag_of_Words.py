# Importing required Packages
import pickle
import re
import operator
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import math
import collections
import gensim

import zipfile
with zipfile.ZipFile("20_NEWSGROUPS_PICKELFILE.pkl.zip", 'r') as zip_ref:
    zip_ref.extractall()

dataset = pickle.load(open('AIML_DS_NEWSGROUPS_PICKELFILE.pkl','rb'))
print(type(dataset))
print(dataset.keys())

# Print frequencies of dataset
print("Class : count")
print("--------------")
number_of_documents = 0
for key in dataset:
    print(key, ':', len(dataset[key]))

# Split dataset
train_set = {}
test_set = {}

# Clean dataset for text editing issues Very useful when dealing with non- unicode characters

for key in dataset:
    dataset[key] = [[i.decode('utf-8', errors='replace').lower() for i in f] for f in dataset[key]]

# Break dataset into 95-5 split for training & testing (for each key)

n_train = 0
n_test = 0
for k in dataset:
    split = int(0.95*len(dataset[k]))
    train_set[k] = dataset[k][0:split]
    test_set[k] = dataset[k][split:-1]
    n_train += len(train_set[k])
    n_test += len(test_set[k])

# Initialize a dictionary to store frequencies of words.
# Key:Value === Word:Count

frequency = defaultdict(int)

for key in train_set:
    for f in train_set[key]:

        # Find all words which consist only of capital and lowercase characters and are between length of 2-9.
        # We ignore all special characters such as !.$ and words containing numbers

        words = re.findall(r'(\b[A-Za-z][a-z]{2,9}\b)', ' '.join(f))

        for word in words:
            frequency[word] += 1

sorted_words = sorted(frequency.items(), key=operator.itemgetter(1), reverse=True)
print("Top-10 most frequent words:")
for word in sorted_words[:10]:
    print(word)

print('----------------------------')
print("10 least frequent words:")
for word in sorted_words[-10:-1]:
    print(word)

# plot a histogram of the counts of various words in descending order.

# %matplotlib inline 

fig = plt.figure()
# set figure size in inches
fig.set_size_inches(20,10)

# make a bar plot
plt.bar(range(len(sorted_words[:100])), [v for k,v in sorted_words[:100]], align='center')
# get the ticks location & labels to the x-axis
plt.xticks(range(len(sorted_words[:100])), [k for k,v in sorted_words[:100]]) # returns locs & labels
locs, labels = plt.xticks()     
# rotate the labels by 90 degrees
plt.setp(labels, rotation=90) 

# set yticks at a size of 10000
y = np.random.randint(low=0, high=max([v for k,v in sorted_words[:100]]), size=10000)
plt.yticks(np.arange(0, max(y), 10000))

plt.show()


valid_words = defaultdict(int)

print('Number of words before preprocessing: ', len(sorted_words))

# Ignore the 25 most frequent words, and the words which appear less than 100 times

ignore_most = 25
ignore_least = 100

feature_number = 0

for word, word_frequency in sorted_words[ignore_most:]:
    if word_frequency > ignore_least:
        valid_words[word] = feature_number
        feature_number += 1

print('Number of words after preprocessing: ', len(valid_words))

word_vector_size = len(valid_words)

def convert_to_BoW(dataset, number_of_documents):
    bow_representation = np.zeros((number_of_documents, word_vector_size))
    labels = np.zeros((number_of_documents, 1))

    i = 0
    for label, class_name in enumerate(dataset):
        
        # For each file
        for f in dataset[class_name]:
            
            # Read all text in file
            text = ' '.join(f).split(' ')
            
            # For each word
            for word in text:
                if word in valid_words:
                    bow_representation[i, valid_words[word]] += 1
            
            # Label of document
            labels[i] = label
            
            # Increment document counter
            i += 1
    
    return bow_representation, labels

# Convert the dataset into their bag of words representation treating train and test separately
train_bow_set, train_bow_labels = convert_to_BoW(train_set, n_train)
test_bow_set, test_bow_labels = convert_to_BoW(test_set, n_test)

# Document classification using Bag-of-Words

def dist(train_features, given_feature):
    squared_difference = (train_features - given_feature)**2
    distances = np.sqrt(np.sum(squared_difference, axis = 1))
    return distances


def kNN(k, train_features, train_labels, given_feature):
    distances = []
    
    n = train_features.shape[0]
    
    # np.tile function repeats the given_feature n times.
    given_feature = np.tile(given_feature, (n, 1))
    
    # Compute distance
    distances = dist(train_features, given_feature)
    sort_neighbors = np.argsort(distances)
    return np.concatenate((distances[sort_neighbors][:k].reshape(-1, 1), train_labels[sort_neighbors][:k].reshape(-1, 1)), axis = 1)

def kNN_classify(k, train_features, train_labels, given_feature):
    tally = collections.Counter()
    tally.update(str(int(nn[1])) for nn in kNN(k, train_features, train_labels, given_feature))
    return int(tally.most_common(1)[0][0])

kNN_classify(3, train_bow_set, train_bow_labels, test_bow_set[0])

accuracy = 0
for i, given_feature in enumerate(test_bow_set):
    print("Progress: {0:.04f}".format((i+1)/len(test_bow_set)), end="\r")
    predicted_class = kNN_classify(3, train_bow_set, train_bow_labels, given_feature)
    if predicted_class == int(test_bow_labels[i]):
        accuracy += 1
print("Accuracy: ", accuracy / len(test_bow_set))