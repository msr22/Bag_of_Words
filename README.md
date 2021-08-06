# Bag_of_Words
Implementation of Bag of Words 

#### Multiclass Text classification

The objective of this project is to learn how to use Bag of Words. It is a multi-class classification problem.

We will be using a dataset that consists of nearly 20000 messages taken from 20 newsgroups.

### Datasource
http://archive.ics.uci.edu/ml/datasets/Twenty+Newsgroups

#### Keywords

* Numpy
* Collections
* Gensim
* Bag-of-Words (Word Frequency, Pre-Processing)
* Bag-of-Words representation

### Data Preparation
- To get a sense of our data, we count the frequencies of the target classes in our news articles in the training set.
- Next, we split our dataset which consists of about 1000 samples per class, into training and test sets. 
- We use about 95% samples from each class in the training set, and the remaining in the test set.

## Bag-of-Words
- ML algorithms need good feature representations of different inputs. Concretely, we would like to represent each news article  D  in terms of a feature vector  V , which can be used for classification. Feature vector  V  is made up of the number of occurences of each word in the vocabulary.

- So, we count the number of occurences (word frequency) of every word in the news documents in the training set.
- We list down the 10 most frequent words & 10 least frequent words.
- Then we plot a histogram of the counts of various words in descending order.

![Alt Text](https://github.com/msr22/Bag_of_Words/blob/main/Figure_1_BoW_count_of_words.png)

## Then we do pre-processing to remove most (25) and least (100) frequent words
- Number of words before preprocessing:  89599
- Number of words after preprocessing:  4096

### Bag-of-Words representation

- The simplest way to represent a document *D* as a vector *V* would be to now count the relevant words in the document. 
- For each document, make a vector of the count of each of the words in the vocabulary (excluding the words removed in the previous step - the "stopwords").

### Document classification using Bag-of-Words

- For the test documents, we use Euclidean distance metric to find similar news articles from your training set and classify using kNN.
- Then we compute the accuracy for the bag-of-words features on the full test set.

