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

- To get a sense of our data, we count the frequencies of the target classes in our news articles in the training set.
- Next, we split our dataset which consists of about 1000 samples per class, into training and test sets. 
- We use about 95% samples from each class in the training set, and the remaining in the test set.

## 1. Bag-of-Words
- ML algorithms need good feature representations of different inputs. Concretely, we would like to represent each news article  D  in terms of a feature vector  V , which can be used for classification. Feature vector  V  is made up of the number of occurences of each word in the vocabulary.

- So, we count the number of occurences (word frequency) of every word in the news documents in the training set.
- We list down the 10 most frequent words & 10 least frequent words.
- Then we plot a histogram of the counts of various words in descending order.



