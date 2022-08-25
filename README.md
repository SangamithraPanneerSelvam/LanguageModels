# LanguageModels

A language model is a probability distribution over sequences of words. Given such a sequence of length m, a language model assigns a probability P to the whole sequence. Language models generate probabilities by training on text corpora in one or many languages.

Here we have 

### Machine learning model:
Machine learning (ML) for natural language processing (NLP) and text analytics involves using machine learning algorithms and “narrow” artificial intelligence (AI) to understand the meaning of text documents.

### Neural network model:
These language models are based on neural networks and are often considered as an advanced approach to execute NLP tasks. Neural language models overcome the shortcomings of classical models such as n-gram and are used for complex tasks such as speech recognition or machine translation. 

### Bert model:
Bidirectional Encoder Representations from Transformers is a transformer-based machine learning technique for natural language processing pre-training developed by Google.


The techniques we use:

### Setup: import packages, read data, Preprocessing, Partitioning.

### Bag-of-Words: Feature Engineering & Feature Selection & Machine Learning with scikit-learn, Testing & Evaluation, Explainability with lime.

### Word Embedding: Fitting a Word2Vec with gensim, Feature Engineering, Hybrid model with LSTM layer and Deep Learning with tensorflow/keras, Testing & Evaluation.

### Language Models: Feature Engineering with transformers, Fine tunning a pre-trained BERT with transformers and tensorflow/keras, Testing & Evaluation.

## The dataset:

![alt text]("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQlORDobV7y5o8XHzdiVy9qNMF-b9pY6soi2rQi4GcltJu_NrtdotikUIN8rLXmNPlVsik&usqp=CAU")

I love superheros ! especially I am a big Marvel fan. So I want to do something interesting for language models and I picked up  the superhero dataset from kaggle. You can do many things with this dataset! But here I am trying to use it for text classification.
I want to see based on the history description of superheros, if I can find who their creators are! 
