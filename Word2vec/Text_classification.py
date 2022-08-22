import numpy as np
import tensorflow
import matplotlib.pyplot as plt
import pandas as pd
import nltk
import re
from sklearn import feature_extraction, model_selection, naive_bayes, pipeline, preprocessing

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn import metrics
import nltk 
import gensim
from gensim import models
import gensim.downloader as gensim_api
import tensorflow
from tensorflow.keras import models,layers,preprocessing 
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import EarlyStopping
# nltk.download('stopwords')
from sklearn.preprocessing import LabelEncoder


# nltk.download('wordnet')
# nltk.download('omw-1.4')

class superheros():

  def __init__(self):
    self.doc_list=[]
    self.doc_test_list = []
    self.load()
    

  def load(self):
    dataset=pd.read_csv("superheroes_nlp_dataset.csv")
    self.classify_dataset=dataset[['history_text','creator']]
  
  def preprocess(self,text,stop_words,stemming=False,lemantizing=False):
  
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    #tokenize
    text_list=text.split()
    # text_list=[str(word) for word in text_list]
    #stopwords
    if stop_words:
      text_list=[word for word in text_list if word not in stop_words]
    
    ## Stemming 
    if stemming ==True:
      st=nltk.stem.porter.PorterStemmer()
      text_list=[st.stem(word) for word in text_list]
      
    ## Lemmatisation 
    if lemantizing==True:
      le=nltk.stem.wordnet.WordNetLemmatizer()
      text_list=[le.lemmatize(word) for word in text_list]

    ## tokenize to string again
      text = " ".join(text_list)
      return text

  def clean_data(self,):
    stopwords_list=nltk.corpus.stopwords.words("english")
    self.classify_dataset["clean_history"]=self.classify_dataset["history_text"].apply(lambda x: self.preprocess(x,stop_words=stopwords_list,stemming=False,lemantizing=True))
    labelencoder = LabelEncoder()
    self.classify_dataset["creator_labels"] = labelencoder.fit_transform(self.classify_dataset["creator"])

  def train_test(self,):
    X_train,X_test,Y_train,Y_test=train_test_split(self.classify_dataset["clean_history"],self.classify_dataset["creator_labels"],test_size=0.3)
    return (X_train,X_test,Y_train,Y_test)

  def bag_of_words(self):
    print("BAG-OF-WORDS WITH SUPPORT VECTOR MACHINE")
    print("--------------------------------------------------------------------------------------------------------")
    self.clean_data()
    X_train,X_test,Y_train,Y_test=self.train_test()
    vectorizer=TfidfVectorizer(max_features=10000,ngram_range=(1,2),lowercase=False)
    classifier= SVC(kernel='linear',random_state=42)
    model=pipeline.Pipeline([("vectorizer",vectorizer),("classifier",classifier)])
    model.fit(X_train,Y_train)
    predicted=model.predict(X_test)
    train_accuracy = metrics.accuracy_score(Y_train, model.predict(X_train))
    test_accuracy = metrics.accuracy_score(Y_test, predicted)
    print("Train Accuracy:",  round(train_accuracy,2))
    print("Test Accuracy:",  round(test_accuracy,2))
    print("Detail:")
    print(metrics.classification_report(Y_test, predicted))
 
    print("--------------------------------------------------------------------------------------------------------")

  def Word2vec(self,):

    print("WORD 2 VEC WITH DEEP NEURAL NETWORK AND LSTM")
    print("--------------------------------------------------------------------------------------------------------")

    def xtrain_processing(X_train):
      doc=X_train
      doc_list=[]

      #creating unigrams
      for sentence in doc:
        text_list=sentence.split()
        list_grams = [" ".join(text_list[i:i+1]) 
                    for i in range(0, len(text_list), 1)]
        self.doc_list.append(list_grams)

      ## detect bigrams and trigrams
      self.bigrams_detector = gensim.models.phrases.Phrases(self.doc_list, 
                      delimiter=" ".encode(), min_count=5, threshold=10)
      self.bigrams_detector = gensim.models.phrases.Phraser(self.bigrams_detector)
      self.trigrams_detector = gensim.models.phrases.Phrases(self.bigrams_detector[self.doc_list], 
                  delimiter=" ".encode(), min_count=5, threshold=10)
      self.trigrams_detector = gensim.models.phrases.Phraser(self.trigrams_detector)

      #Tokenizing, finding indexes and convert back to sequence
      self.tokenizer=preprocessing.text.Tokenizer(lower=True, split= ' ',oov_token="Nan", filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
      self.tokenizer.fit_on_texts(self.doc_list)
      self.dictionary_vocab=self.tokenizer.word_index
      text2seq=self.tokenizer.texts_to_sequences(self.doc_list)

      #padding
      self.X_train_wv=preprocessing.sequence.pad_sequences(text2seq,maxlen=15,padding='post',truncating="post")
      

    
    def word2vec_model():
      nlp=gensim.models.word2vec.Word2Vec(self.doc_list,size=300,window=8,min_count=1,sg=1,iter=30)
      return nlp

    def xtest_processing(X_test):
    
      self.doc_test = X_test

      ## create list of n-grams'
      
      for string in self.doc_test:
          lst_words = string.split()
          lst_grams = [" ".join(lst_words[i:i+1]) for i in range(0, 
                      len(lst_words), 1)]
          self.doc_test_list.append(lst_grams)
          
      ## detect common bigrams and trigrams using the fitted detectors
      lst_corpus = list(self.bigrams_detector[self.doc_test_list])
      lst_corpus = list(self.trigrams_detector[lst_corpus])
      ## text to sequence with the fitted tokenizer
      lst_text2seq = self.tokenizer.texts_to_sequences(lst_corpus)

      ## padding sequence
      self.X_test_wv = preprocessing.sequence.pad_sequences(lst_text2seq, maxlen=15,
                  padding="post", truncating="post")

    def embed(nlp): 

      embeddings = np.zeros((len(self.dictionary_vocab)+1, 300))
      for word,idx in self.dictionary_vocab.items():
          ## update the row with vector
          try:
              embeddings[idx] =  nlp[word]
          ## if word not in model then skip and the row stays all 0s
          except:
              pass

      return embeddings

    def Neural_network(embeddings):
    
      model = keras.Sequential()
      model.add(layers.Embedding(input_dim=embeddings.shape[0],output_dim=embeddings.shape[1] , weights=[embeddings], input_length=15, trainable=False))
      # model.add(layers.Flatten())
      model.add(layers.Bidirectional(layers.LSTM(units=15, dropout=0.2)))
      model.add(layers.Dense(64, activation='relu'))
      model.add(layers.Dense(40, activation='softmax'))

      model.compile(loss='sparse_categorical_crossentropy',
                    optimizer='adam', metrics=['accuracy'])
      
      return model    
    
      
    def train_evaluate(model,Y_train,Y_test):

      dic_y_mapping = {n:label for n,label in 
              enumerate(np.unique(Y_train))}
      inverse_dic = {v:k for k,v in dic_y_mapping.items()}
      Y_train = np.array([inverse_dic[y] for y in Y_train])
      dnn_model=model.fit(x=self.X_train_wv, y=Y_train, batch_size=16, epochs=30, validation_split=0.1,callbacks=EarlyStopping(monitor='val_loss',verbose=1, patience=5))

      # model.evaluate()
      predicted_prob = model.predict(self.X_test_wv)
      predicted = [dic_y_mapping[np.argmax(pred)] for pred in 
                  predicted_prob]
      metrics.accuracy_score(Y_test,predicted)

      print("Detail:")
      print(metrics.classification_report(Y_test, predicted))
      
        
           
    self.clean_data()
    X_train,X_test,Y_train,Y_test=self.train_test()
    xtrain_processing(X_train)
    nlp=word2vec_model()
    xtest_processing(X_test)
    embeddings=embed(nlp)
    model=Neural_network(embeddings)
    train_evaluate(model,Y_train,Y_test)
    
    print("--------------------------------------------------------------------------------------------------------")

if __name__=="__main__":
  obj=superheros()
  obj.bag_of_words()
  obj.Word2vec()      


