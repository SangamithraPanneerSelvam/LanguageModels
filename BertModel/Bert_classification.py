import transformers
# from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer,AdamW, get_linear_schedule_with_warmup
import torch

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.model_selection  import train_test_split
from torch import nn, optim
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
from pylab import rcParams
from matplotlib import rc
import spacy
import nltk
import re
# nltk.download('stopwords')
from sklearn.preprocessing import LabelEncoder


# nltk.download('wordnet')
# nltk.download('omw-1.4')

class DistillBERTClass(torch.nn.Module):

   def __init__(self,n_classes,PRE_TRAINED_MODEL_NAME):
     super(DistillBERTClass,self).__init__()
     self.bert=transformers.DistilBertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
     self.drop=nn.Dropout(p=0.2)
     self.drop = torch.nn.Dropout(0.3)
     self.out = torch.nn.Linear(self.bert.config.hidden_size,n_classes)
    
   def forward(self,input_ids, attention_mask):
        distilbert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        output_1 = self.drop(pooled_output)
        output = self.out(output_1)
        return output


class Encode(Dataset):

  def __init__(self,sentences, targets,tokenizer,max_len):
    self.sentences=sentences
    self.targets=targets
    self.tokenizer=tokenizer
    self.max_len=max_len

  def __len__(self):
    return len(self.sentences)

  def __getitem__(self,item):
    Hero_story=self.sentences[item]
    creator_name=self.targets[item]
    encoding=self.tokenizer.encode_plus(Hero_story,max_length=self.max_len,
                                   pad_to_max_length=True,
                                   add_special_tokens=True,
                                   return_token_type_ids=False,
                                   return_tensors='pt',
                                   return_attention_mask=True,
                                   truncation=True)
    
    return{
        'Hero_History': Hero_story,
        'input_ids':encoding['input_ids'].flatten(),
        'attention_mask':encoding['attention_mask'].flatten(),
        'targets':torch.tensor(creator_name,dtype=torch.long)

    }
    



class superheros():

  def __init__(self):
    self.RANDOM_SEED=42
    self.class_names=42 #since there are 41 creators
    self.BATCH_SIZE=16
    self.MAX_LEN=384
    self.EPOCHS=5
    self.LEARNING_RATE=2e-5
    self.PRE_TRAINED_MODEL_NAME='distilbert-base-uncased'
    self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.load_data()

  def load_data(self):
    data=pd.read_csv("superheroes_nlp_dataset.csv",encoding="utf-8")
    self.dataset=data[['history_text','creator']]

  def load_model(self):
    
    bert_model=DistilBertForSequenceClassification.from_pretrained(self.PRE_TRAINED_MODEL_NAME)
    model=DistillBERTClass(self.class_names,self.PRE_TRAINED_MODEL_NAME)
    self.tokenizer=DistilBertTokenizer.from_pretrained(self.PRE_TRAINED_MODEL_NAME)
    
    return model
  
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

  def clean_data(self):
    stopwords_list=nltk.corpus.stopwords.words("english")
    self.dataset["clean_history"]=self.dataset["history_text"].apply(lambda x: self.preprocess(x,stop_words=stopwords_list,stemming=False,lemantizing=True))
    labelencoder = LabelEncoder()
    self.dataset["creator_labels"] = labelencoder.fit_transform(self.dataset["creator"])

  def split_data(self):

    self.clean_data()
    # X_train,X_test,Y_train,Y_test=train_test_split(self.dataset["clean_history"],self.dataset["creator_labels"],test_size=0.2,random_state=self.RANDOM_SEED)
    df_train,df_test=train_test_split(self.dataset,test_size=0.2,random_state=self.RANDOM_SEED)
    return df_train,df_test

  def data_loader (self,df):

    loader=Encode(sentences=df.clean_history.to_numpy(),targets=df.creator_labels.to_numpy(),tokenizer=self.tokenizer,max_len=self.MAX_LEN)
    
    return DataLoader(loader,num_workers=2,batch_size=self.BATCH_SIZE)

  def encode_dataset(self,df_train,df_test): 
    
    train_loader=self.data_loader(df_train)
    # test_loader=self.data_loader(df_test)

    return train_loader

  def hyperparameters(self,model,train_loader):

    optimizer=AdamW(model.parameters(),lr=self.LEARNING_RATE,correct_bias=False)
    total_steps=len(train_loader) *self.EPOCHS
    scheduler=get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0,num_training_steps=total_steps)
    loss_fn=nn.CrossEntropyLoss().to(self.device)
    return optimizer,scheduler,loss_fn

  def train_epoch(self,model,
        train_loader,    
        loss_fn, 
        optimizer, 
        scheduler, 
        n_total):    
    
    model = model.train()

    losses = []
    correct_predictions = 0
    
    for d in train_loader:
      input_ids = d["input_ids"].to(self.device)
      attention_mask = d["attention_mask"].to(self.device)
      targets = d["targets"].to(self.device)

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )

      _, preds = torch.max(outputs, dim=1)
      loss = loss_fn(outputs, targets)

      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())

      loss.backward()
      nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      optimizer.step()
      scheduler.step()
      optimizer.zero_grad()

    return correct_predictions.double()/n_total, np.mean(losses)

  def epochs(self):
 
    model=self.load_model()
    df_train,df_test=self.split_data()  
    train_loader=self.encode_dataset(df_train,df_test)
    model = model.to(self.device)
    optimizer,scheduler,loss_fn = self.hyperparameters(model,train_loader)

    for epoch in range(self.EPOCHS):

      print(f"Epoch {epoch +1}/{self.EPOCHS}")
      print('-'*10)

      train_acc, train_loss = self.train_epoch(
        model,
        train_loader,    
        loss_fn, 
        optimizer, 
        scheduler, 
        len(df_train)
      )

      print(f"Trainloss {train_loss} accuracy {train_acc}")


  def predict(self,model,new_text):


    encoding=self.tokenizer.encode_plus(new_text,max_length=384,
                                    pad_to_max_length=True,
                                    add_special_tokens=True,
                                    return_token_type_ids=False,
                                    return_tensors='pt',
                                    return_attention_mask=True,
                                    truncation=True)
      
      
      
    input_ids=encoding['input_ids'].to(self.device)
    attention_mask=encoding['attention_mask'].to(self.device)
    
    with torch.no_grad():
        # text = torch.tensor(encoding)
        outputs=model(input_ids=input_ids, attention_mask=attention_mask)
        _,preds=torch.max(outputs,dim=1)
        print(preds)

  
if __name__=="__main__":
  obj=superheros()
  obj.epochs()
          



    
  
    

