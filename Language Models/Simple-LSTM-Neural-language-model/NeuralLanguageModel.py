#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install -U torchtext==0.6.0')


# In[1]:


import torch 
import spacy
import torch.nn as nn
import torch.optim as optim

from torchtext.data import Field,BucketIterator,BPTTIterator

from torch.utils.tensorboard import SummaryWriter


# In[ ]:





# In[2]:


import os
import re
import string
import numpy as np
from tqdm.notebook import tqdm
from collections import defaultdict


# In[3]:


from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
from sklearn.model_selection import train_test_split


# In[ ]:





# # save
# 
# torch.save(model.state_dict(), PATH)
# 
# # load
# model = TheModelClass(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))
# model.eval()

# In[4]:


def get_gpu_details():
    t = torch.cuda.get_device_properties(0).total_memory
    c = torch.cuda.memory_cached(0)
    a = torch.cuda.memory_allocated(0)
    r = torch.cuda.memory_reserved(0) 
    print(torch.cuda.get_device_name())
    print(f'Total GPU Memory {t} B , Cached GPU Memory {c} B, Allocated GPU Memory {a} B , Reserved {r} B')
    
    
    
if torch.cuda.is_available():
    device='cuda:0'
else:
    device='cpu'
print(f'Current Device: {device}')
if device=='cuda:0':
    torch.cuda.empty_cache()
    get_gpu_details()


# In[5]:


filename = 'brown.txt'

exclude = '``'
punctuations = string.punctuation + exclude

def open_and_process_lines(file):
    
    def check_conditions(line):
        if line.startswith('#'):
            return True
        
    def process(line):
        tokens = word_tokenize(line)
        table = str.maketrans('','',punctuations)
        tokens = [token.translate(table) for token in tokens]
        tokens = [word for word in tokens if word.isalnum()]
        return ' '.join(tokens)
    
    lines = open(file,'r',encoding='utf-8').readlines()
    lines = [process(line.lower()) for line in lines if not check_conditions(line)]
        
    return lines


lines = open_and_process_lines(filename)
lines = [line for line in lines if len(line)>0]


# In[ ]:





# In[6]:


train,val,_,_=train_test_split(lines,[1]*len(lines),test_size = 5000, random_state = 42)

train,test,_,_=train_test_split(train,[1]*len(train),test_size = 10000, random_state = 42)


# In[ ]:





# In[7]:


print((
len(train),
len(val),
len(test)))


# In[8]:


train_processed_file = 'train_corpus.txt'
test_processed_file = 'test_corpus.txt'
val_processed_file = 'val_corpus.txt'

train_file = open(train_processed_file,'w',encoding='utf-8')
for sentence in train:
    train_file.write(sentence + '\n')
train_file.close()

test_file = open(test_processed_file,'w',encoding='utf-8')
for sentence in test:
    test_file.write(sentence + '\n')
test_file.close()

val_file = open(val_processed_file,'w',encoding='utf-8')
for sentence in val:
    val_file.write(sentence + '\n')
val_file.close()


# In[9]:


from torchtext.datasets import LanguageModelingDataset


# In[10]:


spacy_english = spacy.load('en')

def english_tokenizer(sentence):
    return [token.text for token in spacy_english.tokenizer(sentence)]


# In[11]:


english = Field(
            sequential = True,
            tokenize = english_tokenizer,
            lower = True,
               )


# In[12]:


train_dataset = LanguageModelingDataset(train_processed_file,english)
val_dataset = LanguageModelingDataset(val_processed_file,english)


# In[13]:


english.build_vocab(train_dataset)


# In[14]:


len(english.vocab)


# In[37]:


torch.save(english,'field_object.pkl')


# In[39]:


english2 = Field(
            sequential = True,
            tokenize = english_tokenizer,
            lower = True,
               )

english2 = torch.load('field_object.pkl')
len(english2.vocab)


# In[15]:


def TrainIterator(batch_size,window_length):
    train_iterator = BPTTIterator(
        train_dataset,
        batch_size = batch_size,
        bptt_len = window_length,
        device = device,
        repeat = False,
        shuffle = True
    )
    print(len(train_iterator))
    return train_iterator
    
def TestIterator(testDataset,batch_size,window_length):
    test_iterator = BPTTIterator(
        testDataset,
        batch_size = batch_size,
        bptt_len = window_length,
        device = device,
        repeat = False,
        shuffle = False
    )
    #print(len(test_iterator))
    return test_iterator

    
def ValIterator(batch_size,window_length):
    val_iterator = BPTTIterator(
        val_dataset,
        batch_size = batch_size,
        bptt_len = window_length,
        device = device,
        repeat = False,
        shuffle = True
    )
    print(len(val_iterator))
    return val_iterator


# In[ ]:





# <img src='lstm.png'>

# In[16]:


class NeuralLM(nn.Module):
    def __init__(self,embedding_size,num_layers,hidden_size,size_of_vocab,dropout_rate):
        
        super(NeuralLM,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_size = embedding_size
        
        
        self.embedding = nn.Embedding(size_of_vocab,embedding_size)
        
        
        self.dropout_rate = dropout_rate
        if self.dropout_rate is not None:
            self.dropout = nn.Dropout(self.dropout_rate)
            
        
        self.fully_connected = nn.Linear(self.hidden_size,self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size,size_of_vocab)
                

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim = 2)
        
        if self.dropout_rate is not None:
            self.RNN = nn.LSTM(self.embedding_size,self.hidden_size,self.num_layers,dropout=self.dropout_rate)
        else:
            self.RNN = nn.LSTM(self.embedding_size,self.hidden_size,self.num_layers)
        
        
    def forward(self, inp,previous_state):
        embedding_op = self.embedding(inp)
        
        if self.dropout_rate is not None:
            embeddding_op = self.dropout(embedding_op)
            
        lstm_outputs,(h_n,c_n) = self.RNN(embedding_op,previous_state)
        
        fc_output = self.fully_connected(lstm_outputs)
        #fc_output = self.relu(fc_output)
        
        if self.dropout_rate is not None:
            fc_output = self.dropout(fc_output)
        
        
        predicted_word = self.output_layer(fc_output)
        #print(predicted_word.shape)
        #predicted_word = self.softmax(predicted_word)
        
        return predicted_word,(h_n,c_n)
        
    def init_state(self,batch_size,device):
        
        h_0 = torch.zeros(self.num_layers,batch_size,self.hidden_size).to(device)
        c_0 = torch.zeros(self.num_layers,batch_size,self.hidden_size).to(device)
        return (h_0,c_0)


# In[ ]:





# In[17]:


def get_accuracy(y_pred,y_true):
    return (y_pred == y_true).sum().item() / y_pred.shape[1]


# In[18]:


def train_per_epoch(model,batch_size,train_iterator,optimizer,criterion,clip):
    current_epoch_loss = 0
    curr_train_acc = 0
    model.train()
    
    for batch in tqdm(train_iterator):
        text = batch.text
        target = batch.target
        
        optimizer.zero_grad()
        
        h_i,c_i = model.init_state(batch_size,device)
        
        predictions,(h_i,c_i) = model(text,(h_i,c_i))
        _,train_preds = torch.max(predictions,dim = 2)

    
        predictions = predictions.permute(0,2,1)
        
        
        
        #print(predictions.shape)
        loss = criterion(predictions,target)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(),clip)
        
        
        optimizer.step()
        
        current_epoch_loss += loss.item()
        
        curr_train_acc += get_accuracy(train_preds,target)
        
    return current_epoch_loss / len(train_iterator),curr_train_acc/len(train_iterator)


# In[ ]:





# In[ ]:





# In[19]:


def trainNLM(model,batch_size,train_iterator,val_iterator,optimizer,scheduler,criterion,clip,num_epochs):
    val_loss_list,val_acc_list,train_loss_list,train_acc_list= [],[],[],[]
    
    for epoch in tqdm(range(num_epochs)):
        current_training_loss,train_acc = train_per_epoch(model,batch_size,train_iterator,optimizer,criterion,clip)
        
        
        
        
        curr_val_loss = 0
        curr_val_acc = 0
        
        model.eval()
        
        
        for val_batch in val_iterator:
            val_text = val_batch.text
            val_target = val_batch.target
            
            with torch.no_grad():
                
                h_i,c_i = model.init_state(batch_size,device)
       
                val_predictions,(h_i,c_i) = model(val_text,(h_i,c_i))

                val_loss = criterion(val_predictions.permute(0,2,1),val_target)
                curr_val_loss += val_loss.item()
                
                _,val_predictions = torch.max(val_predictions,dim = 2)

                curr_val_acc += get_accuracy(val_predictions,val_target)
            
        
        
        scheduler.step(val_loss)
        
        mean_val_loss = curr_val_loss/len(val_iterator)
        mean_val_acc = curr_val_acc/len(val_iterator)
        
        print(f'Epoch: {epoch}/{num_epochs}')
        print(f' train_loss: {current_training_loss} train_acc : {train_acc}')
        print(f'Validation : val_loss {mean_val_loss} val_acc : {mean_val_acc}') 
        
        val_loss_list.append(mean_val_loss)
        val_acc_list.append(mean_val_acc)
        train_loss_list.append(current_training_loss)
        train_acc_list.append(train_acc)
        
        torch.save(model.state_dict(),MODEL_CHECKPOINT)
        torch.save(optimizer.state_dict(),OPTIM_CHECKPOINT)
        
    return val_loss_list,val_acc_list,train_loss_list,train_acc_list


# In[ ]:





# In[20]:


embedding_size = 50
num_layers = 4
hidden_size = 256
vocab_size = len(english.vocab)
dropout = 0.5

batch_size = 128
window_len = 5

learning_rate = 0.01

MODEL_CHECKPOINT = f'NLM_model{window_len}.pth.pt'
OPTIM_CHECKPOINT = f'NLM_optim{window_len}.pth.pt'


neuralLM = NeuralLM(
    embedding_size,
    num_layers,
    hidden_size,
    vocab_size,
    dropout
).to(device)


# In[ ]:





# In[21]:


train_iterator = TrainIterator(batch_size,window_len)


val_iterator = ValIterator(batch_size,window_len)


# In[22]:


learning_rate = 0.01

optimizer = optim.AdamW(
    neuralLM.parameters(),
    lr = learning_rate,
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2,
                                                 threshold=1e-5, threshold_mode='rel',
                                                 cooldown=0, min_lr=0, eps=1e-08, verbose=False)

criterion = nn.CrossEntropyLoss()


# In[23]:


get_gpu_details()


# In[ ]:


# for batch in val_iterator:
#     print(batch.text)
#     print(batch.target)
#     break


# In[24]:


val_loss,val_acc,train_loss,train_acc = trainNLM(neuralLM,batch_size,train_iterator,val_iterator,optimizer,scheduler,criterion,1,50)


# In[59]:


# train_model = NeuralLM(
#     embedding_size,
#     num_layers,
#     hidden_size,
#     vocab_size,
#     dropout
# ).to(device)

# train_model.load_state_dict(torch.load(MODEL_CHECKPOINT))


# In[ ]:





# In[25]:


import matplotlib.pyplot as plt

# summarize history for accuracy
plt.plot(train_acc)
plt.plot(val_acc)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(train_loss)
plt.plot(val_loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# # Testing

# In[ ]:





# In[26]:


test_model = NeuralLM(
    embedding_size,
    num_layers,
    hidden_size,
    vocab_size,
    dropout
).to(device)

test_model.load_state_dict(torch.load(MODEL_CHECKPOINT))


# test_dataset = LanguageModelingDataset(test_processed_file,english)


# In[27]:


def calculate_perplexity_exp(P):
        return (P * np.log(P))/np.log(2)


# In[28]:


def calculate_sentence_perplexity(probabilities):
    perplexity_exp = 0
    joint_prob = 1
    
    #ngram_probab = np.sum(probabilities)
    
    for probability in probabilities:
        joint_prob *= probability
        perplexity_exp += calculate_perplexity_exp(joint_prob)
    
    overall_perplexity = 2**(-perplexity_exp)
    
    return overall_perplexity,joint_prob


# In[ ]:





# In[32]:


def evaluateNLM(model,device,test_sentences,english_field,test_batch_size,window_len):
    
    def write_temp_file(sentence,filename,):
        temp_file = open(filename,'w',encoding='utf-8')
        temp_file.write(sentence)
        temp_file.close()

    def get_test_accuracy(y_pred,y_true):
        return (y_pred == y_true).sum().item() / y_pred.shape[1]
    
    
    temp_test_file = 'temp_test_file.txt'
        
    model.eval()
    
    tot_test_loss = 0 

    loss = []
    
    all_perp = []
    curr_test_acc  = 0
    all_acc = []
    
    for sentence in tqdm(test_sentences):
        write_temp_file(sentence,temp_test_file)
        test_dataset = LanguageModelingDataset(temp_test_file,english_field)
        
        test_iterator = TestIterator(test_dataset,test_batch_size,window_len)
        
        curr_acc = 0
        perp = []
        
        for test_batch in test_iterator:
            test_text = test_batch.text
            test_target = test_batch.target
            
            with torch.no_grad():
                
                h_i,c_i = model.init_state(test_batch_size,device)
                
                test_predictions,(h_i,c_i) = model(test_text,(h_i,c_i))

                #print(test_predictions)
                test_loss = criterion(test_predictions.permute(0,2,1),test_target)
                
                loss.append(test_loss.item())
                
                #perp.append(math.exp(test_loss.item()))
                
                probabilities = model.softmax(test_predictions)
                
                real_prob, pred_indices = torch.max(probabilities,dim=2)
                
                real_prob = real_prob.detach().cpu().numpy()
                
                curr_test_acc += get_test_accuracy(pred_indices,test_target)
                
                ngram_perplexity,ngram_probability = calculate_sentence_perplexity(real_prob)
                perp.append(ngram_perplexity)
                
        
        try:
            perp_mean = np.mean(perp)
            all_perp.append(perp_mean)
            all_acc.append(curr_acc / len(test_iterator))
        except:
            pass

        

    return loss,all_perp,all_acc
    


# In[ ]:





# In[30]:


import math


# In[47]:


test_batch_size = 1
test_window_len = 5

loss,perp,acc = evaluateNLM(test_model,device,test,english,test_batch_size,test_window_len)


# In[34]:


math.exp(np.mean(loss))


# In[35]:


perp2 = [p for p in perp if p > 0]
np.mean(perp2)


# In[50]:


math.exp(np.mean(loss))


# In[48]:


perp2 = [p for p in perp if p > 0]


# In[49]:


np.mean(perp2)


# In[46]:


train_perp_file = open('train_perp.txt','w',encoding='utf-8')

for i in tqdm(range(len(perp2))):
  sentence = train[i].rstrip('\n')
  to_write = f'{sentence}\t{perp2[i]}\n'
  train_perp_file.write(to_write)

train_perp_file.write(f'{np.mean(perp2)}\n')
train_perp_file.close()


# In[52]:


test_perp_file = open('test_perp.txt','w',encoding='utf-8')

for i in tqdm(range(len(perp2))):
  sentence = test[i].rstrip('\n')
  to_write = f'{sentence}\t{perp2[i]}\n'
  test_perp_file.write(to_write)

test_perp_file.write(f'{np.mean(perp2)}\n')
test_perp_file.close()


# In[51]:


len(perp2)

