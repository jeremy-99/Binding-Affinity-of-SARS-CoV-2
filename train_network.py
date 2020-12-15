#!/usr/bin/env python
# coding: utf-8

# In[2]:


from google.colab import drive
drive.mount('/content/drive')


# In[3]:


import os
os.chdir('/content/drive/My Drive/Peptide MHC Project')
import sys
sys.path.extend(".")
from architecture import *
from peptide_embedding import *  
from MHC_sequence_embedding import *

import torch
import torch.nn as nn
import tensorflow as tf
from tensorflow import summary
from tensorboard.backend.event_processing.event_file_inspector import get_inspection_units, print_dict, get_dict_to_print

import datetime
import time
import pytz
import pandas as pd
import numpy as np
import psutil
import shutil
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

print(torch.__version__)
torch.cuda.get_device_name(0)
process = psutil.Process(os.getpid())


# In[4]:


# Full Model w/MHC embedding

# Training dataset
link1 = 'https://raw.githubusercontent.com/cmb-chula/MHCSeqNet/master/cleaned_MHC_all_classes.csv'
x = pd.read_csv(link1)

# Corresponding amino acid sequences
link2 = 'https://raw.githubusercontent.com/cmb-chula/MHCSeqNet/master/PretrainedModels/sequence_model/AlleleInformation.txt'
allele_seq = urllib.request.urlopen(link2)
MHC_sequence_df = MHC_seq_df(allele_seq)

# Remove training examples with unknown MHC sequences
alleles = x['allele'] 
good_idx = alleles.isin(MHC_sequence_df['MHC_allele'])
binding_affinity = x['binding_quality'][good_idx]

# Convert to binary np array
y = np.array(binding_affinity_class(binding_affinity.values.tolist()))
print(y.shape)


# In[5]:


# Gets distribution of peptide lengths
peptides = x['peptide'].to_list()
lens = [len(i) for i in peptides]
unique_lens = []
for length in lens:
  if length not in unique_lens:
    unique_lens.append(length)

counts = {}
fracs = {}
for length in sorted(unique_lens):
  counts[length] = lens.count(length)
  fracs[length] = round(lens.count(length) / len(peptides), 3)
print(counts)
print(fracs)


# In[ ]:


# # Alternatively: can load 140_179 and 50_84 files and concatenate
# # Then delete partial tensors & free memory with torch.cuda.empty_cache()

# print(process.memory_info().rss / 1E9)
# allele_embedding_tensor = torch.load('embeddings/full_alleles_20k.pt')#.cuda()
# print(allele_embedding_tensor.shape)

# # Loads peptide tensor
# print(process.memory_info().rss / 1E9)
# peptide_embedding_tensor = torch.load('embeddings/peptide_tensor_20k.pt')#.cuda()
# print(peptide_embedding_tensor.shape)
# print(process.memory_info().rss / 1E9)


# In[ ]:


# # Create train, val, and test sets
# n = allele_embedding_tensor.shape[0]
# train_len = round(0.8*n)
# val_len = round(0.1*n)
# test_len = n - train_len - val_len

# # Can't shuffle without running out of memory
# train_allele = allele_embedding_tensor[:train_len,:,:]
# train_peptide = peptide_embedding_tensor[:train_len,:,:]
# # train_y = y[:train_len]
# train_y = torch.tensor(y[:train_len]).float()

# val_allele = allele_embedding_tensor[train_len:train_len+val_len,:,:].cuda()
# val_peptide = peptide_embedding_tensor[train_len:train_len+val_len,:,:].cuda()
# # val_y = y[train_len:train_len+val_len]
# val_y = torch.tensor(y[train_len:train_len+val_len]).float().cuda()

# test_allele = allele_embedding_tensor[-test_len:,:,:]#.cuda()
# test_peptide = peptide_embedding_tensor[-test_len:,:,:]#.cuda()
# # Account for potentially longer y
# # test_y = y[-test_len:n+1]  
# test_y = torch.tensor(y[-test_len:n+1]).float()#.cuda()
# print(process.memory_info().rss / 1E9)


# In[6]:


# Set hyperparameters 
n_epochs = 50
batch_size = 32

embedding_dim = 100
hidden_dim = 100  # 128

# bidirectional (*2) & 1st and final states concatenated (*2)
just_last_hidden = False
gru_output_dim = hidden_dim*2 if just_last_hidden else hidden_dim*2*2


# In[7]:


# Define Model
# Need to restart runtine every time architecture.py changes
config = {'embedding_dim':embedding_dim,
          'hidden_dim':hidden_dim,
          'peptide_output_dim':gru_output_dim, 
          'allele_output_dim':gru_output_dim,
          'just_last_hidden':just_last_hidden,
          'hidden_1_dim':200,
          'hidden_2_dim':200,  # None if only two layers
          'dropout_fc':0.4,
          'dropout_input':0.4,
          'dropout_gru':0.3}  # Pytorch doesn't have recurrent dropout

peptide_model = peptide_BiGRU(config).cuda()
allele_model = allele_BiGRU(config).cuda()
output_model = Output_Layer(peptide_model, allele_model, config).cuda()

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(output_model.parameters())
print(process.memory_info().rss / 1E9)


# In[8]:


# Time function
def convert(seconds): 
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return hours, minutes, seconds

# Set up tensorboard
get_ipython().run_line_magic('load_ext', 'tensorboard')

utcnow = datetime.datetime.utcnow()
mytimezone = pytz.timezone('America/Los_Angeles')
local_dt = utcnow.replace(tzinfo=pytz.utc).astimezone(mytimezone)
now = mytimezone.normalize(local_dt)
date_time = now.strftime("%m_%d_%Y_%H:%M:%S")
print(date_time)

train_log_dir = 'logs/tensorboard/train/' + date_time
train_summary_writer = summary.create_file_writer(train_log_dir)
val_log_dir = 'logs/tensorboard/val/' + date_time
val_summary_writer = summary.create_file_writer(val_log_dir)

get_ipython().run_line_magic('tensorboard', '--logdir logs/tensorboard')


# In[12]:


n_files = len(os.listdir('embeddings/peptides/'))
n_train_files = round(0.8*n_files)
n_val_files = round(0.1*n_files)
n_test_files = n_files - n_train_files - n_val_files
SIZE = 4000

def join_tensors(start, stop, labels, size):
  # Creates validation and test set data
  peptides = []
  alleles_50_84 = []
  alleles_140_179 = []
  for i in range(start, stop):
    name = 'tensor_'+str(i)
    peptide_tensor = torch.load('embeddings/peptides/'+name)
    peptides.append(peptide_tensor)
    allele_50_84_tensor = torch.load('embeddings/MHC_50_84/'+name)
    alleles_50_84.append(allele_50_84_tensor)
    allele_140_179_tensor = torch.load('embeddings/MHC_140_179/'+name)
    alleles_140_179.append(allele_140_179_tensor)

  peptide_embedding_tensor = torch.cat(peptides, dim=0).cuda()
  allele_50_84_tensor = torch.cat(alleles_50_84, dim=0)
  allele_140_179_tensor = torch.cat(alleles_140_179, dim=0)
  allele_embedding_tensor = torch.cat((allele_140_179_tensor, allele_50_84_tensor), dim=1)

  begin = start*size
  end = stop*size
  if end < y.shape[0]:
    y_tensor = torch.tensor(labels[begin:end]).float()
  else:
    y_tensor = torch.tensor(labels[begin:]).float()
  return peptide_embedding_tensor.cuda(), allele_embedding_tensor.cuda(), y_tensor.cuda()

val_peptide, val_allele, val_y = join_tensors(n_train_files, n_train_files+n_val_files, y, SIZE)


# In[ ]:





# In[ ]:


# Train model
losses = []
best_epoch = -1
best_val_loss = sys.float_info.max
batch_size = 16
n_batches = train_len // batch_size
start_memory = process.memory_info().rss
kpis = {'Train Loss':[], 'Val Loss':[], 'Train Accuracy':[], 'Val Accuracy':[]}
for epoch in range(n_epochs):
    print('Epoch', epoch+1)
    epoch_start = time.time()
    train_loss = 0.0
    tb_loss = 0.0
    batch_count = 0
    avg_train_acc = 0

    for num in range(n_train_files):
      # Loads tensors for SIZE examples
      name = 'tensor_' + str(num)
      peptide_embedding_tensor = torch.load('embeddings/peptides/'+name)
      MHC_tensor_50_84 = torch.load('embeddings/MHC_50_84/'+name)
      MHC_tensor_140_179 = torch.load('embeddings/MHC_140_179/'+name)
      peptide_embedding_tensor = torch.load(pep_name)
      allele_embedding_tensor = torch.cat(((MHC_tensor_140_179, MHC_tensor_50_84), dim=1))
      y_tensor = torch.tensor(y[num*SIZE:(num+1)*SIZE]).float()

      rand_idx = torch.randperm(train_len)
      for i in range(0, SIZE, batch_size):
        batch_count += 1
        # NEED TO ACCOUNT FOR BATCHES THAT ARE TOO SMALL
        indices = rand_idx[i:i+batch_size]  # Picks random indices without reptition
        optimizer.zero_grad()

        # Predict output and compute loss
        y_hat = output_model(peptide_embedding_tensor[indices].cuda(), allele_embedding_tensor[indices].cuda())
        y_batch = y_tensor[indices].cuda()
        batch_loss = criterion(y_hat.squeeze(), y_batch)
        train_loss += float(batch_loss)
        tb_loss += float(batch_loss)
        batch_acc = float((abs(y_batch - y_hat.squeeze()) < .5).sum() / batch_size)
        avg_train_acc += batch_acc

        # Removing y_hat and y_batch doesn't do anything
        batch_loss.backward()  # This is where the memory goes up
        if batch_count < 5:
          print("made it")
        optimizer.step()
          
        if batch_count == 20:
          seconds = time.time() - epoch_start
          exp_epoch_time = seconds * n_batches / batch_count
          print("Expected Epoch Duration: %d hours %02d minutes %02d seconds" % convert(exp_epoch_time))
          
        if batch_count % 100 == 0:
          print('Epoch {} Batch {}\tLoss: {:.4f}, Accuracy: {:.4f}'.format(epoch+1, batch_count, batch_loss, batch_acc))
        
        if batch_count % 5 == 0: 
          print(batch_count)
          with train_summary_writer.as_default():
            tf.summary.scalar('Train loss', tb_loss, step=epoch*n_batches+batch_count)
          tb_loss = 0.0
          
          with torch.no_grad():
            val_y_hat = output_model(val_peptide, val_allele)
            val_loss = float(criterion(val_y_hat.squeeze(), val_y))
            with val_summary_writer.as_default():
              tf.summary.scalar('Validation loss', val_loss, step=epoch*n_batches+batch_count)
      
      # Don't think this does anything
      del batch_loss
      torch.cuda.empty_cache()

    with torch.no_grad():
      val_y_hat = output_model(val_peptide, val_allele)
      val_loss = float(criterion(val_y_hat.squeeze(), val_y))
      val_acc = float((abs(val_y - val_y_hat.squeeze()) < .5).sum()) / val_len
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      best_model_state_dict = output_model.state_dict()
      best_epoch = epoch
    
    avg_train_acc = avg_train_acc/batch_count
    print('Epoch {} Results'.format(epoch+1))
    print('Train Loss: {:.4f}\tAvg Accuracy: {:.4f}'.format(train_loss, avg_train_acc))
    print('Validation Loss: {:.4f}\tAccuracy: {:.4f}\n'.format(val_loss, val_acc))
    kpis['Val Loss'].append(val_loss)
    kpis['Val Accuracy'].append(val_acc)
    kpis['Train Loss'].append(train_loss)
    kpis['Train Accuracy'].append(avg_train_acc)

    # Time elapsed
    elapsed = time.time() - epoch_start
    print("Epoch Duration: %d hours %02d minutes %02d seconds" % convert(elapsed))

    # Automatically stop training if hasn't improved in 5 epochs
    if epoch - best_epoch == 5 and best_epoch >= 2:
      break
    del val_y_hat
    torch.cuda.empty_cache()


# In[ ]:


# # Train model
# losses = []
# best_epoch = -1
# best_val_loss = sys.float_info.max
# batch_size = 16
# n_batches = train_len // batch_size
# start_memory = process.memory_info().rss
# kpis = {'Train Loss':[], 'Val Loss':[], 'Train Accuracy':[], 'Val Accuracy':[]}
# for epoch in range(n_epochs):
#     print('Epoch', epoch+1)
#     epoch_start = time.time()
#     train_loss = 0.0
#     tb_loss = 0.0
#     batch_count = 0
#     avg_train_acc = 0

#     rand_idx = torch.randperm(train_len)

#     for i in range(0, train_len, batch_size):
#       batch_count += 1
#       indices = rand_idx[i:i+batch_size]
#       optimizer.zero_grad()

#       # Predict output and compute loss
#       y_hat = output_model(train_peptide[indices].cuda(), train_allele[indices].cuda())
#       y_batch = train_y[indices].cuda()
#       batch_loss = criterion(y_hat.squeeze(), y_batch)
#       train_loss += float(batch_loss)
#       tb_loss += float(batch_loss)
#       batch_acc = float((abs(y_batch - y_hat.squeeze()) < .5).sum() / batch_size)
#       avg_train_acc += batch_acc

#       # Removing y_hat and y_batch doesn't do anything
#       batch_loss.backward()  # This is where the memory goes up
#       # print("made it")
#       optimizer.step()
        
#       if batch_count == 20:
#         seconds = time.time() - epoch_start
#         exp_epoch_time = seconds * n_batches / batch_count
#         print("Expected Epoch Duration: %d hours %02d minutes %02d seconds" % convert(exp_epoch_time))
        
#       if batch_count % 100 == 0:
#         print('Epoch {} Batch {}\tLoss: {:.4f}, Accuracy: {:.4f}'.format(epoch+1, batch_count, batch_loss, batch_acc))
      
#       if batch_count % 5 == 0: 
#         print(batch_count)
#         with train_summary_writer.as_default():
#           tf.summary.scalar('Train loss', tb_loss, step=epoch*n_batches+batch_count)
#         tb_loss = 0.0
        
#         with torch.no_grad():
#           val_y_hat = output_model(val_peptide, val_allele)
#           val_loss = float(criterion(val_y_hat.squeeze(), val_y))
#           with val_summary_writer.as_default():
#             tf.summary.scalar('Validation loss', val_loss, step=epoch*n_batches+batch_count)


#       # if batch_count % 50 == 0: 
#       #   break
      
#       # Don't think this does anything
#       del batch_loss
#       torch.cuda.empty_cache()

#     with torch.no_grad():
#       val_y_hat = output_model(val_peptide, val_allele)
#       val_loss = float(criterion(val_y_hat.squeeze(), val_y))
#       val_acc = float((abs(val_y - val_y_hat.squeeze()) < .5).sum()) / val_len
#     if val_loss < best_val_loss:
#       best_val_loss = val_loss
#       best_model_state_dict = output_model.state_dict()
#       best_epoch = epoch
    
#     avg_train_acc = avg_train_acc/batch_count
#     print('Epoch {} Results'.format(epoch+1))
#     print('Train Loss: {:.4f}\tAvg Accuracy: {:.4f}'.format(train_loss, avg_train_acc))
#     print('Validation Loss: {:.4f}\tAccuracy: {:.4f}\n'.format(val_loss, val_acc))
#     kpis['Val Loss'].append(val_loss)
#     kpis['Val Accuracy'].append(val_acc)
#     kpis['Train Loss'].append(train_loss)
#     kpis['Train Accuracy'].append(avg_train_acc)

#     # Time elapsed
#     elapsed = time.time() - epoch_start
#     print("Epoch Duration: %d hours %02d minutes %02d seconds" % convert(elapsed))

#     # Automatically stop training if hasn't improved in 5 epochs
#     if epoch - best_epoch == 5 and best_epoch >= 2:
#       break
#     del val_y_hat
#     torch.cuda.empty_cache()


# In[ ]:


path = '/content/drive/My Drive/Peptide MHC Project/Trained Models/trained_' + str(epoch+1) + '_epochs.pt'
if os.path.exists(path):
  os.remove(path)
print("Best epoch:", best_epoch)
kpis['best_epoch'] = best_epoch
kpis['best_model_state_dict'] = best_model_state_dict
torch.save(kpis, path)


# In[ ]:


test_y_hat = output_model(test_peptide.cuda(), test_allele.cuda())
test_loss = float(criterion(test_y_hat.squeeze(), test_y.cuda()))
test_acc = float((abs(test_y.cuda() - test_y_hat.squeeze()) < .5).sum()) / test_len
print(test_loss)
print(test_acc)


# In[ ]:


# Kills all runs
inspect_units = get_inspection_units(logdir='logs/tensorboard')
run_len = {}

for run in inspect_units:
    path = run[0]
    max_length = 0
    for key, value in get_dict_to_print(run.field_to_obs).items():
        if value is not None:
            length = value['max_step']
            if max_length < length:
                max_length = length
    run_len[path] = max_length

for run, length in run_len.items():
    try:
        print(f'{run} is {length} and was deleted')
        shutil.rmtree(run)
    except:
        print(f"OS didn't let us delete {run}")


# In[ ]:





# In[ ]:




