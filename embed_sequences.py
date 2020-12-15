#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[2]:


get_ipython().system('pip3 install torch==1.2.0+cu92 torchvision==0.4.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html')


# In[3]:


import os
os.chdir('/content/drive/My Drive/Peptide MHC Project')
import torch
import torch.nn as nn 
print(torch.__version__)

import pandas as pd
import numpy as np
from peptide_embedding import *  
from MHC_sequence_embedding import *
import urllib
import shutil


# In[4]:


# Full Model w/ MHC embedding
pretrained_model = torch.load('ssa_L1_100d_lstm3x512_lm_i512_mb64_tau0.5_lambda0.1_p0.05_epoch100.sav')

# Training dataset of peptide sequence, MHC allele name, binary binding affinity (positive, negative)
link1 = 'https://raw.githubusercontent.com/cmb-chula/MHCSeqNet/master/cleaned_MHC_all_classes.csv'
x = pd.read_csv(link1)

# Dataset of corresponding amino acid sequence for MHC alleles (Beta sheet, alpha helix res 140-179, alpha helix res 50-84)
link2 = 'https://raw.githubusercontent.com/cmb-chula/MHCSeqNet/master/PretrainedModels/sequence_model/AlleleInformation.txt'
allele_seq = urllib.request.urlopen(link2)
MHC_sequence_df = MHC_seq_df(allele_seq)

# Only consider training examples whose sequences are known
alleles = x['allele']
good_idx = alleles.isin(MHC_sequence_df['MHC_allele'])

# Saves training MHC sequences as csv
if not os.path.exists("/content/drive/My Drive/Peptide MHC Project/allele_sequences/140_179.csv"):
  classI_alleles = alleles[good_idx]
  # Gets known allele sequences from allele names in training data
  alpha_res_140_179, alpha_res_50_84 = allele_sequence(classI_alleles, MHC_sequence_df)
  alpha_res_140_179.to_csv("/content/drive/My Drive/Peptide MHC Project/allele_sequences/140_179.csv",index = False)
  if not os.path.exists("/content/drive/My Drive/Peptide MHC Project/allele_sequences/50_84.csv"):
    alpha_res_50_84.to_csv("/content/drive/My Drive/Peptide MHC Project/allele_sequences/50_84.csv",index = False)


# In[5]:


# Loads amino acid sequences of MHC alleles 
alpha_res_140_179_read = pd.read_csv("allele_sequences/140_179.csv")
alpha_res_50_84_read = pd.read_csv("allele_sequences/50_84.csv")

# Strip \n
alpha_res_50_84_read.iloc[:,0] = alpha_res_50_84_read.iloc[:,0].map(lambda x: x.rstrip('\n'))
alpha_res_140_179_read.iloc[:,0] = alpha_res_140_179_read.iloc[:,0].map(lambda x: x.rstrip('\n'))

# Converts df of AA sequences to list of numpy arrays with Uniprot-encodings
alpha_res_140_179_encoding = peptide_encoding(alpha_res_140_179_read.squeeze())
alpha_res_50_84_encoding = peptide_encoding(alpha_res_50_84_read.squeeze())
n = len(alpha_res_140_179_encoding)

# Repeats with good peptides in training data
peptides = x['peptide'][good_idx]
peptide_encode = peptide_encoding(peptides)

# Sets up embeddings paths
embeddings_dir = '/content/drive/My Drive/Peptide MHC Project/embeddings'
if not os.path.exists(embeddings_dir):
  os.mkdir(embeddings_dir)


# In[6]:


SIZE = 2000
pep_dir = os.path.join(embeddings_dir, 'peptides')
if os.path.exists(pep_dir):
  shutil.rmtree(pep_dir)
os.mkdir(pep_dir)

# Embeds peptides
peptide_embedding_list = []
tensor_idx = 0
for i in range(0, n, 1000):
  count = 1000 if i + 1000 < n else n - i
  peptide_embeddings = peptide_embedding(peptide_encode[i:i+count], 15, pretrained_model)
  peptide_embeddings = torch.stack(peptide_embeddings)
  peptide_embedding_list.append(peptide_embeddings)
  # Embeds 1000 peptides or the remainder 
  if (i + 1000) % SIZE == 0 or count < 1000:
    print(i)
    peptide_tensor = torch.cat(peptide_embedding_list, dim=0)
    path = os.path.join(pep_dir, 'tensor_' + str(tensor_idx))
    torch.save(peptide_tensor, path)
    tensor_idx += 1
    peptide_embedding_list = []
    del peptide_tensor


# In[7]:


MHC_dir_140_179 = os.path.join(embeddings_dir, 'MHC_140_179')
if os.path.exists(MHC_dir_140_179):
  shutil.rmtree(MHC_dir_140_179)
os.mkdir(MHC_dir_140_179)

# Embeds MHC sequences 140-179, 10000 at a time
alpha_140_179_list = []
tensor_idx = 0
for i in range(0,n,1000):
  count = 1000 if i + 1000 < n else n - i
  allele_140_179_embeddings = peptide_embedding(alpha_res_140_179_encoding[i:i+count], 53, pretrained_model)
  allele_140_179_embeddings = torch.stack(allele_140_179_embeddings)
  alpha_140_179_list.append(allele_140_179_embeddings)
  if (i + 1000) % SIZE == 0 or count < 1000:
    print(i)
    alpha_140_179_tensor = torch.cat(alpha_140_179_list, dim=0)
    path = os.path.join(MHC_dir_140_179, 'tensor_' + str(tensor_idx))
    torch.save(alpha_140_179_tensor, path)
    tensor_idx += 1
    alpha_140_179_list = []
    del alpha_140_179_tensor


# In[ ]:


MHC_dir_50_84 = os.path.join(embeddings_dir, 'MHC_50_84')
if os.path.exists(MHC_dir_50_84):
  shutil.rmtree(MHC_dir_50_84)
os.mkdir(MHC_dir_50_84)

# Embeds MHC sequences 50-84, 10000 at a time
embedding_50_84_list = []
tensor_idx = 0
for i in range(0, n, 1000):
  count = 1000 if i + 1000 < n else n - i
  embedding_50_84 = peptide_embedding(alpha_res_50_84_encoding[i:i+count], 53, pretrained_model)
  embedding_50_84 = torch.stack(embedding_50_84)
  embedding_50_84_list.append(embedding_50_84)
  if (i + 1000) % SIZE == 0 or count < 1000:
    print(i)
    embedding_50_84_tensor = torch.cat(embedding_50_84_list, dim=0)
    path = os.path.join(MHC_dir_50_84, 'tensor_' + str(tensor_idx))
    torch.save(embedding_50_84_tensor, path)
    tensor_idx += 1
    embedding_50_84_list = []


# In[ ]:


# Joins peptide tensors
peptide_tensors = []
pep_dir = '/content/drive/My Drive/Peptide MHC Project/embeddings/peptides'
for f in os.listdir(pep_dir):
  tensor = torch.load(os.path.join(pep_dir, f))
  peptide_tensors.append(tensor)

peptide_tensor = torch.cat(peptide_tensors, dim=0)
print(peptide_tensor.shape)

# Saves tensor
path = '/content/drive/My Drive/Peptide MHC Project/embeddings/peptide_tensor_all.pt'
if os.path.exists(path):
  os.remove(path)
torch.save(peptide_tensor, path)


# In[ ]:


# Joins MHC Allele tensors, 140-179 
tensors_140_179 = []
MHC_dir_140_179 = '/content/drive/My Drive/Peptide MHC Project/embeddings/MHC_140_179'
for f in os.listdir(MHC_dir_140_179):
  tensor = torch.load(os.path.join(MHC_dir_140_179, f))
  tensors_140_179.append(tensor)

MHC_tensor_140_179 = torch.cat(tensors_140_179, dim=0)
print(MHC_tensor_140_179.shape)

# Saves tensor
path = '/content/drive/My Drive/Peptide MHC Project/embeddings/MHC_140_179_all.pt'
if os.path.exists(path):
  os.remove(path)
torch.save(MHC_tensor_140_179, path)


# In[ ]:


# Joins MHC Allele tensors, 50-84
# del MHC_tensor_140_179
MHC_dir_50_84 = '/content/drive/My Drive/Peptide MHC Project/embeddings/MHC_50_84'
# del peptide_tensor
tensors_50_84 = []
for f in os.listdir(MHC_dir_50_84):
  tensor = torch.load(os.path.join(MHC_dir_50_84, f))
  tensors_50_84.append(tensor)

MHC_tensor_50_84 = torch.cat(tensors_50_84, dim=0)
print(MHC_tensor_50_84.shape)

# Saves tensor
path = '/content/drive/My Drive/Peptide MHC Project/embeddings/MHC_50_84_all.pt'
if os.path.exists(path):
  os.remove(path)
torch.save(MHC_tensor_50_84, path)


# In[ ]:


# peptide_embedding_tensor = torch.load('embeddings/peptide_tensor_all.pt')
torch.save(peptide_embedding_tensor[:50000,:,:].clone(), 'embeddings/peptide_tensor_50k.pt')
torch.save(peptide_embedding_tensor[:20000,:,:].clone(), 'embeddings/peptide_tensor_20k.pt')
torch.save(peptide_embedding_tensor[:2000,:,:].clone(), 'embeddings/peptide_tensor_2k.pt')


# In[ ]:


# try:
#   allele_embedding_tensor = torch.cat((MHC_tensor_140_179, MHC_tensor_50_84), dim=1)
# except:
#   print("Loading saved embeddings")
#   alpha_140_179_embedding_tensor = torch.load('embeddings/MHC_140_179_all.pt')
#   print("loaded first")
#   alpha_50_84_embedding_tensor = torch.load('embeddings/MHC_50_84_all.pt')
#   print("loaded second")
#   allele_embedding_tensor = torch.cat((alpha_140_179_embedding_tensor, alpha_50_84_embedding_tensor), dim=1)
#   print("done")
#   del alpha_140_179_embedding_tensor
#   del alpha_50_84_embedding_tensor


# In[ ]:


# Saves tensors
allele_embedding_tensor = torch.cat((MHC_tensor_140_179, MHC_tensor_50_84), dim=1)
all_path = "embeddings/full_alleles_all.pt"
if not os.path.exists(all_path):
  torch.save(allele_embedding_tensor, all_path)
  print("saved whole")

mini_path = "embeddings/full_alleles_2k.pt"
if not os.path.exists(mini_path):
  mini = allele_embedding_tensor[:2000, :, :].clone()
  torch.save(mini, mini_path)
  print("saved mini")
  
subset_path = "embeddings/full_alleles_50k.pt"
if not os.path.exists(subset_path):
  subset = allele_embedding_tensor[:50000, :, :].clone()
  torch.save(subset, "embeddings/full_alleles_50k.pt")
  print("saved subset")


# In[ ]:




