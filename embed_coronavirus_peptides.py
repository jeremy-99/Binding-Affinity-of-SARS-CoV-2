#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


get_ipython().system('pip3 install torch==1.2.0+cu92 torchvision==0.4.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html')


# In[ ]:


import os
os.chdir('/content/drive/My Drive/Peptide MHC Project')


# In[ ]:


import torch
import pandas as pd
from peptide_embedding import *  
from MHC_sequence_embedding import *
import random
print(torch.__version__)


# In[ ]:


# Load coronavirus peptides
corona_df = pd.read_csv('coronavirus/all_corona_peptides.txt', header=None)
corona_df.columns = ['peptide']
corona_encoding = peptide_encoding(corona_df.squeeze())
# print(corona_df.squeeze().iloc[:2])
# print(corona_encoding[:2])


# In[ ]:


# Generates embeddings of coronavirus peptides
corona_embedding_list = []
pretrained_model = torch.load('ssa_L1_100d_lstm3x512_lm_i512_mb64_tau0.5_lambda0.1_p0.05_epoch100.sav')

# TO DO: remove for loop loop bc there are so few peptides
for i in range(0,len(corona_encoding),1000):
  corona_embeddings = peptide_embedding(corona_encoding[i:i+1000], 15, pretrained_model)
  corona_embeddings = torch.stack(corona_embeddings)
  corona_embedding_list.append(corona_embeddings)
  print(i)

peptide_tensor = torch.cat(corona_embedding_list, dim=0)
path = '/content/drive/My Drive/Peptide MHC Project/coronavirus/all_corona_embeddings.pt'
torch.save(peptide_tensor, path)


# In[ ]:


# Gets distribution of peptide lengths
peptides = corona_df['peptide'].to_list()
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


# Randomly choose a subset of peptides
lens = list(range(8, 16))
peps_by_len = [[] for _ in range(8)]
for pep in peptides:
  for length in lens:
    if len(pep) == length:
      peps_by_len[length-8].append(pep)

# More 8-10mers than 11-15mers
select = {8: 100, 9: 250, 10: 150, 11: 25, 12: 25, 13: 25, 14: 25, 15: 25}
peps_subset = []
for i in range(8):
  subset = random.choices(peps_by_len[i], k=select[i+8])
  peps_subset.extend(subset)

print(len(peps_subset))


# In[ ]:


# Saves 
with open("coronavirus/subset_corona_peptides.txt", "w") as f:
    for pep in peps_subset:
        f.write(pep + '\n')

# Convert back to pandas
corona_subset = pd.Series(peps_subset)

# Encodes each peptide
corona_subset_encoding = peptide_encoding(corona_subset)

# Embeds each peptide encoding as a (MaxLen * EmbeddingDim) tensor
peptide_embedding_list = peptide_embedding(corona_subset_encoding, 15, pretrained_model)
peptide_embeddings = torch.stack(peptide_embedding_list)

# Save
path = '/content/drive/My Drive/Peptide MHC Project/coronavirus/subset_corona_embeddings.pt'
torch.save(peptide_embeddings, path)


# In[ ]:


# Embed some MHC alleles
# Dataset of peptide sequence, MHC allele name, binary binding affinity (positive, negative)
link1 = 'https://raw.githubusercontent.com/cmb-chula/MHCSeqNet/master/cleaned_MHC_all_classes.csv'
x = pd.read_csv(link1)

# Dataset of corresponding amino acid sequence for MHC alleles (Beta sheet, alpha helix res 140-179, alpha helix res 50-84)
link2 = 'https://raw.githubusercontent.com/cmb-chula/MHCSeqNet/master/PretrainedModels/sequence_model/AlleleInformation.txt'
allele_seq = urllib.request.urlopen(link2)
MHC_sequence_df = MHC_seq_df(allele_seq)

# Finds alleles in training set with known amino acids
alleles = x['allele']
good_idx = alleles.isin(MHC_sequence_df['MHC_allele'])
classI_alleles = alleles[good_idx]  


# In[ ]:


# Orders alleles whose amino acid sequences are known by frequency
most_common = classI_alleles.value_counts().index.tolist()

# Chooses 10 most common alleles from each HLA class
HLA_As = []
HLA_Bs = []
HLA_Cs = []
for allele in most_common:
  if 'HLA-A' in allele and len(HLA_As) < 10:
    HLA_As.append(allele)
  elif 'HLA-B' in allele and len(HLA_Bs) < 10:
    HLA_Bs.append(allele)
  elif 'HLA-C' in allele and len(HLA_Cs) < 10:
    HLA_Cs.append(allele)

chosen_alleles = HLA_As + HLA_Bs + HLA_Cs
print(len(chosen_alleles))


# In[ ]:


# Gets the alleles' amino acid sequences
aa_seqs = MHC_sequence_df.loc[MHC_sequence_df['MHC_allele'].isin(chosen_alleles)]
aa_50_84 = aa_seqs['Alpha_helix_res_50-84']
aa_140_179 = aa_seqs['Alpha_helix_res_140-179']
# Saves
aa_seqs.to_csv("allele_sequences/30_mhc_alleles.csv")

# Encodes the amino acid sequences for each MHC allele
alpha_res_50_84_encoding = peptide_encoding(aa_50_84)
alpha_res_140_179_encoding = peptide_encoding(aa_140_179)

# Embeds each encoding as a (MaxLen * EmbeddingDim) tensor, then stacks all
embedding_list_50_84 = peptide_embedding(alpha_res_50_84_encoding, 53, pretrained_model)
embeddings_50_84 = torch.stack(embedding_list_50_84)

embedding_list_140_179 = peptide_embedding(alpha_res_140_179_encoding, 53, pretrained_model)
embeddings_140_179 = torch.stack(embedding_list_140_179)

# Save
path1 = '/content/drive/My Drive/Peptide MHC Project/embeddings/common_50_84_embeddings.pt'
torch.save(embeddings_50_84, path1)
path2 = '/content/drive/My Drive/Peptide MHC Project/embeddings/common_140_179_embeddings.pt'
torch.save(embeddings_140_179, path2)


# In[ ]:




