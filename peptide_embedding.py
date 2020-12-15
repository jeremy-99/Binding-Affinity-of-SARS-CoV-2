# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 13:36:53 2020

@author: dfrid
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence

import torch.utils.data
from src.alphabets import Uniprot21#, SecStr8  # SecStr8 unused
from src.utils import pack_sequences, unpack_sequences
# import src.pdb as pdb  # Unused!!!

# 
# link = 'https://raw.githubusercontent.com/cmb-chula/MHCSeqNet/master/cleaned_MHC_all_classes.csv'

# x = pd.read_csv(link)

# peptides = x['peptide']

#
def encode_sequence(x, alphabet):
    # convert to bytes and uppercase
    x = x.encode('utf-8').upper()
    # convert to alphabet index
    x = alphabet.encode(x)
    return x

def peptide_encoding(peptides):
    # Load amino acid alphabet 
    alphabet = Uniprot21()
    # Convert peptide sequences to list of amino acid strings
    peptides = peptides.values.tolist()

    # Encode peptide sequences as arrays of amino acid indexes
    peptide_list = []
    for peptide in peptides:
        peptide_list.append(encode_sequence(peptide, alphabet))
        
    return peptide_list

def allele_encoding(alleles):
    # Load amino acid alphabet 
    alphabet = Uniprot21()

    # Encode peptide sequences as arrays of amino acid indexes
    peptide_list = []
    for peptide in peptides:
        peptide_list.append(encode_sequence(peptide, alphabet))
        
    return peptide_list

    
# Load trained full SSA model for structure-based protein embeddings
# features = torch.load('ssa_L1_100d_lstm3x512_lm_i512_mb64_tau0.5_lambda0.1_p0.05_epoch100.sav')
# features.eval()

# Embed peptide sequences
def peptide_embedding(peptide_list, max_len, pretrained_model):
    #peptide_list = peptide_encoding(peptides)
    c = [torch.from_numpy(peptide).long() for peptide in peptide_list]
    c,order = pack_sequences(c)

    # Load trained full SSA model for structure-based protein embeddings
    features = pretrained_model
    # torch.load('pretrained_models/ssa_L1_100d_lstm3x512_lm_i512_mb64_tau0.5_lambda0.1_p0.05_epoch100.sav')
    #features.eval()
    
    z = features(c)
    z = unpack_sequences(z, order)
    
    # pad sequences to be of same sequnece length 
    maxLength = max_len
    Embedding_List = []
    for elem in z:
        diff = maxLength - elem.shape[0]
        elem = F.pad(elem, (0, 0, 0, diff), 'constant')
        Embedding_List.append(elem)

    # Embedding_List = []
    # for embedding in z:
    #    Embedding_List.append(embedding.detach().numpy())
        
    return Embedding_List


    
#%%
# binding_affinity = x['binding_quality']
# binding_affinity = binding_affinity.values.tolist()

def binding_affinity_class(binding_affinity):
    binding_affinity_class = []
    for i in binding_affinity:
        if i == 'Negative':
            affinity_index = 0
        elif i == 'Positive':
            affinity_index = 1
        binding_affinity_class.append(affinity_index)
    return binding_affinity_class