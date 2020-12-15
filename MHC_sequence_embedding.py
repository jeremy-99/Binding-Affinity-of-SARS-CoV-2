# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 13:07:15 2020

@author: dfrid
"""
import urllib
import numpy as np
import pandas as pd
import re

# link = 'https://raw.githubusercontent.com/cmb-chula/MHCSeqNet/master/PretrainedModels/sequence_model/AlleleInformation.txt'
# allele_sequences = urllib.request.urlopen(link)

# link = 'https://raw.githubusercontent.com/cmb-chula/MHCSeqNet/master/cleaned_MHC_all_classes.csv'
# x = pd.read_csv(link)

# alleles = x['allele']
# unique_alleles = np.unique(x['allele'])
# allele_list = list(unique_alleles)

def MHC_seq_df(allele_seq):
    MHC_sequences = []
    for line in allele_seq:
        decoded_line = line.decode("utf-8").split(',')
        MHC_sequences.append(decoded_line) 
    
    MHC_sequence_df = pd.DataFrame(MHC_sequences, columns = ['MHC_allele', 'Beta_sheet_res_3-125', 'Alpha_helix_res_140-179', 'Alpha_helix_res_50-84'])
    return MHC_sequence_df
    

# for i,seq in enumerate(MHC_sequence_df['Alpha_helix_res_140-179']):
#     foo = re.search('-{6}[A-Z]{4}-{5}',seq)
#     if foo != None:
#         MHC_sequence_df['Alpha_helix_res_140-179'][i] = re.sub('-','',seq)
        
# bad_index = []
# for i,seq in enumerate(MHC_sequence_df['Alpha_helix_res_140-179']):
#     foo = re.search('-',seq)
#     if foo != None:
#         bad_index.append(i)
        
# MHC_sequence_df = MHC_sequence_df.drop(bad_index,axis='index')

# def bad_idx(alleles):
#     bad_index = []
#     for i, allele in enumerate(alleles):
#         foo = re.search('^HLA-D',allele)
#         if foo != None:
#             bad_index.append(i)   
#     # classI_alleles = alleles.drop(bad_index, axis='index')
#     return bad_index


def allele_sequence(classI_alleles, MHC_sequence_df):
    alpha_res_140_179 = [None]*len(classI_alleles)
    alpha_res_50_84 = [None]*len(classI_alleles)
    count = 0
    for allele in classI_alleles:
        print(count)
        allele_bool = (MHC_sequence_df['MHC_allele'] == allele)
        alpha_140_179 = MHC_sequence_df['Alpha_helix_res_140-179'][allele_bool]
        alpha_50_84 = MHC_sequence_df['Alpha_helix_res_50-84'][allele_bool]
    
        #alpha_res_140_179.append(list(alpha_140_179)[0])
        #alpha_res_50_84.append(list(alpha_50_84)[0])
        alpha_res_140_179[count] = list(alpha_140_179)[0]
        alpha_res_50_84[count] = list(alpha_50_84)[0]
        count = count + 1
    
    alpha_res_140_179 = pd.DataFrame(alpha_res_140_179).squeeze()
    alpha_res_50_84 = pd.DataFrame(alpha_res_50_84).squeeze()
    return alpha_res_140_179, alpha_res_50_84

# alpha_140_179_df = pd.DataFrame(list1).squeeze()
# alpha_50_84_df = pd.DataFrame(list2).squeeze()

# peptide_embedding(alpha_140_179_df, 53)
# peptide_emebedding(alpha_140_179_df, 36)



#%%

"""
def one_hot_allele_vec(alleles, unique_alleles):
    unique_allele_list = list(unique_alleles)
    allele_one_hot = []
    for allele in alleles:
        idx = unique_allele_list.index(allele)
        one_hot_vec = np.zeros(len(unique_allele_list))
        one_hot_vec[idx] = 1.0
        allele_one_hot.append(one_hot_vec)
    return allele_one_hot


def one_hot_allele_idx(alleles, unique_alleles):
    unique_allele_list = list(unique_alleles)
    allele_idx = []
    for allele in alleles:
        idx = unique_allele_list.index(allele)
        allele_idx.append(idx)
    return allele_idx

"""