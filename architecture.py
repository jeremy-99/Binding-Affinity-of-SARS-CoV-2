# -*- coding: utf-8 -*-
import torch
import torch.nn as nn 

# Neural network model for processing peptide sequence embeddings and MHC allele embeddings
# and predicting binary (0 or 1) binding affinity as output

# Peptide and MHC pre-trained amino acid sequence embeddings as inputs
# Input embedding dimension = (batch_size * sequence_length * embedding_size)

# Model includes 3 components:
    # Component 1: peptide sequence embedding processing layer - bidirectional GRU
    # Component 2: MHC allele sequence embedding processing layer - bidirectional GRU
    # Component 3: Concatenation of hidden layers from components 1 and 2 and binding affinity prediction - 3 Feed Forward layers

class peptide_BiGRU(nn.Module):
    def __init__(self, config):
        super(peptide_BiGRU, self).__init__()
        self.input_size = config['embedding_dim']
        self.hidden_size = config['hidden_dim']
        self.peptide_bidir_gru = nn.GRU(self.input_size, self.hidden_size, batch_first=True, bidirectional=True)
        
        self.drop_input = nn.Dropout(p=config['dropout_input'], inplace=True)
        self.just_last_hidden = config['just_last_hidden']
        
    def forward(self, x):
        self.drop_input(x)  # Dropout
        # gru_output: (batch_size, seq_length, hidden_size*2), each ex's timesteps are = <forward, backward>
        # gru_hidden: (num_dirs, batch_size, hidden_size), each ex is < forward last ts, backward first ts>
        # Described at https://github.com/pytorch/pytorch/issues/3587
        gru_output, gru_hidden = self.peptide_bidir_gru(x)
        if self.just_last_hidden:
            # concatenation of forward on last timestep, backward on first timestep
            output = gru_hidden.reshape(-1, self.hidden_size*2)
        else:
            # concatenation of first forward, first backward, last forward, last backward
            output = torch.cat((gru_output[:,0,:], gru_output[:,-1,:]), dim=1)
        return output
    
class allele_BiGRU(nn.Module):
    def __init__(self, config):
        super(allele_BiGRU, self).__init__()
        self.input_size = config['embedding_dim']
        self.hidden_size = config['hidden_dim']
        self.allele_bidir_gru = nn.GRU(self.input_size, self.hidden_size, batch_first=True, bidirectional=True)
        self.drop_input = nn.Dropout(p=config['dropout_input'], inplace=True)
        self.just_last_hidden = config['just_last_hidden']
        
    def forward(self, x): 
        self.drop_input(x)
        gru_output, gru_hidden = self.allele_bidir_gru(x)
        if self.just_last_hidden:
            # concatenation of forward on last timestep, backward on first timestep
            output = gru_hidden.reshape(-1, self.hidden_size*2)
        else:
            # concatenation of first forward, first backward, last forward, last backward
            output = torch.cat((gru_output[:,0,:], gru_output[:,-1,:]), dim=1)
        # print(output.shape)
        return output
    
    
class Output_Layer(nn.Module):
    def __init__(self, peptide_BiGRU_model, allele_BiGRU_model, config):
        super(Output_Layer, self).__init__()
        self.peptide_model = peptide_BiGRU_model
        self.allele_model = allele_BiGRU_model
        
        self.input_dim = config['peptide_output_dim'] + config['allele_output_dim']
        self.hidden_1_dim = config['hidden_1_dim']
        self.hidden_2_dim = config['hidden_1_dim']
        
        self.Hidden1 = nn.Linear(self.input_dim, self.hidden_1_dim)
        self.Hidden2 = nn.Linear(self.hidden_1_dim, self.hidden_2_dim)
        self.output = nn.Linear(self.hidden_2_dim,1)
        self.relu = nn.ReLU()
        self.dropout_fc = nn.Dropout(p=config['dropout_fc'])
        # self.fc_layers = config['fc_layers']
        
    def forward(self, peptide_input, allele_input, three_layers=True, dropout=True):
        peptide_output = self.peptide_model(peptide_input)
        allele_output = self.allele_model(allele_input)
        
        input_concat = torch.cat((peptide_output, allele_output), dim=1)
        Hidden1 = self.dropout_fc(self.relu(self.Hidden1(input_concat)))
        if self.hidden_2_dim:
            Hidden2 = self.dropout_fc(self.relu(self.Hidden2(Hidden1)))
            output = torch.sigmoid(self.output(Hidden2))
        else:
            output = torch.sigmoid(self.output(Hidden1))
        return output

