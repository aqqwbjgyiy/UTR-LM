import torch
import torch.nn as nn
from esm.model.esm2_secondarystructure import ESM2 as ESM2_SISS
from esm.model.esm2_supervised import ESM2
from esm.model.esm2_only_secondarystructure import ESM2 as ESM2_SS

class CNN_linear(nn.Module):
    def __init__(self, args, embed_dim, inp_len, layers, heads, alphabet):
        super(CNN_linear, self).__init__()
        
        self.embedding_size = embed_dim
        self.border_mode = 'same'
        self.inp_len = inp_len
        self.nodes = args.nodes
        self.cnn_layers = args.cnn_layers
        self.filter_len = 8
        self.nbr_filters = 120
        self.dropout1 = 0
        self.dropout2 = 0
        self.dropout3 = args.dropout3
        
        if 'SISS' in args.modelfile:
            self.esm2 = ESM2_SISS(num_layers=layers,
                               embed_dim=embed_dim,
                               attention_heads=heads,
                               alphabet=alphabet)
        elif 'SS' in args.modelfile:
            self.esm2 = ESM2_SS(num_layers=layers,
                             embed_dim=embed_dim,
                             attention_heads=heads,
                             alphabet=alphabet)
        else:
            self.esm2 = ESM2(num_layers=layers,
                          embed_dim=embed_dim,
                          attention_heads=heads,
                          alphabet=alphabet)
        
        self.conv1 = nn.Conv1d(in_channels=self.embedding_size,
                            out_channels=self.nbr_filters,
                            kernel_size=self.filter_len,
                            padding=self.border_mode)
        self.conv2 = nn.Conv1d(in_channels=self.nbr_filters,
                            out_channels=self.nbr_filters,
                            kernel_size=self.filter_len,
                            padding=self.border_mode)
        
        self.dropout1 = nn.Dropout(self.dropout1)
        self.dropout2 = nn.Dropout(self.dropout2)
        self.dropout3 = nn.Dropout(self.dropout3)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        
        if args.avg_emb or args.bos_emb:
            self.fc = nn.Linear(in_features=embed_dim, out_features=self.nodes)
            self.linear = nn.Linear(in_features=self.nbr_filters, out_features=self.nodes)
        else:
            self.fc = nn.Linear(in_features=inp_len * embed_dim, out_features=self.nodes)
            self.linear = nn.Linear(in_features=inp_len * self.nbr_filters, out_features=self.nodes)
            
        self.output = nn.Linear(in_features=self.nodes, out_features=1)
        if self.cnn_layers == -1:
            self.direct_output = nn.Linear(in_features=embed_dim, out_features=1)
        if args.magic:
            self.magic_output = nn.Linear(in_features=1, out_features=1)
            
    def forward(self, tokens, args, layers, need_head_weights=True, return_contacts=True, return_representation=True):
        x = self.esm2(tokens, [layers], need_head_weights, return_contacts, return_representation)
        
        if args.avg_emb:
            x = x["representations"][layers][:, 1:self.inp_len+1].mean(1)
            x_o = x.unsqueeze(2)
        elif args.bos_emb:
            x = x["representations"][layers][:, 0]
            x_o = x.unsqueeze(2)
        else:
            x_o = x["representations"][layers][:, 1:self.inp_len+1]
            x_o = x_o.permute(0, 2, 1)

        if self.cnn_layers >= 1:
            x_cnn1 = self.conv1(x_o)
            x_o = self.relu(x_cnn1)
        if self.cnn_layers >= 2:
            x_cnn2 = self.conv2(x_o)
            x_relu2 = self.relu(x_cnn2)
            x_o = self.dropout1(x_relu2)
        if self.cnn_layers >= 3:
            x_cnn3 = self.conv2(x_o)
            x_relu3 = self.relu(x_cnn3)
            x_o = self.dropout2(x_relu3)
        
        x = self.flatten(x_o)
        if self.cnn_layers != -1:
            if self.cnn_layers != 0:
                o_linear = self.linear(x)
            else:
                o_linear = self.fc(x)
            o_relu = self.relu(o_linear)
            o_dropout = self.dropout3(o_relu)
            o = self.output(o_dropout)
        else:
            o = self.direct_output(x)
            
        if args.magic:
            o = self.magic_output(o)
        return o