import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fn
from torch.nn.init import normal_

class MLPLayers(nn.Module):
    r""" MLPLayers
    Args:
        - layers(list): a list contains the size of each layer in mlp layers
        - dropout(float): probability of an element to be zeroed. Default: 0
        - activation(str): activation function after each layer in mlp layers. Default: 'relu'.
                           candidates: 'sigmoid', 'tanh', 'relu', 'leekyrelu', 'none'
    Shape:
        - Input: (:math:`N`, \*, :math:`H_{in}`) where \* means any number of additional dimensions
          :math:`H_{in}` must equal to the first value in `layers`
        - Output: (:math:`N`, \*, :math:`H_{out}`) where :math:`H_{out}` equals to the last value in `layers`
    """
    def __init__(self, layers, dropout=0., activation='relu', bn=False, init_method=None):
        super(MLPLayers, self).__init__()
        self.layers = layers
        self.dropout = dropout
        self.activation = activation
        self.use_bn = bn
        self.init_method = init_method

        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))
            if self.use_bn:
                mlp_modules.append(nn.BatchNorm1d(num_features=output_size))
            activation_func = activation_layer(self.activation, output_size)
            if activation_func is not None:
                mlp_modules.append(activation_func)

        self.mlp_layers = nn.Sequential(*mlp_modules)
        if self.init_method is not None:
            self.apply(self.init_weights)

    def init_weights(self, module):
        # We just initialize the module with normal distribution as the paper said
        if isinstance(module, nn.Linear):
            if self.init_method == 'norm':
                normal_(module.weight.data, 0, 0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, input_feature):
        return self.mlp_layers(input_feature)


class SequenceAttLayer(nn.Module):
    """Attention Layer. Get the representation of each user in the batch.
    Args:
        queries (torch.Tensor): candidate ads, [B, H], H means embedding_size * feat_num
        keys (torch.Tensor): user_hist, [B, T, H]
        keys_length (torch.Tensor): mask, [B]
    Returns:
        torch.Tensor: result
    """
    def __init__(self, att_hidden_size=( 64, 16), activation='sigmoid', softmax_stag=False, return_seq_weight=True):
        super(SequenceAttLayer, self).__init__()
        self.att_hidden_size = att_hidden_size
        self.activation = activation
        self.softmax_stag = softmax_stag
        self.return_seq_weight = return_seq_weight
        self.activation = activation


    def forward(self, queries, keys, mask):
        embedding_size = queries.shape[-1]  # E
        hist_len = keys.shape[1]  # L
        queries = queries.repeat(1, hist_len, 1)
        queries = queries.view(-1, hist_len, embedding_size)

        # MLP Layer
        input_tensor = torch.cat([queries, keys, queries - keys, queries * keys], dim=-1).to(queries.device)  #[bsz,l,4emb]
        att_hidden_size = [input_tensor.shape[-1]] + list(self.att_hidden_size)
        self.att_mlp_layers = MLPLayers(att_hidden_size, activation=self.activation, bn=False).to(queries.device)
        self.dense = nn.Linear(att_hidden_size[-1], 1).to(queries.device)
        output = self.att_mlp_layers(input_tensor)
        output = torch.transpose(self.dense(output), -1, -2)  #[bsz,l,1]  --> [bsz,1,l]

        # get mask
        output = output.squeeze(1)

        # mask
        if self.softmax_stag:
            mask_value = -np.inf
        else:
            mask_value = 0.0

        output = output.masked_fill(mask=mask.squeeze(-1), value=torch.tensor(mask_value))
        output = output.unsqueeze(1)
        output = output / (embedding_size**0.5)

        # get the weight of each user's history list about the target item
        if self.softmax_stag:
            output = fn.softmax(output, dim=2)  # [B, 1, T]

        if not self.return_seq_weight:
            output = torch.matmul(output, keys)  # [B, 1, H]

        return output.squeeze(1)








