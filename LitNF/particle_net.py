import os
import logging
import requests
import time
import functools
import pathlib
import shutil

import awkward


import numpy as np
import pandas as pd

import torch
import torch_geometric
import tqdm.auto as tqdm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

'''
    Downloading the Dataset
'''


def pad_array(jagged_array, max_len, value=0., dtype='float32'):
    rectangluar_array = np.full(shape=(len(jagged_array), max_len), fill_value=value, dtype=dtype)
    for idx, jagged_element in enumerate(jagged_array):
        if len(jagged_element) != 0:
            trunc = jagged_element[:max_len].astype(dtype)
            rectangluar_array[idx, :len(trunc)] = trunc
    return rectangluar_array


'''
    Preparing the Dataset
'''


class ParticleStaticEdgeConv(torch_geometric.nn.MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(ParticleStaticEdgeConv, self).__init__(aggr='max')
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * in_channels, out_channels[0], bias=False),
            torch_geometric.nn.BatchNorm(out_channels[0]), 
            torch.nn.ReLU(),
            torch.nn.Linear(out_channels[0], out_channels[1], bias=False),
            torch_geometric.nn.BatchNorm(out_channels[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(out_channels[1], out_channels[2], bias=False),
            torch_geometric.nn.BatchNorm(out_channels[2]),
            torch.nn.ReLU()
        )

    def forward(self, x, edge_index, k):
        
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, edge_index, x_i, x_j):
        tmp = torch.cat([x_i, x_j - x_i], dim = 1)

        out_mlp = self.mlp(tmp)

        return out_mlp

    def update(self, aggr_out):
        return aggr_out

class ParticleDynamicEdgeConv(ParticleStaticEdgeConv):
    def __init__(self, in_channels, out_channels, k=7):
        super(ParticleDynamicEdgeConv, self).__init__(in_channels, out_channels)
        self.k = k
        self.skip_mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels, out_channels[2], bias=False),
            torch_geometric.nn.BatchNorm(out_channels[2]),
        )
        self.act = torch.nn.ReLU()

    def forward(self, pts, fts, batch=None):
        edges = torch_geometric.nn.knn_graph(pts, self.k, batch, loop=False, flow=self.flow)
        aggrg = super(ParticleDynamicEdgeConv, self).forward(fts, edges, self.k)
        x = self.skip_mlp(fts)
        out = torch.add(aggrg, x)
        return self.act(out)


class ParticleNet(torch.nn.Module):

    def __init__(self, settings):
        super().__init__()
        previous_output_shape = settings['input_features']

        self.input_bn = torch_geometric.nn.BatchNorm(settings['input_features'])

        self.conv_process = torch.nn.ModuleList()
        for layer_idx, layer_param in enumerate(settings['conv_params']):
            K, channels = layer_param
            self.conv_process.append(ParticleDynamicEdgeConv(previous_output_shape, channels, k=K).to(DEVICE))
            previous_output_shape = channels[-1]



        self.fc_process = torch.nn.ModuleList()
        for layer_idx, layer_param in enumerate(settings['fc_params']):
            drop_rate, units = layer_param
            seq = torch.nn.Sequential(
                torch.nn.Linear(previous_output_shape, units),
                torch.nn.Dropout(p=drop_rate),
                torch.nn.ReLU()
            ).to(DEVICE)
            self.fc_process.append(seq)
            previous_output_shape = units


        self.output_mlp_linear = torch.nn.Linear(previous_output_shape, settings['output_classes'])
        self.output_activation = torch.nn.Softmax(dim=1)

    def forward(self, batch):
        # fts = self.input_bn(batch.x)
        batch=batch[:,:90].reshape(-1,30,3)
        pts = batch[:,:,:2]
        fts =batch[:,:,3]

        for idx, layer in enumerate(self.conv_process):
          fts = layer(pts, fts, batch)
          pts = fts

        x = torch_geometric.nn.global_mean_pool(fts, batch)

        for layer in self.fc_process:
            x = layer(x)

        x = self.output_mlp_linear(x)
        # x = self.output_activation(x)
        return x



# model = ParticleNet(settings)
# model = model.to(DEVICE)

# # 