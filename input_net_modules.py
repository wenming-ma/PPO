import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO: Add RNN input module


class InCNN(nn.Module):

    def __init__(self,
                 input_sample: np.array,
                 hidden: list = None,
                 nonlinearity: torch.nn.functional = F.relu
                 ):
        super(InCNN, self).__init__()

        print(input_sample.shape)
        data_height = input_sample.shape[0]
        data_width = input_sample.shape[1]

        if hidden is None:
            hidden = [
                # Dicts for conv layers
                {
                    'in_channels': 3,  # 3 color channels
                    'out_channels': 16,  # 16 output channels = 16 filters
                    'kernel_size': 8,  # 8x8 kernel/filter size
                    'stride': 4
                },
                {
                    'in_channels': 16,
                    'out_channels': 32,
                    'kernel_size': 4,
                    'stride': 2
                },
                # Nr. of nodes for fully connected layers
                256
            ]

        self.pipeline = []

        for i, layer_specs in enumerate(hidden):
            if isinstance(layer_specs, dict):
                # Add conv layer
                self.pipeline.append(
                    nn.Conv2d(in_channels=layer_specs['in_channels'],
                              out_channels=layer_specs['out_channels'],
                              kernel_size=layer_specs['kernel_size'],
                              stride=layer_specs['stride']
                              )
                )

            elif isinstance(layer_specs, int) and isinstance(hidden[i-1], dict):
                # Add flattening before transitioning from conv to FC

                # Compute output size of conv layers
                for l in range(i):
                    data_height = self.out_dim(dim_in=data_height,
                                               pad=hidden[l]['padding'] if 'padding' in hidden[l].keys() else 0,
                                               dial=hidden[l]['dilation'] if 'dilation' in hidden[l].keys() else 1,
                                               k=hidden[l]['kernel_size'],
                                               stride=hidden[l]['stride'] if 'stride' in hidden[l].keys() else 1)
                    data_width = self.out_dim(dim_in=data_width,
                                              pad=hidden[l]['padding'] if 'padding' in hidden[l].keys() else 0,
                                              dial=hidden[l]['dilation'] if 'dilation' in hidden[l].keys() else 1,
                                              k=hidden[l]['kernel_size'],
                                              stride=hidden[l]['stride'] if 'stride' in hidden[l].keys() else 1)

                data_height, data_width = int(data_height), int(data_width)

                flattened_size = data_height * data_width

                # Add flattening layer
                self.pipeline.append(
                    nn.Flatten(start_dim=1)
                )

                # Add FC layer
                self.pipeline.append(
                    nn.Linear(flattened_size, hidden[i])
                )

            else:
                # Add FC layer after a previous one has been added already
                self.pipeline.append(
                    nn.Linear(hidden[i-1], hidden[i])
                )

        # Register all layers
        for i, layer in enumerate(self.pipeline):
            self.add_module("layer_cnn_in_" + str(i), layer)

        self.nonlinearity = nonlinearity


    def out_dim(self, dim_in, pad, dial, k, stride):
        print(dim_in, pad, dial, k, stride)
        return torch.floor(torch.tensor((dim_in + 2*pad - dial*(k-1) - 1)/stride + 1)).numpy()

    def forward(self, x):

        print('Input x:\n', x)

        for layer in self.pipeline:
            x = self.nonlinearity(layer(x))
            print('Modified x:\n', x)

        return x


class InMLP(nn.Module):
    def __init__(self,
                 input_features: int,
                 hidden_nodes: int or list = [50, 50, 50],
                 nonlinearity: torch.nn.functional = F.relu
                 ):
        super(InMLP, self).__init__()

        # Construct NN-processing pipeline consisting of concatenation of layers to be applied to any input
        self.pipeline = [

            # Add input layer
            nn.Linear(
                input_features,
                hidden_nodes[0] if isinstance(hidden_nodes, list) else hidden_nodes
            )

        ]

        # Add optional hidden layers
        if isinstance(hidden_nodes, list):
            for i in range(1, len(hidden_nodes)):
                self.pipeline.append(
                    nn.Linear(hidden_nodes[i-1], hidden_nodes[i])
                )

        # Register all layers
        for i, layer in enumerate(self.pipeline):
            self.add_module("layer_mlp_in_" + str(i), layer)

        self.nonlinearity = nonlinearity


    def forward(self, x):

        for layer in self.pipeline:
            x = self.nonlinearity(layer(x))

        return x


# Testing
#import gym
#env = gym.make('CartPole-v1')
#observation = env.observation_space.sample()
#observation_shape = torch.tensor(observation.shape)
#net = MLP_Module(observation_shape[0])
#net(torch.tensor(observation))
