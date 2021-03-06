import gym
import torch.nn.functional as F
from input_net_modules import InMLP, InCNN
from output_net_modules import OutMLP
from constants import DISCRETE, CONTINUOUS

import torch
import torch.nn as nn


class ValueNet(nn.Module):

    def __init__(self,
                 observation_space: gym.Space = None,
                 input_net_type: str = 'CNN',
                 shared_layers: torch.nn.Module = None,
                 hidden_nodes: int or list = [50, 50, 50],
                 nonlinearity: torch.nn.functional = F.relu
                 ):

        super(ValueNet, self).__init__()

        self.observation_space = observation_space

        # Add input module
        if shared_layers is None:

            if input_net_type.lower() == 'cnn' or input_net_type.lower() == 'visual':
                # Create CNN-NN to encode inputs
                self.input_module = InCNN(
                    input_sample=self.observation_space.sample(),
                )

            else:
                # Compute nr of input features for given gym env
                input_features = sum(self.observation_space.sample().shape)

                # Create MLP-NN to encode inputs
                self.input_module = InMLP(input_features=input_features,
                                          hidden_nodes=hidden_nodes,
                                          nonlinearity=nonlinearity)

        else:
            self.input_module = shared_layers

        # Automatically determine how many input nodes output module is gonna need to have
        input_features_output_module = self.input_module._modules[next(reversed(self.input_module._modules))].out_features

        # Add output module
        self.output_module = OutMLP(input_features=input_features_output_module,
                                    output_features=1,
                                    output_type=CONTINUOUS
                                    )


    def forward(self, x: torch.tensor):

        hidden = self.input_module(x)
        state_value = self.output_module(hidden)

        return state_value


    def get_non_output_layers(self):
        return self.input_module
