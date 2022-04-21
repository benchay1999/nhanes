import torch
import torch.nn as nn
class Classifier(nn.Module):
    def __init__(self, n_layers, n_inputs, n_hidden_units, n_classes):
        super(Classifier, self).__init__()
        layers = []

        if n_layers == 1: # Logistic Regression
            if n_classes > 2:
                layers.append(nn.Linear(n_inputs, n_classes))
            else:
                layers.append(nn.Linear(n_inputs, 1))
        
        else : #Deep  Neural Network
            layers.append(nn.Linear(n_inputs, n_hidden_units))
            layers.append(nn.ReLU())
            for i in range(n_layers-2):
                layers.append(nn.Linear(n_hidden_units, n_hidden_units))
                layers.append(nn.ReLU())
            
            if n_classes > 2:
                layers.append(nn.Linear(n_hidden_units, n_classes))
            else:
                layers.append(nn.Linear(n_hidden_units, 1))
        
        if n_classes > 2:
            layers.append(nn.Softmax(dim=1))
        else:
            layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
