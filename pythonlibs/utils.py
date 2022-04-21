import math
import torch
import numpy as np
import pandas as pd
import torch.nn as nn



class BCELossAccuracy():
    def __init__(self):
        self.loss_function = nn.BCELoss()
    
    @staticmethod
    def accuracy(y_hat, labels):
        with torch.no_grad():
            y_tilde = (y_hat > 0.5).int()
            accuracy = (y_tilde == labels.int()).float().mean().item()
        return accuracy
    
    def __call__(self, y_hat, labels):
        loss = self.loss_function(y_hat, labels)
        accuracy = self.accuracy(y_hat, labels)
        return loss, accuracy


class CELossAccuracy():
    def __init__(self):
        self.loss_function = nn.CrossEntropyLoss()
    
    @staticmethod
    def accuracy(y_hat, labels):
        with torch.no_grad():
            y_tilde = y_hat.argmax(axis=1)
            accuracy = (y_tilde == labels).float().mean().item()
        return accuracy
    
    def __call__(self, y_hat, labels):
        loss = self.loss_function(y_hat, labels)
        accuracy = self.accuracy(y_hat, labels)
        return loss, accuracy

      
