import random
import IPython
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from utils import BCELossAccuracy, CELossAccuracy

def train(data_loader, X_test, y_test, net, optimizer, lr_scheduler, device, n_epochs=200):
    n_classes = 2
    cross_entropy_loss = CELossAccuracy if n_classes > 2 else BCELossAccuracy()
    costs = []
    for epoch in range(n_epochs):
        for i, (x_batch, y_batch) in enumerate(data_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            Yhat = net(x_batch)
            Ytilde = torch.round(Yhat.detach().reshape(-1))
            cost=0

            # loss
            loss, acc = cross_entropy_loss(Yhat.squeeze(), y_batch)
            cost += loss

            optimizer.zero_grad()
            if (torch.isnan(cost)).any():
                continue
            cost.backward()
            optimizer.step()
            costs.append(loss.item())

            # Print the cost per 10 batches
            if (i + 1) % 10 == 0 or (i + 1) == len(data_loader):
                print('Epoch [{}/{}], Batch [{}/{}], Cost: {:.4f}'.format(epoch+1, n_epochs,
                                                                          i+1, len(data_loader),
                                                                          cost.item()), end='\r')
        if lr_scheduler is not None:
            lr_scheduler.step()
    
    Yhat_test = net(X_test).squeeze()
    test_loss, test_acc = cross_entropy_loss(Yhat_test, y_test)
    
    
    return (acc, test_acc), costs, net