import torch                  # PyTorch main module
import torch.nn as nn         # Neural Network classes
import torch.nn.functional as f
import torch.optim as optim   # Optimization algorithms
from torch.utils.tensorboard import SummaryWriter
from torch.nn import Linear, BatchNorm1d

import torch_geometric
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.nn import GCNConv, GATConv, global_max_pool, global_mean_pool
from torch_geometric.data import DataLoader


import numpy as np
import matplotlib.pyplot as plt

import random

import preprocess

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
    
print(device)


def predict_random_graph(dataset, model, device):
    model.eval()
    
    idx = random.randint(0, len(dataset) - 1)
    sample_graph = dataset[idx]
    actual_label = sample_graph.y.item()
    
    sample_loader = DataLoader([sample_graph], batch_size=1)
    batch = next(iter(sample_loader)).to(device)
    
    with torch.no_grad():
        out = model(batch) 
        
        probabilities = torch.exp(out)
        
        confidence, predicted_class = torch.max(probabilities, dim=1)
        
        confidence = confidence.item()
        predicted_class = predicted_class.item()
        
    print(f"\n--- Random Graph Prediction (Index {idx}) ---")
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence:      {confidence * 100:.2f}%")
    print(f"Actual Class:    {actual_label}")


data = preprocess.SHREC_Dataset("data/shrec_16/")
print(data.get_summary())

train_loader = DataLoader(data, batch_size=32, shuffle=True)

data = data.shuffle()

total_graphs = len(data)
train_size = int(0.8 * total_graphs)
val_size = int(0.1 * total_graphs)

train_dataset = data[:train_size]
val_dataset = data[train_size:train_size + val_size]
test_dataset = data[train_size + val_size:]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super().__init__()
        
        self.conv1 = GATConv(num_node_features, 32, heads=4, concat=False)
        self.bn1 = BatchNorm1d(32) 
        
        self.conv2 = GATConv(32, 32, heads=4, concat=False)
        self.bn2 = BatchNorm1d(32) 
        
        self.lin = Linear(32 * 2, num_classes)


    def forward(self, batch_data):
        x, edge_index, batch_idx = batch_data.x, batch_data.edge_index, batch_data.batch

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = f.relu(x)
        x = f.dropout(x, p=0.5, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = f.relu(x)

        x_mean = global_mean_pool(x, batch_idx) 
        x_max = global_max_pool(x, batch_idx)
        
        x = torch.cat([x_mean, x_max], dim=1) 
        
        x = self.lin(x)

        return f.log_softmax(x, dim=1)

def evaluate(loader, model):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            pred = out.argmax(dim=1)
            correct += int((pred == batch.y).sum())
            total += batch.y.size(0)
            
    return correct / total

model = GCN(data.num_node_features, data.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

writer = SummaryWriter(log_dir='logs/runs/')

for epoch in range(400):
    model.train()
    total_loss = 0

    for batch in train_loader:
        batch = batch.to(device)

        optimizer.zero_grad()
        out = model(batch)
        loss = f.nll_loss(out, batch.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    val_acc = evaluate(val_loader, model)

    writer.add_scalar('Training Loss', avg_loss, epoch)
    writer.add_scalar('Validation Accuracy', val_acc, epoch)

    print(f'Epoch: {epoch:03d}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}')

writer.close()

model.eval()
correct = 0
total = 0

test_acc = evaluate(test_loader, model)
print(f'Test Accuracy: {test_acc:.4f}')

print("============ Sample Classification ============")
predict_random_graph(test_dataset, model, device)
predict_random_graph(test_dataset, model, device)
predict_random_graph(test_dataset, model, device)
