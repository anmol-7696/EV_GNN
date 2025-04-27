import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import numpy as np

# 1. Load and preprocess a small part of the dataset
def load_traffic_data(folder_path):
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    dfs = []
    for file in files[:10]:  # Just a few files for demo purposes
        df = pd.read_csv(file)
        dfs.append(df)
    data = pd.concat(dfs)
    return data

folder_path = "/Users/anmolpreetsingh/Desktop/trafficData.csv"  # Change this
traffic_data = pd.read_csv(folder_path)

print("Loaded data:", traffic_data.shape)

# 2. Basic preprocessing
traffic_data = traffic_data.dropna()
traffic_data['TIMESTAMP'] = pd.to_datetime(traffic_data['TIMESTAMP'])
traffic_data = traffic_data.sort_values(by='TIMESTAMP')

# Take one snapshot for simplicity
snapshot = traffic_data[traffic_data['TIMESTAMP'] == traffic_data['TIMESTAMP'].iloc[0]]

# 3. Create node features
# Example: use 'AvgSpeed' and 'VehicleCount' as node features
node_features = snapshot[['avgSpeed', 'vehicleCount']].values
x = torch.tensor(node_features, dtype=torch.float)

# Safe edge building
num_nodes = x.shape[0]
edge_index = []

if num_nodes > 1:  # Need at least two nodes to build edges
    k = min(3, num_nodes - 1)  # Each node connects to up to 3 others

    for i in range(num_nodes):
        neighbors = np.random.choice(
            [j for j in range(num_nodes) if j != i],  # no self-loops
            size=k,
            replace=False
        )
        for j in neighbors:
            edge_index.append([i, j])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
else:
    print("Not enough nodes to create edges.")
    edge_index = torch.empty((2, 0), dtype=torch.long)  # Empty edge_index

print("Edge index shape:", edge_index.shape)


# 5. Define target
# Let's say we want to predict 'VehicleCount'
y = torch.tensor(snapshot['vehicleCount'].values, dtype=torch.float)

# 6. Create a PyTorch Geometric data object
data = Data(x=x, edge_index=edge_index, y=y)

# 7. Define a simple GCN
class TrafficGCN(torch.nn.Module):
    def __init__(self):
        super(TrafficGCN, self).__init__()
        self.conv1 = GCNConv(2, 16)  # 2 input features -> 16 hidden
        self.conv2 = GCNConv(16, 1)  # 16 hidden -> 1 output (VehicleCount prediction)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x.squeeze()

# 8. Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TrafficGCN().to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()
for epoch in range(201):
    optimizer.zero_grad()
    out = model(data)
    loss = F.mse_loss(out, data.y)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f'Epoch {epoch} | Loss: {loss.item():.4f}')

# 9. Done! You have a working model.
