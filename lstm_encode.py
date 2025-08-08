# 1.3 - 20 epochs instead of 15 and used new learning rate from hyperparam sweeps

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import sys
sys.path.append("C:/Users/tobys/Downloads/GBM-ML-main/GBM-ML-main")

'''
This section is copied from the original encode.py script.
This is all of our data loading and processing.
'''

# Process data
lcs = pd.read_csv('lcs.csv')
channels = ['n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'na', 'nb', 'b0', 'b1']

# Fill missing channels with noise
for channel in channels: 
    missing_indices = lcs[channel].isnull()  
    num_missing = missing_indices.sum()
    noise = np.random.normal(loc=lcs[channel].mean(), scale=lcs[channel].std(), size=num_missing)  
    lcs.loc[missing_indices, channel] = noise   

time_series_list = []
burst_ids = []
grouped = lcs.groupby('burst')
for burst, group in grouped:
    time_series_data = group[channels].values
    time_series_tensor = torch.tensor(time_series_data, dtype=torch.float32)
    time_series_list.append(time_series_tensor)
    burst_ids.append(burst)

# Padding with zeros
time_series_list = nn.utils.rnn.pad_sequence(time_series_list, batch_first=True, padding_value=0.0)

# Normalize the light curves
scaler = StandardScaler()
time_series_list_2d = time_series_list.reshape(time_series_list.shape[0], -1)
time_series_list_2d = scaler.fit_transform(time_series_list_2d)
time_series_list = time_series_list_2d.reshape(time_series_list.shape)
time_series_list = torch.tensor(time_series_list, dtype=torch.float32)  # <-- Add this line

# Dataset Class
class GRBDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
    

'''
LSTM Autoencoder

I read up on LSTMs and looked at actual code of LSTM implementation in the following places primarily: 
- PyTorch Documentation https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
- Geeks For Geeks Tutorial https://www.geeksforgeeks.org/deep-learning/long-short-term-memory-networks-using-pytorch/
- PyTorch Forums About Encoding/Decoding w/ an LSTM https://discuss.pytorch.org/t/encoder-decoder-lstm-model-for-time-series-forecasting/189892/3

I wrote the code generally systematically:
- I built out the LSTM framework from the examples I found online.
- I adapted them to GRB analysis from the Transformer encoder.py script written by Dr. Sravan.
- I then began the process of training, testing clustering, and adding complexity to the architecture
  such as attention and bidirectionality.
- I am now in the process of sweeping and adding more complexity.

Model Components:
- Attention:
    I added attention because, to my understanding, in each light curve the actual burst is far more useful to us than
    the baseline from the noise. Attention lets the model weigh the parts of each light curve differently, and thus can add
    more weight to the bursts rather than the noise. You can see it implemented in the Encoder Class.
    Can be implemented better. The attention mechanism is currently a singular linear layer rather than a multi-head or some other
    more complex attention component.
- Bidirectionality:
    This was much easier to implement. Rather than adding attention layers, we can simply enable bidirectionality in the
    PyTorch LSTM arguments. So convenient! This makes the model process each light curve forwards and backwards and, I think that
    will lead to better reconstruction and clustering.

Notes:
- There is no intermediate network.
- Dropout has yet to be tuned.
- Could use a more complex decoder.
- Could use a more robust attention mechanism.
- Could use a more explicit regularization method than dropout alone.
- I have not tried a Time-Aware LSTM yet (will require a new preprocessing script).
- 
'''

# Bidirectional LSTM Autoencoder Model w/ attention
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, latent_size, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.fc_latent = nn.Linear(hidden_size * 2, latent_size)  # compress to latent

    def forward(self, x):
        out, _ = self.lstm(x)  # out: [batch, time, hidden_size*2]

        attn_scores = self.attention(out)              # [batch, time, 1]
        attn_weights = torch.softmax(attn_scores, 1)   # normalize over time
        context = torch.sum(attn_weights * out, dim=1) # [batch, hidden_size*2]

        latent = self.fc_latent(context)               # [batch, latent_size]
        return latent, attn_weights                    # return latent features (what we're trying to extract) + attention weights


class Decoder(nn.Module):
    def __init__(self, latent_size, hidden_size, num_layers, output_size, seq_len):
        super().__init__()
        self.fc_expand = nn.Linear(latent_size, hidden_size * 2)
        self.lstm = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc_out = nn.Linear(hidden_size * 2, output_size)
        self.seq_len = seq_len

    def forward(self, latent):
        # Expand latent vector to all timesteps
        repeated = self.fc_expand(latent).unsqueeze(1).repeat(1, self.seq_len, 1)
        
        output, _ = self.lstm(repeated)     # [batch, time, hidden_size*2]
        output = self.fc_out(output)        # [batch, time, output_size]
        return output


class BiLSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, latent_size, seq_len, dropout):
        super().__init__()
        self.encoder = Encoder(input_size, hidden_size, num_layers, latent_size, dropout)
        self.decoder = Decoder(latent_size, hidden_size, num_layers, input_size, seq_len)

    def forward(self, x):
        latent, attn_weights = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent, attn_weights
    

# Standard Training and Implementation

# Parameters
input_dim       = 14       # Number of detectors (features per timestep)
hidden_dim      = 16       # LSTM hidden state size
latent_dim      = 64       # Size of latent representation (embedding)
num_layers      = 2        # Number of LSTM layers
dropout         = 0.4      # Dropout between LSTM layers
batch_size      = 16       # Number of GRBs per batch
num_epochs      = 20       # Training epochs
learning_rate   = 0.00012  # Optimizer learning rate
sequence_length = np.shape(time_series_list)[1]  # Timesteps per GRB


model = BiLSTMAutoencoder(
    input_dim,
    hidden_dim,
    num_layers,
    latent_dim,
    sequence_length,
    dropout
)

# Define the loss function and optimizer and scheduler
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

# Get data
dataset = GRBDataset(time_series_list)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    for i, batch in enumerate(dataloader):
        batch = batch.float()
        optimizer.zero_grad()
        reconstructed, _, _ = model(batch)
        loss = criterion(reconstructed, batch)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)


        # Print the final batch loss for this epoch
        if i == len(dataloader) - 1:
            print(f"Epoch {epoch+1}, Final batch loss: {loss.item():.4f}")
    
# Extract latent features / Inference
model.eval()
latent_feats = []
with torch.no_grad():
    for batch in dataloader:
        _, latent, _ = model(batch)
        latent_feats.append(latent.numpy())

latent_feats = np.concatenate(latent_feats, axis=0)
np.save("latent_feats.npy", latent_feats)
np.save("burst_list.npy", np.array(burst_ids))