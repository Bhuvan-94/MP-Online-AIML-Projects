
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SVDModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=20):
        super(SVDModel, self).__init__()
        self.user_emb = nn.Embedding(num_users+1, embedding_dim)
        self.item_emb = nn.Embedding(num_items+1, embedding_dim)
        self.user_bias = nn.Embedding(num_users+1, 1)
        self.item_bias = nn.Embedding(num_items+1, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

    def forward(self, user, item):
        pred = self.global_bias + self.user_bias(user).squeeze() + self.item_bias(item).squeeze() + \
               (self.user_emb(user) * self.item_emb(item)).sum(1)
        return pred

class CollaborativeEngine:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None

    def train(self, train_data, num_users, num_items, epochs=10):
        self.model = SVDModel(num_users, num_items)
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        users = torch.tensor(train_data['user_id'].values, dtype=torch.long)
        items = torch.tensor(train_data['movie_id'].values, dtype=torch.long)
        ratings = torch.tensor(train_data['rating'].values, dtype=torch.float)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            preds = self.model(users, items)
            loss = criterion(preds, ratings)
            loss.backward()
            optimizer.step()
        
        torch.save(self.model.state_dict(), self.model_path)
