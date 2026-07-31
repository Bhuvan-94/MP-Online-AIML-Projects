
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_model(model, test_data):
    model.eval()
    users = torch.tensor(test_data['user_id'].values, dtype=torch.long)
    items = torch.tensor(test_data['movie_id'].values, dtype=torch.long)
    actuals = test_data['rating'].values
    with torch.no_grad():
        preds = model(users, items).numpy()
    
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    mae = mean_absolute_error(actuals, preds)
    return rmse, mae
