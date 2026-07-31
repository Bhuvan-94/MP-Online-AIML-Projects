
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data(interactions, test_size=0.2):
    train, test = train_test_split(interactions, test_size=test_size, random_state=42)
    return train, test
