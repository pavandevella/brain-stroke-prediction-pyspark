import pandas as pd

class DataLoader:
    """Load healthcare stroke data"""

    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None

    def load_data(self):
        self.data = pd.read_csv(self.filepath)
        print(f"Data loaded: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
        return self.data

    def get_data(self):
        return self.data