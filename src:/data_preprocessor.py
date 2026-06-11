from sklearn.preprocessing import StandardScaler, LabelEncoder

class DataPreprocessor:

    def __init__(self, data):
        self.data = data.copy()
        self.scaler = StandardScaler()

    def preprocess(self):
        if 'gender' in self.data.columns:
            le = LabelEncoder()
            self.data['gender'] = le.fit_transform(self.data['gender'])

        numeric_cols = ['age', 'avg_glucose_level', 'bmi']

        for col in numeric_cols:
            if col in self.data.columns:
                self.data[col] = self.scaler.fit_transform(
                    self.data[[col]]
                )

        return self.data