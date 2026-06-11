from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

class ModelTrainer:

    def __init__(self, data):
        self.data = data

    def train(self):
        X = self.data.drop('stroke', axis=1)
        y = self.data['stroke']

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)

        print(f"Model Accuracy: {accuracy:.2f}")

        return model