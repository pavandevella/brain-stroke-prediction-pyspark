from src.data_loader import DataLoader
from src.data_preprocessor import DataPreprocessor
from src.model_trainer import ModelTrainer

def main():
    print("=== Brain Stroke Prediction Project ===")

    loader = DataLoader("data/stroke_data.csv")
    data = loader.load_data()

    preprocessor = DataPreprocessor(data)
    processed_data = preprocessor.preprocess()

    trainer = ModelTrainer(processed_data)
    trainer.train()

    print("Project completed successfully!")

if __name__ == "__main__":
    main()