# train_model.py
from utils.storage import CSVDataStorage
from utils.model_trainer import TrafficModelTrainer

# Step 1: Load merged and cleaned data
data_storage = CSVDataStorage(data_dir="data")
data = data_storage.load_data()


# Step 2: Train the model and save it as .pkl
if data is not None:
    trainer = TrafficModelTrainer(model_path="models/traffic_model.pkl")
    result = trainer.train_model(data)

    if result:
        print("✅ Model trained and saved successfully.")
        print("Metrics:", result['metrics'])
    else:
        print("❌ Model training failed.")
else:
    print("❌ Data loading failed.")
