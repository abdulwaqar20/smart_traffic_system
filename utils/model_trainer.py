# trainer.py
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from typing import Optional, Dict, Union
from datetime import datetime
import joblib
import pandas as pd
import os
import numpy as np

class TrafficModelTrainer:
    def __init__(self, model_path="models/traffic_model.pkl", save_interval=10):
        self.model_path = model_path
        self.save_interval = save_interval
        self.model = None
        self.features = None
        self.metrics = None
        self.last_trained = None
        self.label_encoders = {}  # Store encoders per column
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

    def train_model(self, data: pd.DataFrame, test_size=0.2, n_estimators=100, random_state=42, verbose=True) -> Optional[Dict]:
        try:
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Input must be a DataFrame")
            if 'congestion_level' not in data.columns:
                raise ValueError("Missing target column: 'congestion_level'")

            # Drop unused columns safely
            X = data.drop(columns=['congestion_level', 'timestamp', 'city', 'description'], errors='ignore')
            y = data['congestion_level']

            if X.empty:
                raise ValueError("No usable features found")

            # Encode categorical columns
            for col in X.select_dtypes(include=['object', 'category']).columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))  # Convert to string first to avoid issues
                self.label_encoders[col] = le

            self.features = list(X.columns)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=random_state,
                max_depth=10,
                min_samples_split=5,
                n_jobs=-1
            )
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)

            self.metrics = {
                'MAE': round(mean_absolute_error(y_test, y_pred), 2),
                'R2': round(r2_score(y_test, y_pred), 2),
                'RMSE': round(np.sqrt(mean_squared_error(y_test, y_pred)), 2)
            }

            self.last_trained = datetime.now()
            self._save_model(verbose)

            if verbose:
                print(f"✅ Model trained at {self.last_trained}")
                print(f"📊 Metrics: {self.metrics}")
                print(f"🔍 Features: {self.features}")

            return {
                'model': self.model,
                'metrics': self.metrics,
                'features': self.features,
                'last_trained': self.last_trained
            }

        except Exception as e:
            print(f"❌ Training error: {str(e)}")
            return None

    def _save_model(self, verbose=True):
        try:
            joblib.dump({
                'model': self.model,
                'features': self.features,
                'metrics': self.metrics,
                'last_trained': self.last_trained,
                'label_encoders': self.label_encoders  # Save encoders
            }, self.model_path)
            if verbose:
                print(f"💾 Model saved at {self.model_path}")
            return True
        except Exception as e:
            print(f"❌ Failed to save model: {str(e)}")
            return False

    def load_model(self) -> Optional[RandomForestRegressor]:
        try:
            if not os.path.exists(self.model_path):
                print("⚠️ Model file not found")
                return None
            data = joblib.load(self.model_path)
            if isinstance(data, dict):
                self.model = data.get('model')
                self.features = data.get('features')
                self.metrics = data.get('metrics')
                self.last_trained = data.get('last_trained')
                self.label_encoders = data.get('label_encoders', {})
                print(f"✅ Model loaded (trained on {self.last_trained})")
                return self.model
            else:
                self.model = data
                return self.model
        except Exception as e:
            print(f"❌ Load error: {str(e)}")
            return None

    def predict(self, input_data: Union[pd.DataFrame, dict], verbose=True) -> Optional[Union[float, list]]:
        try:
            if self.model is None:
                if not self.load_model():
                    raise ValueError("Model not loaded")

            if isinstance(input_data, dict):
                input_data = pd.DataFrame([input_data])

            # Apply same label encoding on input_data categorical features
            for col, le in self.label_encoders.items():
                if col in input_data.columns:
                    # Convert to string first to avoid issues
                    input_data[col] = input_data[col].astype(str)
                    input_data[col] = le.transform(input_data[col])
                else:
                    raise ValueError(f"Missing feature for prediction: {col}")

            missing = [f for f in self.features if f not in input_data.columns]
            if missing:
                raise ValueError(f"Missing features: {missing}")

            X = input_data[self.features]
            predictions = self.model.predict(X)

            if verbose and len(predictions) == 1:
                print(f"🔮 Predicted: {predictions[0]:.2f}%")

            return predictions[0] if len(predictions) == 1 else predictions

        except Exception as e:
            print(f"❌ Prediction error: {str(e)}")
            return None
