# storage.py

import pandas as pd
import os

class CSVDataStorage:
    def __init__(self, data_dir="data"):
        self.data_file = os.path.join(data_dir, "traffic.csv")
        self.city_coordinates = {
            'New York': {'latitude': 40.7128, 'longitude': -74.0060},
            'London': {'latitude': 51.5074, 'longitude': -0.1278},
            'Tokyo': {'latitude': 35.6762, 'longitude': 139.6503}
        }

    def load_all_data(self):
        """Load and preprocess the traffic CSV data"""
        try:
            if not os.path.exists(self.data_file):
                raise FileNotFoundError(f"CSV file not found: {self.data_file}")
            
            df = pd.read_csv(self.data_file)

            # Required columns check
            required_cols = [
                'timestamp', 'City', 'free_flow_speed', 'current_speed',
                'temperature', 'humidity'
            ]
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")

            # Fill defaults for optional columns
            defaults = {
                'precipitation': 0,
                'weather_condition': 'normal',
                'Severity_x': 1,
                'Severity_y': 1,
                'Description': 'no description',
                'distance': 0,
                'Latitude': 0,
                'Longitude': 0
            }

            for col, default in defaults.items():
                if col not in df.columns:
                    df[col] = default

            # Timestamp conversion
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df.dropna(subset=['timestamp'], inplace=True)

            # Feature engineering
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

            # Congestion calculation
            df['congestion_level'] = (1 - (df['current_speed'] / df['free_flow_speed'].replace(0, 1))) * 100
            df['congestion_level'] = df['congestion_level'].clip(0, 100)

            # Final fill for missing values
            fill_values = {
                'temperature': 20,
                'humidity': 50,
                'precipitation': 0
            }
            for col, val in fill_values.items():
                df[col] = df[col].fillna(val)

            df.rename(columns={"City": "city"}, inplace=True)
            print("✅ Data loaded and preprocessed successfully")
            return df.dropna()

        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return None

    def get_city_coordinates(self, city_name):
        return self.city_coordinates.get(city_name, {"latitude": 0, "longitude": 0})
