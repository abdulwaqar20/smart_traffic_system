from seaborn import heatmap
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import folium_static
from utils.storage import CSVDataStorage
from utils.model_trainer import TrafficModelTrainer
from datetime import datetime, timedelta
import numpy as np
import os
import requests
from typing import Dict, List
from folium.plugins import MarkerCluster, HeatMap

# Initialize services
data_storage = CSVDataStorage(data_dir="data")
model_trainer = TrafficModelTrainer(model_path="models/traffic_model.pkl")

# TomTom Configuration
TOMTOM_API_KEY = "AdjhtQTjFRnPR3Ld7PTOlG0HaWk21Vox"
TOMTOM_BASE_URL = "https://api.tomtom.com"
TOMTOM_MAP_STYLE = "main"  # Options: "main", "night", "satellite"

# Streamlit UI Configuration
st.set_page_config(
    layout="wide", 
    page_title="Advanced Traffic Prediction System",
    page_icon="🚦"
)

# Custom CSS for advanced styling
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .stAlert {
        border-radius: 10px;
    }
    .st-b7 {
        color: white;
    }
    .stTab {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 8px 16px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stSelectbox, .stSlider, .stNumberInput {
        background-color: white;
        border-radius: 5px;
    }
    .map-container {
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .route-info {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin-top: 10px;
    }
    .incident-marker {
        font-size: 12px;
        color: white;
        background: red;
        border-radius: 50%;
        width: 20px;
        height: 20px;
        text-align: center;
        line-height: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Helper Functions
@st.cache_data(ttl=3600)
def load_all_data():
    return data_storage.load_all_data()

def create_tomtom_map(center, zoom=12):
    """Create a Folium map with TomTom tiles"""
    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles=None,
        control_scale=True
    )
    
    # Add TomTom base map
    folium.TileLayer(
        tiles=f'https://{{s}}.api.tomtom.com/map/1/tile/{TOMTOM_MAP_STYLE}/{{z}}/{{x}}/{{y}}.png?key={TOMTOM_API_KEY}',
        attr='TomTom',
        name='TomTom Base Map',
        max_native_zoom=22,
        max_zoom=22
    ).add_to(m)
    
    # Add traffic flow layer
    folium.TileLayer(
        tiles=f'https://{{s}}.api.tomtom.com/map/1/tile/flow/{{z}}/{{x}}/{{y}}.png?key={TOMTOM_API_KEY}',
        attr='TomTom Traffic Flow',
        name='Traffic Flow',
        overlay=True
    ).add_to(m)
    
    return m

def get_tomtom_route(start_coords, end_coords):
    """Get route coordinates using TomTom API"""
    try:
        url = f"{TOMTOM_BASE_URL}/routing/1/calculateRoute/{start_coords[1]},{start_coords[0]}:{end_coords[1]},{end_coords[0]}/json"
        params = {
            'key': TOMTOM_API_KEY,
            'travelMode': 'car',
            'traffic': 'true',
            'routeType': 'fastest',
            'considerTraffic': 'true'
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if data.get('routes'):
            route = []
            # Extract coordinates from the route
            for leg in data['routes'][0]['legs']:
                for point in leg['points']:
                    route.append([point['latitude'], point['longitude']])
            
            summary = data['routes'][0]['summary']
            distance = summary['lengthInMeters'] / 1000  # km
            duration = summary['travelTimeInSeconds'] / 60  # minutes
            traffic_delay = summary['trafficDelayInSeconds'] / 60  # minutes
            traffic_length = summary['trafficLengthInMeters'] / 1000  # km
            
            return route, distance, duration, traffic_delay, traffic_length
    except Exception as e:
        st.error(f"Route calculation failed: {str(e)}")
    return None, None, None, None, None

def get_tomtom_incidents(bounds):
    """Get traffic incidents from TomTom within given bounds"""
    try:
        url = f"{TOMTOM_BASE_URL}/traffic/services/5/incidentDetails"
        params = {
            'key': TOMTOM_API_KEY,
            'bbox': f"{bounds['west']},{bounds['south']},{bounds['east']},{bounds['north']}",
            'fields': '{incidents{type,geometry{type,coordinates},properties{iconCategory,startTime,severity}}}',
            'language': 'en-US',
            'timeValidityFilter': 'present'
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        return data.get('incidents', [])
    except Exception as e:
        st.error(f"Failed to get incidents: {str(e)}")
        return []

class InputValidator:
    @staticmethod
    def validate_traffic_inputs(input_data: Dict) -> List[str]:
        errors = []
        if not 0 <= input_data['hour'] <= 23:
            errors.append("Hour must be between 0-23")
        if not -20 <= input_data['temperature'] <= 50:
            errors.append("Temperature must be between -20°C and 50°C")
        if not 0 <= input_data['humidity'] <= 100:
            errors.append("Humidity must be between 0-100%")
        if input_data['precipitation'] < 0:
            errors.append("Precipitation cannot be negative")
        if input_data['incident_count'] < 0:
            errors.append("Incident count cannot be negative")
        return errors

# Main App
st.title("🚦 Traffic Congestion Prediction System")

# Live Route Planning Section
with st.container():
    st.header("📍 Live Route Planning")
    col1, col2 = st.columns(2)
    
    with col1:
        cities = list(data_storage.city_coordinates.keys())
        start_city = st.selectbox("Start City", cities, index=0)
        
    with col2:
        end_city = st.selectbox("Destination City", cities, index=1 if len(cities) > 1 else 0)
    
    if st.button("Show Best Route"):
        with st.spinner("Calculating best route..."):
            # Get coordinates
            start_coords = data_storage.get_city_coordinates(start_city)
            end_coords = data_storage.get_city_coordinates(end_city)
            
            # Create base map
            m = folium.Map(
                location=[start_coords['latitude'], start_coords['longitude']],
                zoom_start=12,
                tiles='OpenStreetMap'
            )
            
            # Get route (simplified version)
            try:
                route = [
                    [start_coords['latitude'], start_coords['longitude']],
                    [end_coords['latitude'], end_coords['longitude']]
                ]
                
                # Draw route line
                folium.PolyLine(
                    route,
                    color='blue',
                    weight=5,
                    opacity=0.7
                ).add_to(m)
                
                # Add markers
                folium.Marker(
                    [start_coords['latitude'], start_coords['longitude']],
                    popup=f"Start: {start_city}",
                    icon=folium.Icon(color='green')
                ).add_to(m)
                
                folium.Marker(
                    [end_coords['latitude'], end_coords['longitude']],
                    popup=f"Destination: {end_city}",
                    icon=folium.Icon(color='red')
                ).add_to(m)
                
                # Display route info
                st.success("Route found!")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Straight-line Distance", 
                            f"{np.sqrt((start_coords['latitude']-end_coords['latitude'])**2 + (start_coords['longitude']-end_coords['longitude'])**2)*111:.1f} km")
                
                # Show map
                st.subheader("Best Route")
                folium_static(m, height=500)
                
            except Exception as e:
                st.error(f"Error showing route: {str(e)}")
# Model Training Sidebar
with st.sidebar:
    st.header("Model Configuration")
    
    # Model version selection
    try:
        model_files = [f for f in os.listdir("models") if f.startswith("traffic_model_v")]
        model_files.sort(reverse=True)
        
        selected_model = st.selectbox(
            "Select Model Version",
            model_files,
            index=0 if model_files else None
        )
        
        if st.button("Load Selected Model"):
            model_trainer.model_path = os.path.join("models", selected_model)
            st.success(f"Loaded model: {selected_model}")
    except FileNotFoundError:
        st.warning("No models found in 'models' directory")
    
    # Training settings
    with st.expander("Training Settings", expanded=True):
        train_col1, train_col2 = st.columns(2)
        with train_col1:
            test_size = st.slider("Test Size (%)", 10, 40, 20)
        with train_col2:
            n_estimators = st.slider("Number of Trees", 50, 200, 100)
        
        if st.button("Train Model", help="Train a new prediction model"):
            with st.spinner("Training model..."):
                data = load_all_data()
                if data is not None:
                    model = model_trainer.train_model(
                        data, 
                        test_size=test_size/100, 
                        n_estimators=n_estimators
                    )
                    if model:
                        model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
                        model_path = f"models/traffic_model_v{model_version}.pkl"
                        model_trainer.save_model(model, model_path)
                        st.success(f"Model trained successfully! Version: {model_version}")
                        
                        with st.expander("Feature Importance"):
                            feat_imp = pd.DataFrame({
                                'Feature': model.feature_names_in_,
                                'Importance': model.feature_importances_
                            }).sort_values('Importance', ascending=False)
                            st.dataframe(feat_imp, hide_index=True)
                    else:
                        st.error("Model training failed")
                else:
                    st.error("Could not load data for training")

# Main Prediction Interface
st.header("🚦 Traffic Congestion Prediction")

# Load model or show warning
model = model_trainer.load_model()
if not model:
    st.warning("⚠️ No trained model available. Please train a model first.")
else:
    # Create prediction form with enhanced UI
    with st.form("prediction_form"):
        st.subheader("Traffic Prediction Parameters")
        cols = st.columns(3)
        
        with cols[0]:
            city = st.selectbox(
                "City", 
                list(data_storage.city_coordinates.keys()),
                help="Select the city for prediction"
            )
            hour = st.slider(
                "Hour of day", 
                0, 23, 17,
                help="Select the hour to predict congestion for (0=midnight, 23=11PM)"
            )
            day_of_week = st.selectbox(
                "Day of week", 
                ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                index=0,
                help="Select the day of week"
            )
            
        with cols[1]:
            temperature = st.number_input(
                "Temperature (°C)", 
                value=22.0, 
                min_value=-20.0, 
                max_value=50.0,
                help="Enter current temperature"
            )
            humidity = st.slider(
                "Humidity (%)", 
                0, 100, 65,
                help="Select current humidity level"
            )
            incidents = st.number_input(
                "Number of incidents", 
                0, 50, 2,
                help="Enter number of reported incidents"
            )
            
        with cols[2]:
            weather = st.selectbox(
                "Weather condition", 
                ["Clear", "Clouds", "Rain", "Snow", "Fog"],
                index=0,
                help="Select current weather condition"
            )
            precipitation = st.slider(
                "Precipitation (mm)", 
                0.0, 20.0, 0.0,
                help="Select precipitation amount"
            )
        
        submitted = st.form_submit_button(
            "Predict Congestion",
            type="primary",
            help="Generate congestion prediction based on inputs"
        )
        
        if submitted:
            # Prepare input data
            day_mapping = {
                "Monday": 0, "Tuesday": 1, "Wednesday": 2,
                "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
            }
            weather_mapping = {
                "Clear": 0, "Clouds": 1, "Rain": 2, "Snow": 3, "Fog": 4
            }
            
            input_data = {
                'hour': hour,
                'day_of_week': day_mapping[day_of_week],
                'is_weekend': 1 if day_mapping[day_of_week] in [5,6] else 0,
                'temperature': temperature,
                'weather_condition': weather_mapping[weather],
                'precipitation': precipitation,
                'humidity': humidity,
                'visibility': 10,  # Default value
                'incident_count': incidents
            }
            
            # Validate inputs
            validator = InputValidator()
            errors = validator.validate_traffic_inputs(input_data)
            
            if errors:
                for error in errors:
                    st.error(f"Validation Error: {error}")
            else:
                # Create dataframe with correct feature order
                input_df = pd.DataFrame([input_data], columns=model.feature_names_in_)
                
                # Make prediction
                try:
                    prediction = model.predict(input_df)[0]
                    prediction = np.clip(prediction, 0, 100)  # Ensure between 0-100%
                    
                    # Display results
                    st.success("Prediction Generated Successfully!")
                    
                    # Metrics columns
                    res_col1, res_col2, res_col3 = st.columns(3)
                    with res_col1:
                        st.metric(
                            "Predicted Congestion", 
                            f"{prediction:.1f}%",
                            help="Predicted congestion level (0-100%)"
                        )
                    
                    with res_col2:
                        congestion_level = "Low"
                        if prediction > 40: 
                            congestion_level = "High"
                            color = "red"
                        elif prediction > 20: 
                            congestion_level = "Moderate"
                            color = "orange"
                        else: 
                            congestion_level = "Low"
                            color = "green"
                        
                        st.metric(
                            "Congestion Level", 
                            congestion_level,
                            help="Interpretation of congestion level"
                        )
                    
                    with res_col3:
                        if prediction > 40:
                            action = "Avoid travel"
                            help_text = "Severe congestion expected"
                        elif prediction > 20:
                            action = "Expect delays"
                            help_text = "Moderate congestion expected"
                        else:
                            action = "Normal traffic"
                            help_text = "Minimal congestion expected"
                        
                        st.metric(
                            "Recommended Action", 
                            action,
                            help=help_text
                        )
                    
                    # Create gauge chart
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prediction,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Congestion Level", 'font': {'size': 24}},
                        gauge={
                            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                            'bar': {'color': color},
                            'steps': [
                                {'range': [0, 20], 'color': "lightgreen"},
                                {'range': [20, 40], 'color': "lightyellow"},
                                {'range': [40, 100], 'color': "lightcoral"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': prediction
                            }
                        }
                    ))
                    
                    fig.update_layout(height=300, margin=dict(l=50, r=50, b=50, t=50))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show prediction on TomTom map
                    city_coords = data_storage.get_city_coordinates(city)
                    m = create_tomtom_map([city_coords['latitude'], city_coords['longitude']])
                    
                    # Add prediction marker
                    folium.CircleMarker(
                        location=[city_coords['latitude'], city_coords['longitude']],
                        radius=10 + (prediction / 5),
                        color=color,
                        fill=True,
                        fill_color=color,
                        popup=f"""
                        <b>Predicted Congestion:</b> {prediction:.1f}%<br>
                        <b>Time:</b> {hour}:00 {day_of_week}<br>
                        <b>Weather:</b> {weather}<br>
                        <b>Incidents:</b> {incidents}
                        """,
                        tooltip=f"Predicted congestion: {prediction:.1f}%"
                    ).add_to(m)
                    
                    # Add layer control
                    folium.LayerControl().add_to(m)
                    
                    st.subheader("Predicted Traffic Conditions")
                    folium_static(m, width=800, height=500)
                    
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")

# Data Analysis Section
st.header("📊 Advanced Traffic Data Analysis")
data = load_all_data()

if data is not None:
    tab1, tab2, tab3, tab4 = st.tabs(["Trends", "Correlations", "Raw Data", "Map View"])
    
    with tab1:
        st.subheader("Historical Trends Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            city = st.selectbox(
                "Select City", 
                data['city'].unique(), 
                key='city_select'
            )
            metric = st.selectbox(
                "Select Metric", 
                ['congestion_level', 'current_speed', 'temperature', 'incident_count'],
                key='metric_select'
            )
        
        with col2:
            agg_level = st.selectbox(
                "Aggregation Level", 
                ["Raw", "Hourly", "Daily", "Weekly"],
                index=1,
                key='agg_level'
            )
        
        city_data = data[data['city'] == city].copy()
        
        if not city_data.empty:
            if agg_level != "Raw":
                agg_map = {
                    'Hourly': 'H',
                    'Daily': 'D',
                    'Weekly': 'W'
                }
                
                # Convert timestamp to datetime index if not already
                if not pd.api.types.is_datetime64_any_dtype(city_data['timestamp']):
                    city_data['timestamp'] = pd.to_datetime(city_data['timestamp'])
                
                # Select only numeric columns for aggregation
                numeric_cols = city_data.select_dtypes(include=[np.number]).columns.tolist()
                
                # Add timestamp back to numeric columns if it was removed
                if 'timestamp' not in numeric_cols:
                    numeric_cols.append('timestamp')
                
                # Perform resampling on numeric columns only
                city_data = city_data[numeric_cols].set_index('timestamp').resample(agg_map[agg_level]).mean().reset_index()
            
            fig = px.line(
                city_data, 
                x='timestamp', 
                y=metric,
                title=f"{metric.replace('_', ' ').title()} in {city} ({agg_level} View)",
                labels={'timestamp': 'Date', metric: metric.replace('_', ' ').title()},
                color_discrete_sequence=['#1f77b4']
            )
            
            fig.update_layout(
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="1d", step="day", stepmode="backward"),
                            dict(count=7, label="1w", step="day", stepmode="backward"),
                            dict(count=1, label="1m", step="month", stepmode="backward"),
                            dict(step="all")
                        ])
                    ),
                    rangeslider=dict(visible=True),
                    type="date"
                ),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data available for selected city")
    
    with tab2:
        st.subheader("Feature Correlations")
        numeric_cols = data.select_dtypes(include=['number']).columns
        corr_matrix = data[numeric_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu',
            zmin=-1,
            zmax=1,
            title="Correlation Matrix"
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Raw Data Explorer")
        st.info("Use the filters below to explore the raw traffic data")
        
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            filter_city = st.selectbox(
                "Filter City",
                ['All'] + list(data['city'].unique()),
                key='filter_city'
            )
        
        filtered_data = data.copy()
        if filter_city != 'All':
            filtered_data = filtered_data[filtered_data['city'] == filter_city]
        
        st.dataframe(
            filtered_data.sort_values('timestamp', ascending=False).head(1000),
            height=500,
            use_container_width=True
        )
    
    with tab4:
        st.subheader("Geospatial Traffic View")
        
        map_city = st.selectbox("Map City", data['city'].unique())
        show_heatmap = st.checkbox("Show Heatmap", True)
        
        city_coords = data_storage.get_city_coordinates(map_city)
        map_data = data[data['city'] == map_city].copy()
        
        if not map_data.empty:
            m = folium.Map(location=[city_coords['latitude'], city_coords['longitude']], zoom_start=12)
            
            if show_heatmap:
                heat_data = []
                for _, row in map_data.iterrows():
                    if pd.notna(row['latitude']) and pd.notna(row['longitude']):
                        heat_data.append([row['latitude'], row['longitude'], row['congestion_level']/100])
                
                HeatMap(heat_data).add_to(m)
            
            folium_static(m, width=800, height=500)
        else:
            st.warning("No data available for selected city")
else:
    st.warning("No data available for visualization. Please ensure data files are properly loaded.")

# Footer
st.markdown("---")
st.caption(f"""
    Advanced Traffic Prediction System | 
    Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | 
    Developed for Intelligent Transportation Solutions""")