import streamlit as st
import pickle
import requests
import os
import pandas as pd
import joblib

# -----------------------------------------------
# 1. MODEL CONFIGURATION & DOWNLOAD SETUP
# -----------------------------------------------

MODEL_REVENUE_URL = "https://drive.google.com/uc?export=1b5-h4W1vVuw5Byu7Umz58zQYYvDasBFr" 

# URL for the features file (already provided by the user)
MODEL_FEATURES_URL = "https://drive.google.com/uc?export=download&id=1CrTPIKPj6jXEOSktfvh8bAyJj1SaHG8m" 

# Define the local file names for saving
MODEL_FEATURES_PATH = "model_features.pkl"
MODEL_REVENUE_PATH = "movie_revenue_model.pkl"

@st.cache_resource
def download_and_load_model(download_url, local_path):
    """
    Checks if the model file exists locally. If not, downloads it from the
    provided Google Drive URL and then loads it using joblib.
    """

    # Check if the file already exists locally (useful for Streamlit Cloud caching)
    if not os.path.exists(local_path):
        st.info(f"Model file '{local_path}' not found. Attempting download...")
        
        try:
            with st.spinner(f'Downloading large file: {local_path}...'):
                # Use requests to perform the file download
                response = requests.get(download_url, stream=True)
                response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

                # Write the downloaded content to the local file path
                with open(local_path, 'wb') as file:
                    file.write(response.content)
            
            st.success(f"'{local_path}' download complete!")
            
        except requests.exceptions.RequestException as e:
            st.error(f"FATAL ERROR: Failed to download model file '{local_path}'. Check the Google Drive URL and sharing permissions. Error: {e}")
            return None # Return None if download fails

    # Load the model using joblib
    try:
        loaded_object = joblib.load(local_path)
        return loaded_object
    except Exception as e:
        st.error(f"FATAL ERROR: Failed to load '{local_path}' using joblib. Is the file corrupted? Error: {e}")
        return None


# -----------------------------------------------
# 2. Load the Saved Model and Features (Using the new function)
# -----------------------------------------------

# Load the features file first
model_features = download_and_load_model(MODEL_FEATURES_URL, MODEL_FEATURES_PATH)

# Load the trained model
model = download_and_load_model(MODEL_REVENUE_URL, MODEL_REVENUE_PATH)


# --- 3. Run App Logic Only if Both Models Loaded Successfully ---

if model is None or model_features is None:
    st.stop() # Stop the app if model loading failed (error message already displayed)

# --- 4. Build the User Interface ---
st.title('🎬 Movie Revenue Predictor')
st.write("Enter the movie's details to predict its potential revenue.")

# Identify the genre columns the model was trained on
genre_cols = [col for col in model_features.columns if col.startswith('genre_')]
display_genres = sorted([col.replace('genre_', '') for col in genre_cols])

with st.sidebar:
    st.header("Movie Features")
    budget = st.number_input('Budget (in USD)', min_value=10000, max_value=400000000, value=50000000, step=1000000)
    runtime = st.number_input('Runtime (in minutes)', min_value=60, max_value=240, value=120)
    selected_genres = st.multiselect('Select Genres', options=display_genres)
    release_year = st.number_input('Release Year', min_value=1980, max_value=2025, value=2023)
    release_month = st.slider('Release Month', 1, 12, 6)
    release_dayofweek = st.slider('Release Day of Week (0=Monday, 6=Sunday)', 0, 6, 4)

# --- 5. Prepare Input for Prediction ---
# Create a DataFrame with all feature columns initialized to 0
# The model_features object loaded from the PKL file is assumed to be a DataFrame
input_df = pd.DataFrame(0, index=[0], columns=model_features.columns)

# Update the DataFrame with the user's direct inputs
input_df['budget'] = budget
input_df['runtime'] = runtime
input_df['release_year'] = release_year
input_df['release_month'] = release_month
input_df['release_dayofweek'] = release_dayofweek

# Set the selected genre columns to 1
for genre in selected_genres:
    genre_column_name = f'genre_{genre}'
    if genre_column_name in input_df.columns:
        input_df[genre_column_name] = 1

# --- 6. Make a Prediction ---
# The prediction happens inside a button click to avoid re-running on every widget change
if st.sidebar.button('Predict Revenue'):
    prediction = model.predict(input_df)
    st.subheader('Predicted Revenue:')
    st.success(f'${prediction[0]:,.2f}')


