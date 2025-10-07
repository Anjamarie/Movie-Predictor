import streamlit as st
import pandas as pd
import joblib
import requests
import os
from huggingface_hub.utils import RepositoryNotFoundError, EntryNotFoundError, HfHubDownloadError

# -----------------------------------------------
# 1. CRITICAL CONFIGURATION: Using standard Hugging Face resolve URL 
# -----------------------------------------------
# NOTE: The lowercase repo ID 'anjamarie/movie-predictor' is used for best compatibility.
HF_REPO_ID = "anjamarie/movie-predictor"
MODEL_REVENUE_FILE = "movie_revenue_model.pkl" 
MODEL_FEATURES_FILE = "model_features.pkl" 

# Defining the standard resolve URLs
BASE_HF_URL = "https://huggingface.co"
MODEL_FEATURES_CDN_URL = f"{BASE_HF_URL}/{HF_REPO_ID}/resolve/main/{MODEL_FEATURES_FILE}"
MODEL_REVENUE_CDN_URL = f"{BASE_HF_URL}/{HF_REPO_ID}/resolve/main/{MODEL_REVENUE_FILE}"

# Define local paths (used by joblib)
MODEL_FEATURES_PATH = "model_features.pkl"
MODEL_REVENUE_PATH = "movie_revenue_model.pkl"
# -----------------------------------------------

@st.cache_resource
def download_and_load_model_hf(file_name, cdn_url, local_path):
    """
    Uses the direct URL to download the file using requests with a timeout, 
    which is essential for diagnosing persistent network/proxy blocks.
    """
    st.info(f"Attempting to download '{file_name}' from: {cdn_url}")
    
    if os.path.exists(local_path):
        st.success(f"'{local_path}' found in cache.")
    else:
        try:
            with st.spinner(f'Downloading large file: {file_name}...'):
                # Added timeout=10 to prevent logs from stalling on network hang
                response = requests.get(cdn_url, stream=True, timeout=10) 
                response.raise_for_status() # Raise exception for 4xx or 5xx errors

                with open(local_path, 'wb') as file:
                    file.write(response.content)
                
                st.success(f"'{file_name}' download complete via standard link!")
            
        except requests.exceptions.HTTPError as e:
            # Captures 403 Forbidden or 404 Not Found errors clearly
            st.error(f"FATAL ERROR (HTTP {e.response.status_code}): Download failed.")
            st.error(f"Action: Check Hugging Face repo '{HF_REPO_ID}' file casing and Public status.")
            st.warning(f"Error details: {e}")
            return None 
            
        except requests.exceptions.Timeout:
            # Captures network stalls (proxy/firewall block)
            st.error("FATAL ERROR (Network Timeout): Download stalled for 10 seconds.")
            st.warning("Action: This indicates a strict network block on the Streamlit server.")
            return None

        except Exception as e:
            # Catches any other error during download
            st.error(f"FATAL ERROR (Unknown Download Error): {type(e).__name__}: {e}")
            return None

    # Load the object using joblib
    try:
        loaded_object = joblib.load(local_path)
        st.success(f"'{file_name}' loaded successfully.")
        return loaded_object
    except Exception as e:
        # Catches corruption/bad file format after download
        st.error(f"FATAL ERROR (Loading/Corrupted): Failed to load '{local_path}'. Is the file corrupted?")
        st.error(f"Reason: {type(e).__name__}: {e}")
        return None 


# -----------------------------------------------
# 2. Load the Saved Model and Features 
# -----------------------------------------------

# Load the features file first
model_features = download_and_load_model_hf(MODEL_FEATURES_FILE, MODEL_FEATURES_CDN_URL, MODEL_FEATURES_PATH)

# Load the trained model
model = download_and_load_model_hf(MODEL_REVENUE_FILE, MODEL_REVENUE_CDN_URL, MODEL_REVENUE_PATH)


# --- 3. Run App Logic Only if Both Models Loaded Successfully ---

if model is None or model_features is None:
    st.stop() 

# --- 4. Build the User Interface (using robust feature list extraction) ---
st.title('🎬 Movie Revenue Predictor')
st.write("Enter the movie's details to predict its potential revenue.")

# Get feature list: handles DataFrame columns, Series index, or standard list object
if hasattr(model_features, 'columns'):
    all_features = list(model_features.columns)
elif hasattr(model_features, 'index'):
    all_features = list(model_features.index)
elif isinstance(model_features, (list, set, tuple)):
    all_features = list(model_features)
else:
    # If all else fails, assume it's an iterable list of features
    try:
        all_features = list(model_features)
    except:
        st.error("FATAL UI ERROR: Could not extract features from the loaded 'model_features.pkl'.")
        st.stop()
    
# Proceed with feature extraction
genre_cols = [col for col in all_features if col.startswith('genre_')]
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
# Ensure input_df creation uses the available features list
input_df = pd.DataFrame(0, index=[0], columns=all_features)

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
if st.sidebar.button('Predict Revenue'):
    # Ensure the model variable exists before predicting
    if model is not None:
        prediction = model.predict(input_df)
        st.subheader('Predicted Revenue:')
        st.success(f'${prediction[0]:,.2f}')
    else:
        st.warning("Model not available for prediction.")
