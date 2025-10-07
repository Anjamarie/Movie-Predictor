import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError, EntryNotFoundError

# -----------------------------------------------
# 1. CRITICAL CONFIGURATION: CHECK CASE SENSITIVITY
# -----------------------------------------------
# NOTE: Casing here MUST match Hugging Face EXACTLY!
HF_REPO_ID = "Anjamarie/Movie-Predictor" 
MODEL_REVENUE_FILE = "movie_revenue_model.pkl" 
MODEL_FEATURES_FILE = "model_features.pkl" 
# -----------------------------------------------

@st.cache_resource
def download_and_load_model_hf(file_name, repo_id):
    """
    Uses huggingface_hub to download the file into the local cache and loads it.
    """
    st.info(f"Attempting to load '{file_name}' from repository: {repo_id}")
    
    try:
        # Download the file. If successful, this returns the path in the local cache.
        cache_path = hf_hub_download(repo_id=repo_id, filename=file_name, revision="main")
        
        st.success(f"'{file_name}' downloaded successfully. Loading model...")
        
        # Load the model using joblib from the cached path
        loaded_object = joblib.load(cache_path)
        return loaded_object
        
    except RepositoryNotFoundError:
        st.error(f"FATAL ERROR: Repository '{repo_id}' not found. Please verify the casing of your username and repository name on Hugging Face.")
    except EntryNotFoundError:
        st.error(f"FATAL ERROR: File '{file_name}' not found in repo '{repo_id}'. Please verify the file name casing.")
    except Exception as e:
        # Catch other errors, like network failure or corrupted file
        st.error(f"FATAL ERROR: Unknown error during download/load of '{file_name}'. Reason: {type(e).__name__}: {e}")
        st.warning("Ensure the Hugging Face repo is set to **Public**.")
    
    return None # Return None if any error occurred


# -----------------------------------------------
# 2. Load the Saved Model and Features 
# -----------------------------------------------

# Load the features file first
model_features = download_and_load_model_hf(MODEL_FEATURES_FILE, HF_REPO_ID)

# Load the trained model
model = download_and_load_model_hf(MODEL_REVENUE_FILE, HF_REPO_ID)


# --- 3. Run App Logic Only if Both Models Loaded Successfully ---

if model is None or model_features is None:
    st.stop() 

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
if st.sidebar.button('Predict Revenue'):
    prediction = model.predict(input_df)
    st.subheader('Predicted Revenue:')
    st.success(f'${prediction[0]:,.2f}')



