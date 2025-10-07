import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download
# Import the specific error classes for clearer debugging
from huggingface_hub.utils import RepositoryNotFoundError, EntryNotFoundError, HfHubDownloadError

# -----------------------------------------------
# 1. CRITICAL CONFIGURATION: Lowercase for Reliability
# -----------------------------------------------
# The repository ID is converted to lowercase here to bypass case-sensitivity issues,
# which are often the cause of "Permission Denied" errors when the repo is public.
HF_REPO_ID = "Anjamarie/Movie-Predictor"  # Note: Converted to all lowercase
MODEL_REVENUE_FILE = "movie_revenue_model.pkl" 
MODEL_FEATURES_FILE = "model_features.pkl" 
# -----------------------------------------------

@st.cache_resource
def download_and_load_model_hf(file_name, repo_id):
    """
    Uses huggingface_hub to download the file into the local cache and loads it.
    Includes robust error logging for specific Hugging Face failure modes.
    """
    st.info(f"Attempting to load '{file_name}' from repository: {repo_id}")
    
    try:
        # Download the file. This handles caching and redirects internally.
        cache_path = hf_hub_download(repo_id=repo_id, filename=file_name, revision="main")
        
        st.success(f"'{file_name}' downloaded successfully. Loading model...")
        
        # Load the model using joblib from the cached path
        loaded_object = joblib.load(cache_path)
        return loaded_object
        
    except RepositoryNotFoundError:
        st.error(f"FATAL ERROR (404/Repo Not Found): Repository '{repo_id}' does not exist.")
        st.error("Action: Verify the casing of your username and repository name on Hugging Face.")
    
    except EntryNotFoundError:
        st.error(f"FATAL ERROR (404/File Not Found): File '{file_name}' is missing in repo '{repo_id}'.")
        st.error("Action: Verify the file name casing and ensure the file was uploaded successfully.")
        
    except HfHubDownloadError as e:
        # Catch network-level errors, including 403 Forbidden
        st.error(f"FATAL ERROR (Download Failed): Network error during file transfer.")
        st.error(f"Hugging Face Hub Error: {e}")
        st.warning("Action: Ensure the Hugging Face repo is set to **Public** (not gated or private).")

    except Exception as e:
        # Catch other errors, like corrupted file or joblib failure
        st.error(f"FATAL ERROR (Loading/Other): An unknown error occurred while loading '{file_name}'.")
        st.error(f"Reason: {type(e).__name__}: {e}")
    
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
# We assume model_features is a list of column names or a similar object with a .columns attribute
if hasattr(model_features, 'columns'):
    all_features = model_features.columns
else:
    # If it's just a list/set of strings, treat it as a list
    all_features = model_features 
    
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
    prediction = model.predict(input_df)
    st.subheader('Predicted Revenue:')
    st.success(f'${prediction[0]:,.2f}')



