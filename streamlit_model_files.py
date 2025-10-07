import tmdbsimple as tmdb
import pandas as pd
import time
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import ast
import joblib
import os
import requests

# This script is run ONCE locally to generate model files that perfectly
# match the required input features of the Streamlit app (app.py).

# ----------------------------------------------------
# 1. Configuration (MUST BE SECURE if running outside local environment)
# ----------------------------------------------------

# NOTE: API_KEY is exposed here. For production, use environment variables.
tmdb.API_KEY = '7eb2f50ca573c609c0bac8e9f804514d'

PAGES_TO_FETCH = 10  # Reduced pages for quick test generation

# Define the features that the Streamlit app actually collects from the user.
STREAMLIT_INPUT_FEATURES = [
    'budget',
    'runtime',
    'release_year',
    'release_month',
    'release_dayofweek',
]
# Genres will be added dynamically later

# ----------------------------------------------------
# 2. Data Acquisition Function (Copied from Notebook)
# ----------------------------------------------------

def get_movie_data(movie_id):
    """Fetches detailed data for a single movie from TMDb."""
    try:
        movie = tmdb.Movies(movie_id)
        info = movie.info()
        credits = movie.credits()
        keywords = movie.keywords()
        
        director = next((person['name'] for person in credits['crew'] if person['job'] == 'Director'), None)
        cast = [actor['name'] for actor in credits['cast'][:5]]
        
        # Filter for quality data before returning
        if info.get('budget', 0) == 0 or info.get('revenue', 0) == 0:
            return None
        return {
            'id': info['id'],
            'title': info['title'],
            'release_date': info.get('release_date'),
            'budget': info.get('budget'),
            'revenue': info.get('revenue'),
            'runtime': info.get('runtime'),
            'genres': [genre['name'] for genre in info.get('genres', [])],
            'cast': cast,
            'director': director,
            'keywords': [keyword['name'] for keyword in keywords.get('keywords', [])],
            'production_companies': [company['name'] for company in info.get('production_companies', [])[:5]]
        }
    except Exception as e:
        return None

# ----------------------------------------------------
# 3. Fetch Data (Reduced Pages for Quicker Local Run)
# ----------------------------------------------------

print(f"Fetching data for {PAGES_TO_FETCH * 20} movies...")
all_movie_data = []

for page in tqdm(range(1, PAGES_TO_FETCH + 1), desc="Fetching Pages"):
    try:
        discover = tmdb.Discover()
        response = discover.movie(page=page, sort_by='popularity.desc')
        
        page_movie_ids = [movie['id'] for movie in response['results']]
        
        for movie_id in page_movie_ids:
            data = get_movie_data(movie_id)
            if data:
                all_movie_data.append(data)
            time.sleep(0.05) 
            
    except Exception as e:
        print(f"Error on page {page}: {e}")
        time.sleep(1)

df = pd.DataFrame(all_movie_data)
print(f"\nSuccessfully fetched and processed data for {len(df)} movies.")

# ----------------------------------------------------
# 4. Feature Engineering (Simplified for Streamlit Inputs)
# ----------------------------------------------------

# Convert release_date and extract date features
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['release_year'] = df['release_date'].dt.year
df['release_month'] = df['release_date'].dt.month
df['release_dayofweek'] = df['release_date'].dt.dayofweek

# Drop rows where release_date failed conversion (NaNs)
df.dropna(subset=['release_year'], inplace=True)

# Engineer Genre Features (Matching the app.py multiselect logic)
# Note: We need the list of genres used in the notebook to be consistent with the app.
top_genres = df['genres'].explode().value_counts().nlargest(10).index
for genre in top_genres:
    df[f'genre_{genre}'] = df['genres'].apply(lambda x: 1 if genre in x else 0)

# Add the new genre dummy features to the list of expected Streamlit features
STREAMLIT_INPUT_FEATURES.extend([f'genre_{g}' for g in top_genres])

# ----------------------------------------------------
# 5. Prep Data and Train the Streamlit-Compatible Model
# ----------------------------------------------------

# Define features (X) and target (y)
y = df['revenue']
X = df[STREAMLIT_INPUT_FEATURES].copy() # CRITICAL: Only use the simple input features

# Final check for NaNs in X before training
X.dropna(inplace=True)
y = y[X.index] # Align y with cleaned X

print(f"\nFinal training dataset size (Streamlit features only): {X.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=11)

# Use the best parameters found in your original notebook (0.05, max_depth: 5, n_estimators: 100)
best_params = {'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 100}
gbr_final_for_app = GradientBoostingRegressor(**best_params, random_state=11)

print("\nTraining final Streamlit-compatible model...")
gbr_final_for_app.fit(X_train, y_train)

# ----------------------------------------------------
# 6. Save the Final Model and Feature List
# ----------------------------------------------------

# 1. Save the trained model object
joblib.dump(gbr_final_for_app, 'movie_revenue_model.pkl')

# 2. Save ONLY the list of feature names the Streamlit app must provide
joblib.dump(X.columns.tolist(), 'model_features.pkl')

print("\n---------------------------------------------------------")
print("SUCCESS: New model files generated locally.")
print("Saved Model Features (model_features.pkl):")
print(X.columns.tolist())
print("---------------------------------------------------------")
print("NEXT STEP: UPLOAD both 'movie_revenue_model.pkl' and 'model_features.pkl'")
print("to your Hugging Face repository, overwriting the old files.")
