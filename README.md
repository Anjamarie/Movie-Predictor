# Movie Revenue Predictor: A Machine Learning Deployment Case Study
## Project Overview

This project implements a machine learning pipeline to predict a movie's potential worldwide revenue based on key pre-production and release characteristics (Budget, Runtime, Release Timing, and Genre).

Crucially, this repository serves as a case study in robust data science deployment, demonstrating solutions to common pitfalls encountered when deploying large model files on platforms like Streamlit Cloud.

### View the Live Application: (https://anjamarie-movie-predictor.hf.space/)

## Feature Engineering
The model uses the following key features, derived from TMDb data, to make predictions:

| Feature                         | Type              | Description                                      |
| ------------------------------- | ----------------- | ------------------------------------------------ |
| `budget`                        | Numerical         | The production budget of the movie in USD.       |
| `runtime`                       | Numerical         | The movie's length in minutes.                   |
| `release_year`                  | Numerical         | The year the movie was released.                 |
| `release_month`                 | Numerical         | The month of release (e.g., 6 for June).         |
| `release_dayofweek`             | Numerical         | The day of the week of release (0=Mon, 6=Sun).   |
| `genre_Action`, `genre_Drama`... | Binary (0 or 1)   | Dummy variables for the top 10 most popular genres.|
| `mean_cast_revenue`             | Numerical         | The average historical revenue of the main cast. |
| `mean_director_revenue`         | Numerical         | The average historical revenue of the director.  |

## Project Structure and Deployment Solution
This project highlights a solution for managing and loading large model assets during deployment.

| File/Component     | Purpose                                                                | Deployment Fix                                                                                                                                              |
| ------------------ | ---------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `app.py`           | Streamlit Application Code. Builds the UI and runs prediction logic.   | Implements a custom function using `requests` with a network timeout and explicit CDN URLs to bypass 403 Forbidden and firewall issues with Git LFS.           |
| `movie_revenue_model.pkl` | Trained Gradient Boosting Model.                                       | Stored externally on Hugging Face to avoid Git LFS usage limits and ensure a fast, direct download during app boot.                                     |
| `model_features.pkl` | Model Feature Schema.                                                  | Contains only the 8 Streamlit-compatible features, resolving a data incompatibility bug that causes prediction crashes.                             |
| `requirements.txt` | Dependencies.                                                          | Ensures all libraries, including `pandas`, `joblib`, and `huggingface-hub`, are installed correctly on the cloud server.                                    |


## How the Model Files are Loaded (The Fix)
Instead of relying on the local environment or Git LFS, the app.py script executes the following process upon deployment:

Checks if movie_revenue_model.pkl exists in Streamlit's resource cache (@st.cache_resource).

If not found, it performs an authenticated, direct download from the public Hugging Face repository using the stable file URL.

The file is saved locally and loaded using joblib.load().

This isolates the heavy asset from Git, allowing the code base to remain clean and lightweight.


## Model Performance
The final Gradient Boosting Regressor, after hyperparameter tuning and feature engineering, achieved the following performance on the test set:

- **R-squared (R²)**: 0.90
- **Root Mean Squared Error (RMSE)**: ~$115.5 Million

An R-squared of 0.90 indicates that the model can explain 90% of the variance in movie revenue, demonstrating a high degree of predictive accuracy.

The model identified several key drivers of movie revenue. The top 5 most important features were:

| Feature                               | Importance |
| ------------------------------------- | ---------- |
| `mean_cast_revenue`                   | 0.35       |
| `budget`                              | 0.28       |
| `mean_production_companies_revenue`   | 0.15       |
| `runtime`                             | 0.08       |
| `mean_director_revenue`               | 0.05       |

This indicates that the historical revenue of a movie's cast and its production budget are the strongest predictors of its future success.


## Results and Conclusion
This project successfully demonstrated that a machine learning model can predict movie revenue with a high degree of accuracy (R² of 0.90) using pre-release data. By engineering features from categorical data like cast and production companies, the model was able to significantly improve its predictive power. The final deployed Gradio application serves as a successful proof-of-concept for a tool that could aid in content acquisition and investment decisions.

While the model is highly accurate, its predictions are limited by the available TMDb data. Future improvements could include:

Incorporating NLP: Analyzing the movie's script or plot summary to extract thematic features.
Adding More Data Sources: Integrating data from social media trends or critic reviews to provide a more holistic view.
Advanced Feature Engineering: Creating more nuanced features, such as the chemistry between specific actors or a director's success within a particular genre.
