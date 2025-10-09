# Movie Revenue Predictor: A Machine Learning Deployment Case Study
## Project Overview

This project implements a machine learning pipeline to predict a movie's potential worldwide revenue based on key pre-production and release characteristics (Budget, Runtime, Release Timing, and Genre).

Crucially, this repository serves as a case study in robust data science deployment, demonstrating solutions to common pitfalls encountered when deploying large model files on platforms like Streamlit Cloud.

### View the Live Application: (https://anjamarie-movie-predictor.hf.space/)

## Prediction Features
The model uses the following eight key features, derived from TMDb data, to make predictions:

| Feature | Type | Description |

| budget | Numerical | The production budget of the movie. |

| runtime | Numerical | The movie's length in minutes. |

| release_year | Numerical | The year the movie was released. |

| release_month | Numerical | The month of release (e.g., 6 for June). |

| release_dayofweek | Numerical | The day of the week of release (0=Mon, 6=Sun). |

| genre_Action, genre_Drama, etc. | Binary (0 or 1) | Dummy variables for the 10 most popular genres. |


## Project Structure and Deployment Solution
This project highlights a solution for managing and loading large model assets during deployment.

| File/Component | Purpose | Deployment Fix |
| app.py |
Streamlit Application Code. Builds the UI and runs the prediction logic. | Implements a custom function using requests with a network timeout and explicit CDN URLs to bypass known 403 Forbidden and firewall issues associated with Streamlit's proxy and Git LFS. |
| movie_revenue_model.pkl | 

Trained Gradient Boosting Model. | Stored externally on Hugging Face to avoid Git LFS usage limits and ensure a fast, direct download during app boot. |
| model_features.pkl | 

Model Feature Schema. | Contains only the 8 Streamlit-compatible features (not the mean-encoded features), resolving the data incompatibility bug that causes prediction crashes. |

| requirements.txt | Dependencies. | Ensures all libraries, including pandas, joblib, and huggingface-hub, are installed correctly on the cloud server. |



## How the Model Files are Loaded (The Fix)
Instead of relying on the local environment or Git LFS, the app.py script executes the following process upon deployment:

Checks if movie_revenue_model.pkl exists in Streamlit's resource cache (@st.cache_resource).

If not found, it performs an authenticated, direct download from the public Hugging Face repository using the stable file URL.

The file is saved locally and loaded using joblib.load().

This isolates the heavy asset from Git, allowing the code base to remain clean and lightweight.


## Model Performance
Waiting on feedback from streamlit.io

## Results and Conclusion
Waiting on feedbak from streamlit.io
