import gradio as gr
import pandas as pd
import joblib

# --- 1. Load the Model and Feature List ---
# These are loaded only once when the app starts.
model = joblib.load('movie_revenue_model.pkl')
model_features = joblib.load('model_features.pkl')

# --- 2. Create the Prediction Function ---
# This function takes user inputs and returns the prediction.
def predict_revenue(budget, runtime, release_year, genre):
    
    # Create an empty DataFrame with the correct feature columns
    # and set all values to 0.
    input_df = pd.DataFrame(columns=model_features)
    input_df.loc[0] = 0
    
    # --- Feature Engineering ---
    # Update the DataFrame with the user's input values.
    input_df.at[0, 'budget'] = budget
    input_df.at[0, 'runtime'] = runtime
    input_df.at[0, 'release_year'] = release_year
    
    # One-hot encode the selected genre.
    genre_column = f'genre_{genre}'
    if genre_column in input_df.columns:
        input_df.at[0, genre_column] = 1
        
    # --- Prediction ---
    # Make the prediction and format the output.
    prediction = model.predict(input_df)[0]
    formatted_prediction = f"${prediction:,.2f}"
    
    return formatted_prediction

# --- 3. Define the Gradio Interface ---
# Create the input and output components for the app.
iface = gr.Interface(
    fn=predict_revenue,
    inputs=[
        gr.Number(label="Budget ($)"),
        gr.Number(label="Runtime (minutes)"),
        gr.Slider(minimum=2000, maximum=2025, step=1, label="Release Year"),
        gr.Dropdown(
            choices=['Action', 'Adventure', 'Drama', 'Thriller', 'Comedy', 'Science Fiction', 'Fantasy'],
            label="Genre"
        )
    ],
    outputs=gr.Textbox(label="Predicted Revenue"),
    title="Movie Revenue Predictor",
    description="Enter the movie details to predict its box office revenue."
)

# --- 4. Launch the App ---
iface.launch()
