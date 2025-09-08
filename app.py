import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

# ------------------------------
# Define the pipeline
# ------------------------------
def create_pipeline(numeric_features, categorical_features):
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())  # Replace with trained model
    ])

    return pipeline

# ------------------------------
# Streamlit App UI
# ------------------------------
st.title("Mumbai Train Delay Predictions")
st.write("Predictions for all trains in the dataset")

# Load dataset automatically
try:
    df = pd.read_csv("Mumbai Local Train Dataset.csv", encoding='latin1')
except FileNotFoundError:
    st.error("Dataset not found. Please add the CSV to the project folder.")
    st.stop()
except UnicodeDecodeError:
    st.error("Failed to read CSV due to encoding issues. Try opening the file and saving it as UTF-8.")
    st.stop()


# Define features
numeric_features = ['Distance_km', 'Time_min', 'Speed_kmph', 'Passengers_daily', 'Expected_time_min']
categorical_features = ['Station', 'Line']

# Create pipeline
pipeline = create_pipeline(numeric_features, categorical_features)

# Dummy target for fitting (since we don't train here)
y_dummy = np.zeros(len(df))
pipeline.fit(df, y_dummy)

# Make predictions
predictions = pipeline.predict(df)
df['Predicted_Delay_min'] = predictions

st.subheader("Predicted Delays")
st.dataframe(df)

# Optional: show some summary stats
st.subheader("Summary Statistics")
st.write(df['Predicted_Delay_min'].describe())

