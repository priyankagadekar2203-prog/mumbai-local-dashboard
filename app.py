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
    """
    Create a scikit-learn pipeline with preprocessing and model
    """
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
        ('regressor', LinearRegression())  # Replace with your trained model
    ])

    return pipeline

# ------------------------------
# Streamlit App UI
# ------------------------------
st.title("Mumbai Train Delay Prediction")

st.write("""
Upload a CSV file containing the following columns:
'Station', 'Line', 'Distance_km', 'Time_min', 'Speed_kmph', 'Passengers_daily', 'Expected_time_min', 'Delay_min'
""")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Input Data")
    st.dataframe(df)

    # Define features
    numeric_features = ['Distance_km', 'Time_min', 'Speed_kmph', 'Passengers_daily', 'Expected_time_min']
    categorical_features = ['Station', 'Line']

    # Create pipeline
    pipeline = create_pipeline(numeric_features, categorical_features)

    # Dummy target for fitting
    y_dummy = np.zeros(len(df))
    pipeline.fit(df, y_dummy)

    # Make predictions
    predictions = pipeline.predict(df)

    st.subheader("Predictions")
    df['Predicted_Delay_min'] = predictions
    st.dataframe(df)

else:
    st.info("Please upload a CSV file to make predictions.")
