import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
import zipfile
import requests
from io import BytesIO

# ------------------------------
# Streamlit App UI
# ------------------------------
st.title("Mumbai Train Delay Predictions")
st.write("Predictions for all trains from Git ZIP dataset")

# ------------------------------
# Download ZIP from GitHub
# ------------------------------
zip_url = "https://github.com/yourusername/yourrepo/raw/main/mumbai_local_project.zip"  # replace with your Git ZIP URL

try:
    st.info("Downloading dataset from Git...")
    response = requests.get(zip_url)
    if response.status_code != 200:
        st.error("Failed to download ZIP from GitHub")
        st.stop()

    # Load ZIP into memory
    zip_file = zipfile.ZipFile(BytesIO(response.content))

    # List files in ZIP
    zip_files = zip_file.namelist()
    st.write("Files in ZIP:", zip_files)

    # Pick the CSV file (replace with actual CSV name inside ZIP)
    csv_name = [f for f in zip_files if f.endswith(".csv")][0]

    # Read CSV into pandas
    with zip_file.open(csv_name) as f:
        df = pd.read_csv(f, encoding='latin1')

except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# ------------------------------
# Clean column names
# ------------------------------
df.columns = df.columns.str.strip()
st.subheader("Train Data")
st.dataframe(df)

# ------------------------------
# Define pipeline
# ------------------------------
numeric_features = ['Distance_km', 'Time_min', 'Speed_kmph', 'Passengers_daily', 'Expected_time_min']
categorical_features = ['Station', 'Line']

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
    ('regressor', LinearRegression())  # Replace with trained model if available
])

# ------------------------------
# Fit pipeline with dummy target
# ------------------------------
y_dummy = np.zeros(len(df))  # just to allow pipeline to fit
pipeline.fit(df, y_dummy)

# ------------------------------
# Make predictions
# ------------------------------
predictions = pipeline.predict(df)
df['Predicted_Delay_min'] = predictions

st.subheader("Predicted Delays")
st.dataframe(df)

# Optional: summary
st.subheader("Summary Statistics")
st.write(df['Predicted_Delay_min'].describe())


