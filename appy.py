import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load the data
def load_data():
    df = pd.read_csv('bank.csv', delimiter=';')  # Adjust the delimiter if necessary
    return df

df = load_data()

# Preprocess the data
X = df.drop(columns=[' class'])  # Features
y = df[' class']  # Target variable

# Standardize features (if necessary)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create and train the model
model = LogisticRegression(penalty='l1', solver='liblinear')
model.fit(X_scaled, y)

# Streamlit UI
st.title("Bankruptcy Prediction App")
st.write("Enter the features to predict the class (bankruptcy or non-bankruptcy).")

# Get user input for features
input_features = []
for feature_name in X.columns:
    input_feature = st.number_input(f"Enter {feature_name}:", min_value=0.0, max_value=1.0, step=0.01)
    input_features.append(input_feature)

# Predict the class based on user input
if st.button("Predict"):
    user_input = pd.DataFrame([input_features], columns=X.columns)
    user_input_scaled = scaler.transform(user_input)
    prediction = model.predict(user_input_scaled)
    st.success(f"The predicted class is: {prediction[0]}")
