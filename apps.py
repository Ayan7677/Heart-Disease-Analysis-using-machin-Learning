import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the heart disease dataset (replace 'heart.csv' with your dataset)
df = pd.read_csv('heart.csv')

# Display the dataset
st.subheader('Heart Disease Dataset')
st.write(df)

# Sidebar - Collects user input features into dataframe
st.sidebar.header('User Input Features')

def user_input_features():
    age = st.sidebar.slider('Age', 29, 77, 50)
    sex = st.sidebar.selectbox('Sex', [1,0])
    cp = st.sidebar.selectbox('Chest Pain Type', [0, 1, 2, 3])
    trestbps = st.sidebar.slider('Resting Blood Pressure (mm Hg)', 94, 200, 120)
    chol = st.sidebar.slider('Cholesterol (mg/dl)', 126, 564, 250)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', [0, 1])
    restecg = st.sidebar.selectbox('Resting Electrocardiographic Results', [0, 1, 2])
    thalach = st.sidebar.slider('Maximum Heart Rate Achieved', 71, 202, 150)
    exang = st.sidebar.selectbox('Exercise Induced Angina', [0, 1])
    oldpeak = st.sidebar.slider('ST Depression Induced by Exercise Relative to Rest', 0.0, 6.2, 3.0)
    slope = st.sidebar.selectbox('Slope of the Peak Exercise ST Segment', [0, 1, 2])
    ca = st.sidebar.slider('Number of Major Vessels (0-3) Colored by Flourosopy', 0, 3, 1)
    thal = st.sidebar.selectbox('Thalassemia', [0, 1, 2, 3])
    user_data = {'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
                 'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
                 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal}
    features = pd.DataFrame(user_data, index=[0])
    return features

# Assign user input features to a variable
user_input = user_input_features()

# Define features and target variable
X = df.drop('target', axis=1)
y = df['target']

# Train the model
model = RandomForestClassifier()
model.fit(X, y)

# Make predictions
prediction = model.predict(user_input)
prediction_proba = model.predict_proba(user_input)

# Display prediction
st.subheader('Prediction')
target_labels = ['No Heart Disease', 'Heart Disease']
st.write(target_labels[prediction[0]])
st.write('Probability:', prediction_proba[0][1])
# Additional information
st.subheader('Prediction Probability')
st.write('Probability of No Heart Disease:', prediction_proba[0][0])
st.write('Probability of Heart Disease:', prediction_proba[0][1])
