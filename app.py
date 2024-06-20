import streamlit as st
import pickle
import numpy as np
import base64

# Load the saved model
with open(r"C:\Users\suraj\Desktop\ML Project\diabetics_prediction_model", 'rb') as f:
    model = pickle.load(f)
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


img = get_img_as_base64("image.jpg")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://images.unsplash.com/photo-1501426026826-31c667bdf23d");
background-size: 180%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}

[data-testid="stSidebar"] > div:first-child {{
background-image: url("data:image/png;base64,{img}");
background-position: center; 
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Define the feature input fields
def get_user_input():
    st.header('User Input Features')

    pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=0, step=1)
    glucose = st.number_input('Glucose', min_value=0, max_value=200, value=120)
    blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=140, value=70)
    skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=20)
    insulin = st.number_input('Insulin', min_value=0, max_value=900, value=80)
    bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.5)
    age = st.number_input('Age', min_value=0, max_value=120, value=30)

    # Store inputs into a list
    features = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]

    return np.array(features).reshape(1, -1)

# Streamlit App
st.title('Diabetes Prediction')

# Set background image using HTML and CSS
st.markdown(
    """
    <style>
    body {
        background-image: url('background.jpg');
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Get user input
user_input = get_user_input()

# Add a button to generate prediction
if st.button('Predict'):
    # Make prediction
    prediction = model.predict(user_input)
    prediction_proba = model.predict_proba(user_input)

    # Display the result
    st.subheader('Prediction')
    if prediction[0] == 1:
        st.write('The model predicts that the individual is likely to have diabetes.')
    else:
        st.write('The model predicts that the individual is unlikely to have diabetes.')

    # Display input values
    st.subheader('Input values:')
    st.write(user_input)

    # Display prediction probability
    st.subheader('Prediction Probability')
    st.write(f'Probability of having diabetes: {prediction_proba[0][1]:.2f}')
