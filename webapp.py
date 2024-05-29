import numpy as np
import pickle
import streamlit as st

# Load the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

# Function for prediction
def diabetes_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    
    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

# Main function to run the app
def main():
    # Title of the app
    st.title('Diabetes Prediction Web App')
    st.write("""
    ## Predict if a person has diabetes using machine learning
    Fill in the details below to get the prediction.
    """)

    # Organize inputs into columns
    col1, col2 = st.columns(2)

    with col1:
        Pregnancies = st.number_input('Number of Pregnancies', min_value=0, step=1)
        Glucose = st.slider('Glucose Level', 0, 200, 110)
        BloodPressure = st.slider('Blood Pressure value', 0, 150, 70)
        SkinThickness = st.slider('Skin Thickness value', 0, 100, 20)
    
    with col2:
        Insulin = st.slider('Insulin Level', 0, 900, 30)
        BMI = st.slider('BMI value', 0.0, 70.0, 15.0)
        DiabetesPedigreeFunction = st.slider('Diabetes Pedigree Function value', 0.0, 2.5, 0.5)
        Age = st.number_input('Age of the Person', min_value=1, max_value=120, step=1)

    # Code for prediction
    diagnosis = ''

    # Create a button for prediction
    if st.button('Diabetes Test Result'):
        with st.spinner('Predicting...'):
            diagnosis = diabetes_prediction([
                Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, 
                BMI, DiabetesPedigreeFunction, Age
            ])
        st.success(diagnosis)

    # Sidebar for additional information
    st.sidebar.header("About")
    st.sidebar.write("""
    This app uses a machine learning model to predict whether a person is diabetic based on various health parameters.
    """)
    st.sidebar.write("""
    **Model Details:**
    - Model: Trained on the PIMA Indian Diabetes Dataset
    - Features: Number of pregnancies, Glucose level, Blood pressure, Skin thickness, Insulin level, BMI, Diabetes pedigree function, Age
    """)

# Run the app
if __name__ == '__main__':
    main()
