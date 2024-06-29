import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load('titanic_model.pkl')

# Define the Streamlit app
st.title('Titanic Survival Prediction')

# User inputs for prediction
st.markdown('### Enter Passenger Details:')
Pclass = st.selectbox('Passenger Class', [1, 2, 3])
Sex = st.selectbox('Sex', ['male', 'female'])
Age = st.slider('Age', 0, 100, 30)
SibSp = st.slider('Siblings/Spouses Aboard', 0, 10, 0)
Parch = st.slider('Parents/Children Aboard', 0, 10, 0)
Fare = st.number_input('Fare', min_value=0.0, max_value=1000.0, value=50.0)

# Convert inputs to numeric
sex = 1 if Sex == 'female' else 0


# Predict function
def predict_survival(Pclass, Sex, Age, SibSp, Parch, Fare):
    input_data = np.array([[Pclass, sex, Age, SibSp, Parch, Fare]])
    prediction = model.predict(input_data)
    return prediction


# Predict button
if st.button('Predict'):
    prediction = predict_survival(Pclass, sex, Age, SibSp, Parch, Fare)
    if prediction[0] == 0:
        st.write('Unfortunately, the passenger is predicted not to survive.')
    else:
        st.write('The passenger is predicted to survive!')

