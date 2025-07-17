import streamlit as st
import pickle
import numpy as np

# Load model
with open("academic_predictor.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🎓 Academic Success Predictor")

st.write("Enter student academic data:")

attendance = st.slider("Attendance (%)", 0, 100, 75)
midterm = st.slider("Midterm Score", 0, 100, 68)
assignment1 = st.slider("Assignment 1 Score", 0, 100, 80)
assignment2 = st.slider("Assignment 2 Score", 0, 100, 80)
quiz1 = st.slider("Quiz 1 Score", 0, 100, 70)
quiz2 = st.slider("Quiz 2 Score", 0, 100, 70)
participation = st.slider("Participation Score", 0, 100, 85)
previous = st.slider("Last Year Score", 0, 100, 75)
missed = st.slider("Missed Assignments", 0, 2, 0)

if st.button("Predict"):
    input_data = np.array([[attendance, midterm, assignment1, assignment2, quiz1, quiz2 , participation, previous, missed]])
    prediction = model.predict(input_data)[0]
    result = "Pass" if prediction == 1 else "Fail"
    st.subheader(f"Prediction: {result}")
