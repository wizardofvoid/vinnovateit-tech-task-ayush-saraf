import streamlit as st
import pickle
import numpy as np
from together import Together
import os

os.environ["TOGETHER_API_KEY"] = "074b4378fb0c7f824df7c0abbcb452bac9e95109f49283c790269a9a1c6e3db9"

def llama_prompt(attendance, midterm, assignment_avg, quiz_avg, participation, previous, missed, result):
    if(result==1):
        keyword = "pass"
    else:
        keyword = "fail"
    
    client = Together()
    response = client.chat.completions.create(
    model = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    messages=[
      {
        "role": "user",
        "content": f"""The student {keyword}. Their data is:
    - Attendance: {attendance}%
    - Midterm: {midterm}
    - Assignment Avg: {assignment_avg}
    - Quiz Avg: {quiz_avg}
    - Participation: {participation}
    - Previous Score: {previous}
    - Missed Assignments: {missed}

    Based on this data, what is the best area for them to improve? Give a short explanation."""
      }
    ]
    )
    return (response.choices[0].message.content)

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

assignment_avg = (assignment1+assignment2)/2
quiz_avg = (quiz1+quiz2)/2
if st.button("Predict"):
    input_data = np.array([[attendance, midterm, assignment_avg, quiz_avg, participation, previous, missed]])
    prediction = model.predict(input_data)[0]
    result = "Pass" if prediction == 1 else "Fail"
    st.subheader(f"Prediction: {result}")
    with st.spinner("Analyzing with LLaMA..."):
        explaination = llama_prompt(attendance, midterm, assignment_avg, quiz_avg, participation, previous, missed, result)
        st.markdown("### AI Explanation")
        st.write(explaination.strip())
