#!/usr/bin/env python
# coding: utf-8

# In[5]:


import streamlit as st
import numpy as np
import pandas as pd
import joblib


model = joblib.load("linear_regression_model.pkl")

# Streamlit UI
st.title("Salary Prediction App")
st.sidebar.header("User Input")


age = st.sidebar.slider("Age", min_value=18, max_value=65, value=30)
hourly_rate = st.sidebar.slider("Hourly Rate", min_value=30, max_value=100, value=50)
percent_salary_hike = st.sidebar.slider("Percent Salary Hike", min_value=0, max_value=25, value=10)
performance_rating = st.sidebar.slider("Performance Rating", min_value=1, max_value=4, value=3)
total_working_years = st.sidebar.slider("Total Working Years", min_value=0, max_value=40, value=10)

st.sidebar.subheader("Department:")
selected_department = st.sidebar.selectbox("Select Department", ["R&D", "Sales", "HR"])

st.sidebar.subheader("Gender:")
selected_gender = st.sidebar.selectbox("Select Gender", ["Male", "Female"])

st.sidebar.subheader("Job Role:")
selected_job_role = st.sidebar.selectbox("Select Job Role", ["HealthCare", "HR", "Lab-Tech", "Manager", "Manufctr Dir", "Research Dir", "Research Scientist", "Sales Exe", "Sales Rep"])

def predict():
    try:
        # Map the selected categorical values to numerical values
        department_mapping = {"R&D": 1, "Sales": 2, "HR": 3}
        gender_mapping = {"Male": 1, "Female": 2}
        job_role_mapping = {
            "HealthCare": 1, "HR": 2, "Lab-Tech": 3, "Manager": 4,
            "Manufctr Dir": 5, "Research Dir": 6, "Research Scientist": 7,
            "Sales Exe": 8, "Sales Rep": 9
        }
        
        department = department_mapping[selected_department]
        gender = gender_mapping[selected_gender]
        job_role = job_role_mapping[selected_job_role]

        row = np.array([age, hourly_rate, percent_salary_hike, performance_rating, total_working_years, department, gender, job_role])
        X = pd.DataFrame([row])
        prediction = model.predict(X)[0]
        return prediction
    except Exception as e:
        return str(e)

result = ""
if st.button('Predict', on_click=predict):
    result = predict()

st.subheader("Prediction Result:")
if isinstance(result, (int, float)):
    st.success("The predicted income is Rs {:.2f}".format(result))
else:
    st.error("predicted income .......: {}".format(result))


# In[ ]:




