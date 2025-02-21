import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load the trained disease prediction model
with open("disease_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load symptom-to-disease mapping
symptom_mapping = pd.read_csv("symptom_disease_mapping.csv")

st.title("Automatic Disease Diagnosis App")

# Collect user symptoms
selected_symptoms = st.multiselect("Select Symptoms", symptom_mapping["Symptom"].unique())

def get_symptom_vector(symptoms):
    vector = np.zeros(len(symptom_mapping))
    for symptom in symptoms:
        index = symptom_mapping[symptom_mapping["Symptom"] == symptom].index[0]
        vector[index] = 1
    return vector.reshape(1, -1)

if st.button("Predict Disease"):
    if selected_symptoms:
        symptom_vector = get_symptom_vector(selected_symptoms)
        prediction = model.predict(symptom_vector)[0]
        st.success(f"Predicted Disease: {prediction}")
    else:
        st.warning("Please select at least one symptom.")
