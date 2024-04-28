import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st

model = pk.load(open('model.pkl', 'rb'))

st.header("Receipe Prediction")

recipe_data = pd.read_csv('All_Diets.csv')

def get_meal_name(cuisine_name):
    cuisine_name = cuisine_name.split(' ')[0]
    return cuisine_name.strip()

recipe_data['Recipe_name'] = recipe_data['Recipe_name'].apply(get_meal_name) 

diet_name = st.selectbox('Select diet type', recipe_data['Diet_type'].unique())
cuisine_type = st.selectbox('Cuisine type', recipe_data['Cuisine_type'].unique())

protein_min = recipe_data['Protein(g)'].min()
protein_max = recipe_data['Protein(g)'].max()
protein = st.slider('Protein intake', protein_min, protein_max)

carbs = st.selectbox('Carbs intake', recipe_data['Carbs(g)'].unique())
fat = st.selectbox('Fat intake', recipe_data['Fat(g)'].unique())

if st.button("Predict"):
    input_data_model = pd.DataFrame([[diet_name, cuisine_type, protein, carbs, fat]],
                                     columns=['Diet_type', 'Cuisine_type', 'Protein(g)', 'Carbs(g)', 'Fat(g)'])

    # Perform one-hot encoding for categorical variables
    input_data_model = pd.get_dummies(input_data_model, columns=['Diet_type', 'Cuisine_type'])

    # Ensure all columns in the input data frame match the model's feature names
    missing_cols = set(model.feature_names_in_) - set(input_data_model.columns)
    for col in missing_cols:
        input_data_model[col] = 0

    # Reorder columns to match the order used during training
    input_data_model = input_data_model[model.feature_names_in_]

    # Model prediction
    prediction = model.predict(input_data_model)

    st.markdown('Item will be   ' + str(prediction[0]))
