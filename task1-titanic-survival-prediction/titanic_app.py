import streamlit as st 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import pickle

import base64

@st.cache_resource ()
def predict_sample (sample) :
    loaded_model = pickle.load (open ('LogisticRegression.sav', 'rb'))
    sample = np.array (sample).reshape (1, -1)
    prediction = loaded_model.predict (sample) 
    print (prediction [0])

    if prediction [0] == 1 :
        st.success ("Relieving news ! The passenger has survived the Titanic sinking.")
    else :
        st.error ("Our regrets ! The passenger did not survive the Titanic sinking.")

@st.cache_resource ()
def exploratory_analysis (data) :
    st.header ('Exploratory Data Analysis')
    st.markdown ('Dataset :')
    st.write (data.head ())

    st.subheader ('1. Dataset Loading')
    st.write (f"Shape of the dataset : {data.shape}")

    st.markdown ("Descriptive statistics :\n")
    st.write (data.describe ())

    st.subheader ('2. Data Processing')
    st.markdown ('''In the dataset, there are none relevent informations (variables)
                for making the prediction such as the first column, "PassengerID",
                "Name", "Ticket" and "Cabin". So we will drop those variables.''')
    none_relevent_variables = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    for variable in none_relevent_variables :
        data = data.drop (variable, axis = 1)
    st.markdown ('Lets take a look to the head of the processed data :')
    st.write (data.head ())

    st.subheader ('3 Handling Missing (NaN) Values')
    st.markdown ('''This step is important to ensure accurate analysis and modeling.
                There are various methods to handle missing values in a pandas dataframe :\n\n
    * Drop rows or columns\n
    * Fill with a constant value\n
    * Fill with statical measures (mean, median, mode)\n
    * Forward or backward fill\n
    * Interpolation\n

    \nLet's explore the dataset looking for NaN values.''')
    st.markdown (f"Missing values (NaN) in the resulting dataset :\n ")
    # Retrive the names of variable that presents NaN values
    nan_columns = [column for column in data.columns if data [column].isna ().sum () > 0]
    for variable in nan_columns :
        # Count the number of NaN Values
        nan_number = data [variable].isna ().sum ()
        # Compute the percentage according to the entire dataset
        nans_percentage = round ((nan_number / len (data)) * 100, 3)
        st.markdown (f"{variable} has {nan_number} NaN values, which represents {nans_percentage} % of the whole dataset.")
    
    st.markdown (f"""Variables with NaN values types :\n
    Age's type : {data ['Age'].dtypes}\n
    Fare's type : {data ['Fare'].dtypes}""")

    st.markdown ('''Both variables `Age` and `Fare` holds numerical values.
    To handle their NaN values, the appropriate method is to *fill them with
    a statical measure*. We will use the most obvious statical measure which is the `mean`.''')
    
    # Get the mean of each of Age and Fare
    mean_age = data ['Age'].mean ()
    st.write ('The mean values of the variable `Age` is :', mean_age)
    mean_fare = data ['Fare'].mean ()
    st.write ('The mean values of the variable `Fare` is :', mean_fare)

    # Fill NaN values with the mean
    data ['Age'].fillna (mean_age, inplace = True)
    data ['Fare'].fillna (mean_fare, inplace = True)
    st.markdown ('To replace the NaN values with the mean, we used the `fillina ()` function')

    st.write ('Sum of NaN values after replacing them with the means :', data.isna ().sum ())

    st.subheader ('2.5 Handling Duplicated registrations')
    st.markdown ('''Here, we will adopt the easiest and obvious way of handling
    duplicate rows i.e. we are just going to drop them.''')
    st.write ('Dataset shape before dropping duplicates :', data.shape)
    # Count the number of duplicated values
    st.write ('Duplicated Values Count :', data.duplicated ().sum ())
    # Drop duplicated values
    data = data.drop_duplicates ()
    st.write ('Data head after dropping duplicates using the function `drop_duplicates () :', data.head ())

    st.subheader ('Encoding Categorical Features')
    st.markdown ('To avoid potential bias or assumtions by the coming algorithm, we will do one-hot encoding via the method `pd.get_dummies ()`.')
    data = pd.get_dummies (data)
    st.markdown (f"Shape of the encoded data set : {data.shape}")
    st.write ('Head of the encoded data', data.head ())

    st.subheader ('2. Correlation')
    st.markdown ('''
        Now that all variable in the dataset are numerical, let's see their correlations.Let r be the value of the correlation between two variables :
        * If **r < 0**, the two variables are **negatively** correlated
        * If **r = 0**, the two variables are **uncorrelated**
        * If **r > 0**, the two variables are **positively** correlated

        Notes :
        * If **0 <= |r| < 0.2**, then **almost no correlation**
        * If **0.2 <= |r| < 0.4**, then **low correlation**
        * If **0.4 <= |r| < 0.6**, then **average or moderate correlation**
        * If **0.6 <= |r| < 0.8**, then **good correlation**
        * If **0.8 <= |r| <= 0.1**, then **high correlation**
            - If **0.9 <= |r| <= 0.99**, then **excellent correlation**
            - If **|r| = 1**, **perfect correlation**
    ''')
    correlation = data.corr ()
    # st.write (correlation)
    fig, ax = plt.subplots (figsize = (14, 10))
    sns.heatmap (correlation, annot = True, cmap = 'Blues')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot ()
    st.caption ('Correlation Matrix')

def prediction () :
    st.subheader ('Sir/Mme, YOU need to fill all necessaty informations in order to make a prediction about the survival !')
    
    col1, col2, col3 = st.columns (3)
    with col1 :
        sex = st.radio ('Sex', ['Female', 'Male'])
        Pclass = st.radio ('Pclass', [1, 2, 3])
        embarked = st.radio ('Embarked', ['C', 'Q', 'S'])
    with col2 :
        SibSp = st.slider ('SibSp', 0, 1, 8)
        Parch = st.slider ('Parch', 0, 1, 9)
    with col3 :
        Age = st.number_input ('Age', step = 1., format = "%.2f", min_value = 0.0)
        Fare = st.number_input ('Fare', step = 1.0, format = "%.4f", min_value = 0.0)
    
    Sex_female, Sex_male = 0, 0
    if sex == 'Female' :
        Sex_female = 1
    else :
        Sex_male = 1
    
    Embarked_C, Embarked_Q, Embarked_S = 0, 0, 0
    if embarked == 'C' :
        Embarked_C = 1
    elif embarked == 'Q' :
        Embarked_Q = 1
    else :
        Embarked_S =1

    sample = (Pclass, Age, SibSp, Parch, Fare, Sex_female, Sex_male, Embarked_C, Embarked_Q, Embarked_S)
    if st.button ('Predict') :
        result = predict_sample (sample)

def main () :
    st.title ('Titanic Survival Prediction')
    st.image ('titanic.jpg')
    data = pd.read_csv ("tested.csv")

    app_mode = st.sidebar.selectbox ('Select Page', ['Home', 'Exploratory Data Analysis', 'Prediction'])
    # app_mode = 'Exploratory Data Analysis'
    # app_mode = 'Prediction'
    if app_mode == 'Exploratory Data Analysis' :
        exploratory_analysis (data)
    elif app_mode == 'Prediction' :
        prediction ()
    else :
        st.subheader ('Synopsis')
        st.markdown ('''This project is about building a model that predicts
        whether a passenger on the Titanic survived or not using the Titanic
        dataset provided [here](https://www.kaggle.com/datasets/brendan45774/test-file).
        It is a classic beginner project with readily available data. The
        dataset typically used for this project contains information about
        individual passengers, such as their age, gender, ticket class,
        fare, cabin, and whether or not they survived.''')
    
    st.markdown ('Made by H. Moustapha Ousmane')


if __name__ == '__main__' :
    main ()
