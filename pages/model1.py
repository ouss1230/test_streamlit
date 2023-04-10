
import streamlit as st
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import recall_score, make_scorer
import joblib
import time
import matplotlib.pyplot as plt
import time
start_time = time.time()

st.write('''
# Mortality Prediction in ICU App

This app predicts mortality
''')
st.subheader('Instructions')
st.write('''

Please use the left sidebar to input your patients measured clinical variables and then click Submit.
''')

st.sidebar.header('User Input Parameters')


def specificity_scorer(y_true, y_pred):
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

def app():
    def user_input_features():
        with st.sidebar.form("my_form"):
            f1 = st.text_input('Temperature F', 98.24)
            f2 = st.text_input('Arterial Blood Pressure mean (mmHg)',80.574799)
            f3 = st.text_input('Hematocrit', 31.209394)

            f5 = st.text_input('glucose',134.806876)
            f6 = st.text_input('serum glucose', 135.253645)
            f7 = st.text_input('White blood cells', 12.026495)
            f8 = st.text_input('Potassium serum',4.970038)
            f9 = st.text_input('Systolic ABP', 119.180408)
            f10 = st.text_input('Fraction of Inspired Oxygen (FiO2)',  0.540000)
            f11 = st.text_input('Heart Rate', 85.053329)
            f4 = st.text_input('Creatinine', 1.573615)
            f12 = st.text_input('Respiratory Rate (insp/min)', 19.237812)
            f13 = st.text_input('Glasgow Coma Scale (GCS)',  12.670284)
            submitted = st.form_submit_button("Submit")




        data = {'Temperature F' : f1,
                'ABP mean' : f2,
                'Arterial Blood Pressure mean': f2,
                'Hematocrit': f3,

                'FiO2 3': float(f10) * 100,
                'Creatinine 1': f4,
                'crea 2': f4,
                'White blood cells 1': f7,
                'crea 1': f4,
                'glucose' : f5,
                'serum glucose': f6,
                'White blood cells 2': f7,

                'Potassium serum' : f8,
                'Systolic ABP' : f9,
                'FiO2 1': f10,
                'Heart Rate 1' : f11,
                'Creatinine 2': f4,
                'crea 3': f4,
                'Respiratory Rate 4' : f12,
                'Respiratory Rate 8': f12,
                'Respiratory Rate 0': f12,
                'Respiratory Rate 6': f12,

                'GCS' : f13}

        features = pd.DataFrame(data, index = [0])
        return features



    df = user_input_features()

    time.sleep(30)
    st.subheader('User Input Parameters')
    st.write(df)







    #####################


    target = {0. : "Predicted in-hospital mortality: No", 1. : "Predicted in-hospital mortality: Yes"}

    ####################

    grid_search_SMOTE = joblib.load('/Users/oussamagharsellaoui/Desktop/grid_search_SMOTE.pkl')

    y_pred_SMOTE = grid_search_SMOTE.predict(df)
    y_pred_SMOTE_Proba = grid_search_SMOTE.predict_proba(df)

    #####################

    st.subheader('Prediction')
    st.write(target[y_pred_SMOTE[0]])

    ###################
    st.subheader('Prediction Probability')
    st.write(y_pred_SMOTE_Proba)

    st.subheader('Feature Importance')

    def plot():
        # Fixing random state for reproducibility
        np.random.seed(19680801)


        plt.rcdefaults()
        fig, ax = plt.subplots()

        # Example data
        feature = ('Temperature F', 'Arterial Blood Pressure mean', 'Hematocrit',
               'Glucose', 'Serum Glucose', 'White blood cells', 'Potassium Serum', 'Systolic ABP',
               'FiO2', 'Heart Rate', 'Creatinine', 'Respiratory Rate', 'GCS')
        y_pos = np.arange(len(feature))
        performance = np.array([  63.74708701,   69.05565628,   82.51479313,  222.02963092,
                230.25553567,  242.07161324,  243.47163398,  283.68505253,
                284.86777474,  291.66598997,  367.56226471,  965.42848092,
               3953.53753601])
        error = np.random.rand(len(feature))

        ax.barh(y_pos, performance, xerr=error, align='center', color = '#0F0F0F')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Importance Score')
        ax.set_title('The ANOVA f-test Feature Importance',fontdict ={'color': 'white'} )
        #ax.set_facecolor('silver')
        #fig.patch.set_facecolor('#121212')
        fig.patch.set_facecolor('#0F0F0F')


        #fig.patches.Patch()

        ax.yaxis.label.set_color('w')
        ax.xaxis.label.set_color('w')
        ax.tick_params(axis='x', colors='w')
        ax.tick_params(axis='y', colors='w')

        return st.pyplot(fig)

    plot()



    st.write('Running time: ', " %s seconds " % round((time.time() - start_time), 2))

    st.write('''
    
    # Contact Us
    
    email: oussemagh@hotmail.com''')

