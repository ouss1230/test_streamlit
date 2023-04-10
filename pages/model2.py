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
# GOSSIS app

This app predicts mortality
''')
st.subheader('Instructions')
st.write('''

Please use the left sidebar to input your patients measured clinical variables and then click Submit.
''')

st.sidebar.header('User Input Parameters')
def app():

    def user_input_features():
        with st.sidebar.form("my_form"):
            f1 = st.text_input('d1_mbp_noninvasive_min', 64.941541)
            f2 = st.text_input('d1_mbp_min', 64.871859)
            f3 = st.text_input('d1_temp_min', 36.268391)

            f4 = st.text_input('d1_arterial_ph_min', 7.324530)
            f5 = st.text_input('d1_sysbp_noninvasive_min', 96.993313)
            f6 = st.text_input('d1_sysbp_min', 96.92387)
            f7 = st.text_input('d1_spo2_min', 90.454826)
            f8 = st.text_input('ventilated_apache (1 or 0)', 1)
            f9 = st.text_input('gcs_verbal_apache (1 to 5)', 4)
            f10 = st.text_input('gcs_eyes_apache (1 to 4)', 4)
            f11 = st.text_input('gcs_motor_apache (1 to 6)', 6)
            f12 = st.text_input('d1_lactate_min', 2.125128)
            f13 = st.text_input('d1_lactate_max', 2.927383)
            options = {
                "KNN": "KNN (Manhattan and k = 15)",
                "RF": "Random Forest Classifier",
                "LR": "Logistic Regression",
            }

            option = st.selectbox("Model", list(options.items()), 0, format_func=lambda o: o[1])

            global op
            op = option[0]

            submitted = st.form_submit_button("Submit")

        data = {'d1_mbp_noninvasive_min': f1,
                'd1_mbp_min': f2,

                'd1_temp_min': f3,


                'd1_arterial_ph_min': f4,
                'd1_sysbp_noninvasive_min': f5,
                'd1_sysbp_min': f6,
                'd1_spo2_min': f7,
                'ventilated_apache': f8,
                'gcs_verbal_apache': f9,
                'gcs_eyes_apache': f10,

                'gcs_motor_apache': f11,
                'd1_lactate_min': f12,
                'd1_lactate_max': f13}

        features = pd.DataFrame(data, index=[0])
        return features

    df = user_input_features()

    time.sleep(30)
    st.subheader('User Input Parameters')
    st.write(df)

    #####################

    target = {0.: "Predicted in-hospital mortality: No", 1.: "Predicted in-hospital mortality: Yes"}

    ####################

    grid_search_SMOTE = joblib.load('/Users/oussamagharsellaoui/Desktop/import/grid_search_SMOTE_{}.pkl'.format(op))
    if op == "RF":
        scaler = joblib.load('/Users/oussamagharsellaoui/Desktop/import/scaler')
        df = scaler.transform(df)
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
        feature = ('d1_mbp_noninvasive_min', 'd1_mbp_min', 'd1_temp_min',
       'd1_arterial_ph_min', 'd1_sysbp_noninvasive_min', 'd1_sysbp_min',
       'd1_spo2_min', 'ventilated_apache', 'gcs_verbal_apache',
       'gcs_eyes_apache', 'gcs_motor_apache', 'd1_lactate_min',
       'd1_lactate_max')
        y_pos = np.arange(len(feature))
        performance = np.array([2487.1482812172494, 2501.645244195261, 2772.4861992363826, 2106.6239494659358, 2858.323611405216, 2877.039682106197, 3009.85726652349, 3388.7694778989317, 3853.905054477654, 4522.604107438405, 5431.547120691283, 5381.4150178009995, 6816.322395894872])
        error = np.random.rand(len(feature))

        ax.barh(y_pos, performance, xerr=error, align='center', color='#0F0F0F')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Importance Score')
        ax.set_title('The ANOVA f-test Feature Importance', fontdict={'color': 'white'})
        # ax.set_facecolor('silver')
        # fig.patch.set_facecolor('#121212')
        fig.patch.set_facecolor('#0F0F0F')

        # fig.patches.Patch()

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