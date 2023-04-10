import streamlit as st
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import recall_score, make_scorer
import joblib
import time
import matplotlib.pyplot as plt
import time



def app():
    st.write('''
    # Mortality Prediction in ICU App and GOSSIS app

    this is the home page 
    ''')