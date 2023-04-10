import streamlit as st
from multiclass import MultiAppfunc
from pages import home




app = MultiAppfunc()

st.markdown("""
# Welcome to the Web Application

Please Choose one model: 

""")

# Add all your application here
app.add_app("home", home.app)


# The main app
app.run()