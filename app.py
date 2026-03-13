# Streamlit analytics dashboard for WHALER MVP

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.title('WHALER MVP Analytics Dashboard')

# Upload CSV file
uploaded_file = st.file_uploader("Choose a CSV file containing earnings data", type='csv')

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data)

    # Visualization
    st.subheader('Earnings Over Time')
    plt.figure(figsize=(10, 5))
    plt.plot(data['Date'], data['Earnings'], marker='o')
    plt.title('Earnings Trends')
    plt.xlabel('Date')
    plt.ylabel('Earnings')
    st.pyplot()

    # Add more visualizations as needed
