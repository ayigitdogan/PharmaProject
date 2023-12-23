import streamlit as st
import pandas as pd
import numpy as np


@st.cache_data
def getData():
    return pd.read_parquet("sale_data.parquet")


def calculate(df):
    return df


st.set_page_config(page_title='Promo Planning App', layout='wide')
st.title("Pharmaceutical Promotion Planning App")

df = getData()

col1, col2 = st.columns((1, 2))
with col1:
    max_promo_length = st.number_input("Max Promotion Length", value=30)
    max_promo_length = st.number_input("Min Promotion Lengrh", value=1)

    total_paid_cap = st.number_input("Total Paid Quantity Capacity", value=0)
    total_free_cap = st.number_input("Total Free Quantity Capacity", value=0)

    daily_paid_cap = st.number_input("Daily Paid Quantity Capacity", value=0)
    daily_free_cap = st.number_input("Daily Free Quantity Capacity", value=0)

    daily_product_count = st.number_input("Daily Product Count", value=1)
    horizon = st.slider("Time Horizon", 1, 31, 3)

with col2:
    result = calculate(df)
    st.dataframe(result.head())
