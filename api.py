import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pulp import *


# Parameters
products = [14,  16,  20,  21,  29,  30,  32,  38,  39,  41,  49, 138, 148]
I=len(products)
T=30    # Time horizon
equation = 'TotalQty ~ FreeQty + Q("1") + Q("2") + Q("3") + Q("4") + Q("5") + Q("6") + Q("7") + Q("8") + Q("9") + Q("10") + Q("11") + Q("12") + Q("13") + Q("14") + Q("15") + Q("16") + Q("17") + Q("18") + Q("19") + Q("20") + Q("21") + Q("22") + Q("23") + Q("24") + Q("25") + Q("26") + Q("27") + Q("28") + Q("29") + Q("30") + Q("31") - 1'

@st.cache_data

def getData():
    df = pd.read_parquet("Sales_Data_v2.parquet")
    df["TotalQty"] = df["PaidQty"] + df["FreeQty"]
    df = df.groupby([df['Date'].dt.date, df["ProductId"]])[["TotalQty", "FreeQty"]].sum()
    df = df.reset_index()
    return df


def fit_forecasting_model(df):
    coeff = []
    
    for p in products:
        d = df[df.ProductId == p]
        d["Date"] = pd.to_datetime(d.Date)
        d["dayofmonth"] = d.Date.dt.day
        d = d[["Date", "TotalQty", "FreeQty", "dayofmonth"]].set_index("Date")
        one_hot = pd.get_dummies(d['dayofmonth']).astype(int)
        d = d.drop('dayofmonth', axis = 1)
        d = d.join(one_hot)
        d.columns = [str(i) for i in list(d.columns)]
        fit = smf.ols(equation, d).fit()
        coeff.append(fit.params)
    
    coeff = pd.DataFrame(coeff)
    coeff.index = products
    Base_Demand = np.array(coeff.drop("FreeQty", axis=1))
    Promo_Sensitivity = np.array(coeff.FreeQty)

    return Base_Demand, Promo_Sensitivity
    

def optimize_promotions(Base_Demand, Promo_Sensitivity, Total_PaidQty_Limit, Total_FreeQty_Limit,
                        max_promo_length, min_promo_length, daily_paid_cap, daily_free_cap):
    
    # Parameters

    PHorizon = list(range(0, T))
    AugHorizon = list(range(1, T))
    Products = list(range(0, I))

    # LP Model

    Promo_Model = LpProblem("Pharma_Promotion_Model", LpMaximize)

    # Decision variables

    # Paid Quantity
    P = LpVariable.dicts("Paid_Quantity",(Products,PHorizon),lowBound=0, upBound=daily_paid_cap, cat='Integer')
    # Free Quantity
    F = LpVariable.dicts("Free_Quantity",(Products,PHorizon),lowBound=0, upBound=daily_free_cap, cat='Integer')
    # Is Product "i" promoted at time "t"?
    X = LpVariable.dicts("IsPromoted",(Products,PHorizon),cat='Binary')
    # Promotion Length
    L = LpVariable.dicts("Promo_Length",Products,lowBound=min_promo_length, upBound=max_promo_length, cat='Integer')
    
    # Objective Function
    Total_Paid_Quantity = lpSum(lpSum(P[i][t] for i in Products) for t in PHorizon)
    Total_Free_Quantity = lpSum(lpSum(F[i][t] for i in Products) for t in PHorizon)
    Promo_Model += Total_Paid_Quantity

    # Constraints

        # DEMAND SATISFACTION
    for i in Products:
        for t in PHorizon:
            Promo_Model += P[i][t] + F[i][t] <= Base_Demand[i][t] + Promo_Sensitivity[i]*F[i][t]
    # NO FREE QTY WITHOUT PROMOTION
    Big_M = daily_free_cap
    for i in Products:
        for t in PHorizon:
            Promo_Model += F[i][t] <= X[i][t]*Big_M
            Promo_Model += F[i][t] >= X[i][t]-0.5
    # TOTAL PAID QUANTITY PER PRODUCT
    for i in Products:
        Promo_Model += lpSum(P[i][t] for t in PHorizon) <= Total_PaidQty_Limit
    # TOTAL FREE QUANTITY PER PRODUCT
    for i in Products:
        Promo_Model += lpSum(F[i][t] for t in PHorizon) <= Total_FreeQty_Limit
    # PROMOTED PRODUCTS AT TIME "t"
    for t in PHorizon:
        Promo_Model += lpSum(X[i][t] for i in Products) <= 13
    # PROMO PERIOD FOR PRODUCT "i"  
    for i in Products:
        Promo_Model += lpSum(X[i][t] for t in PHorizon) == L[i]

    # Solve the Model
        
    solver = GUROBI()
    solver.solve(Promo_Model)

    # Promo Ratio Calculation

    Promo_Ratio = np.zeros((I, T), dtype = float)
    for i in Products:
        for t in PHorizon:
            Promo_Ratio[i][t] = F[i][t].varValue/(P[i][t].varValue + F[i][t].varValue)
    df_Promo_Ratio = pd.DataFrame(Promo_Ratio,
                     index = products,
                     columns = range(1,T+1))
    
    return df_Promo_Ratio


# LAYOUT

st.set_page_config(page_title='Promo Planning App', layout='wide')
st.title("Pharmaceutical Promotion Planning App")

df = getData()

col1, col2 = st.columns((1, 2))
with col1:
    max_promo_length = st.number_input("Max Promotion Length", value=30)
    min_promo_length = st.number_input("Min Promotion Length", value=1)

    total_paid_cap = st.number_input("Total Paid Quantity Capacity", value=10000000)
    total_free_cap = st.number_input("Total Free Quantity Capacity", value=10000)

    daily_paid_cap = st.number_input("Daily Paid Quantity Capacity", value=100000)
    daily_free_cap = st.number_input("Daily Free Quantity Capacity", value=1000)

    daily_product_count = st.number_input("Daily Promoted Product Count", value=1)
    # horizon = st.slider("Time Horizon", 1, 31, 3)

with col2:
    Base_Demand, Promo_Sensitivity = fit_forecasting_model(df)
    df_Promo_Ratio = optimize_promotions(Base_Demand, Promo_Sensitivity, total_paid_cap, total_free_cap, 
                                         max_promo_length, min_promo_length, daily_paid_cap, daily_free_cap)

    st.dataframe(df_Promo_Ratio)