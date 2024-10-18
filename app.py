import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle

st.set_page_config(layout="wide")
st.header("Worldwide Analysis of Quality of Life and Economic Factors")

st.write("This app enables you to explore the relationships between poverty, life expectancy, and GDP across various countries and years. Use the panels to select options and interact with the data.")
tab1, tab2, tab3 = st.tabs(["Global Overview", "Country Deep Dive", "Data Explorer"])

df = pd.read_csv("global_development_data.csv")

with tab1:
    st.header("Global overview")
    min_year = int(df['year'].min()) 
    max_year = int(df['year'].max())
    selected_year = st.slider(
    "Select year for visualization", 
    min_value=min_year, 
    max_value=max_year, 
    value=min_year  # Default value (optional)
    )

    # Filter the DataFrame based on the selected year
    filtered_data = df[df['year'] == selected_year]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.write("mean of life expectancy")
        mean_life_expectancy = filtered_data["Life Expectancy (IHME)"].mean()
    with col2:
        st.write("Global Median GDP per capita")
        medianGPT = filtered_data["GDP per capita"].median()
    with col3:
        st.write("Global Poverty Average")
        headcount_ratio_upper_mid_income_povline_mean = filtered_data["headcount_ratio_upper_mid_income_povline"].mean()
    with col4:
        st.write("Number of Countries")
        num_countries = len(filtered_data["country"].unique())

    fig = px.scatter(
                filtered_data, 
                x="GDP per capita", 
                y="Life Expectancy (IHME)",
                hover_name="country",
                log_x=True,
                size="Population",
                color="country", 
                title="Scatter Plot of GDP per Capita Over Time"
            )
    st.plotly_chart(fig)

    st.header("Prediction")
    feature1 = st.number_input("GDP per capita")
    feature2 = st.number_input("headcount_ratio_upper_mid_income_povline")
    feature3 = st.number_input("year")
    X_input = np.array([[feature1, feature2, feature3]])  

    # Load the model from a file
    with open('rf_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    
    # Use the model for prediction
    y_pred = model.predict(X_input)
    st.write(f"predicted life expectancy: {y_pred[0]:.2f} years")

    feature_importances = model.feature_importances_
    importance_df = pd.DataFrame({
    'Feature': ['GDP per capita', 'Headcount Ratio Upper Middle Income Poverty Line', 'Year'],
    'Importance': feature_importances
    })
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title='Feature Importance')
    st.plotly_chart(fig)


with tab3:
    st.header("Data Explorer")
    st.dataframe(df)

    selected_countries = st.multiselect("Select countries to view", df.country.unique(), default="China")
    min_year = int(df['year'].min())
    max_year = int(df['year'].max())

    selected_year_range = st.slider("Select the year range", min_year, max_year, (min_year, max_year))

    data = df[
        (df['country'].isin(selected_countries)) & 
        (df['year'] >= selected_year_range[0]) & 
        (df['year'] <= selected_year_range[1])
    ]
    st.dataframe(data)

    csv = data.to_csv(index=False)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='data.csv',
    )



