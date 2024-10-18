# Import necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import pickle

# Load dataset 
df = pd.read_csv('global_development_data.csv')

X = df[['GDP per capita', 'headcount_ratio_upper_mid_income_povline', 'year']]  # Features
y = df['Life Expectancy (IHME)']  # Target variable


# Initialize the RandomForestRegressor model
rf_model = RandomForestRegressor(
    n_estimators=100,  # Number of trees in the forest
    max_depth=None,  # Maximum depth of the tree
    random_state=42
)

# Train the model
rf_model.fit(X, y)


# Save the model to a file
with open('rf_model.pkl', 'wb') as model_file:
    pickle.dump(rf_model, model_file)

print("Model trained and saved successfully.")