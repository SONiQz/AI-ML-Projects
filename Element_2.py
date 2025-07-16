import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np

data = pd.read_csv('buying_history.csv')

train_data = data[data['ItemsBought'].notnull()]  # Rows with non-null target values
test_data = data[data['ItemsBought'].isnull()]  # Rows with null target values

features = ['Year', 'Month']  # List of feature column names
target = 'ItemsBought'  # Target column name

X_train = train_data[features]
y_train = train_data[target]
X_test = test_data[features]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define Different Regression Models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor()
}

# Iterate over the models and evaluate their performance
for model_name, model in models.items():
    # Split the training data into train-validation sets
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=42
    )

    # Train the model
    model.fit(X_train_split, y_train_split)

    # Make predictions on the validation set
    y_val_pred = model.predict(X_val_split)

    # Evaluate the model using mean squared error (MSE)
    mse = mean_squared_error(y_val_split, y_val_pred)
    print(f"{model_name} - MSE: {mse:.2f}")

##### Selecting the Lowest MSE Results in Most Accurate Handling ######
params_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10]
}

# Perform grid search to find the best hyperparameters
grid_search = GridSearchCV(estimator=GradientBoostingRegressor(), param_grid=params_grid, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)

# Get the best model with the tuned hyperparameters
best_model = grid_search.best_estimator_

# Make predictions on the test set
predicted_values = best_model.predict(X_test_scaled)
predicted_values = np.round(predicted_values, decimals=1)

# Update the missing values in the original DataFrame with the predicted values
data.loc[data[target].isnull(), target] = predicted_values

# Save the updated DataFrame to a new CSV file
data.to_csv('updated_data2.csv', index=False)
