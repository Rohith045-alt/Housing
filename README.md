House Price Prediction using Linear Regression
This README summarizes the steps taken to build and evaluate a Linear Regression model for predicting house prices using the Housing.csv dataset.

Project Steps and Discussion
Step 1: Data Import and Preprocessing
The initial phase involved loading the dataset and preparing it for model training.

Data Loading: The Housing.csv dataset was loaded into a Pandas DataFrame.

Initial Inspection: We examined the first few rows and the data types of each column. The dataset contained 545 entries with no missing values. It included 6 numerical columns (price, area, bedrooms, bathrooms, stories, parking) and 7 categorical columns (mainroad, guestroom, basement, hotwaterheating, airconditioning, prefarea, furnishingstatus).

Categorical Feature Encoding: To make the categorical features usable by the Linear Regression model, one-hot encoding was applied. This converted each categorical column into multiple binary (0 or 1) numerical columns. The drop_first=True argument was used to avoid multicollinearity by dropping one category from each feature.

Result: The DataFrame was transformed, resulting in 14 numerical columns, ready for modeling.

Step 2: Train-Test Split
After preprocessing, the dataset was divided into training and testing sets to ensure the model's performance could be evaluated on unseen data.

Feature and Target Separation:

The price column was identified as the target variable (y).

All remaining columns were designated as features (X).

Data Splitting: The train_test_split function from sklearn.model_selection was used to divide the data:

X_train, y_train: 80% of the data for training the model (436 samples, 13 features).

X_test, y_test: 20% of the data for testing the model's performance (109 samples, 13 features).

random_state=42 was used for reproducibility of the split.

Step 3: Fit a Linear Regression Model
With the data prepared, a Linear Regression model was instantiated and trained.

Model Initialization: A LinearRegression model from sklearn.linear_model was created.

Model Training: The model was fitted to the training data (X_train and y_train). This process involves the model learning the relationships between the features and the target variable by finding the optimal coefficients for each feature.

Step 4: Evaluate Model Performance
The trained model's effectiveness was assessed using standard regression metrics.

Predictions: The trained model made predictions (y_pred) on the X_test dataset.

Metric Calculation:

Mean Absolute Error (MAE): Calculated as $970,043.40. This indicates that, on average, the model's predictions are off by this amount from the actual house prices.

Mean Squared Error (MSE): Calculated as $1,754,318,687,330.66. This metric penalizes larger errors more heavily.

R-squared (R²): Calculated as 0.6529. This means that approximately 65.29% of the variance in house prices can be explained by the features included in the model. An R² of 0.6529 suggests a moderately good fit.

Step 5: Plot Regression Line and Interpret Coefficients
Finally, the model's predictions were visualized, and the learned coefficients were interpreted to understand feature importance.

Regression Line Plot (Actual vs. Predicted):

A scatter plot was generated showing the actual house prices on the x-axis and the predicted house prices on the y-axis.

A red dashed line representing the ideal scenario (where actual equals predicted) was overlaid for comparison.

This plot helps visualize how closely the model's predictions align with the true values.

Coefficient Interpretation:

Coefficients: Each coefficient indicates the change in house price for a one-unit increase in a feature, holding others constant.

Positive Coefficients: Features like area, bedrooms, bathrooms, stories, parking, mainroad_yes, guestroom_yes, basement_yes, hotwaterheating_yes, airconditioning_yes, and prefarea_yes had positive coefficients. This suggests that an increase in these features generally leads to a higher predicted house price. For instance, each additional bathroom adds approximately $1,094,440 to the price, and having air conditioning adds about $791,427.

Negative Coefficients: furnishingstatus_semi-furnished and furnishingstatus_unfurnished had negative coefficients. This indicates that, compared to a fully furnished house (the baseline), semi-furnished and unfurnished houses are predicted to be cheaper.

Intercept: The model's intercept was $260,032.36, representing the estimated baseline price when all other features are zero.

Conclusion
This project successfully demonstrates the process of building a Linear Regression model for house price prediction. From data preprocessing and splitting to model training, evaluation, and interpretation, each step provides valuable insights into the factors influencing house prices and the model's predictive capabilities. The model explains about 65% of the variance in house prices, indicating a reasonable performance.
