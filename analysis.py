import pandas as pd
import seaborn as sns
import sklearn as sk
from sklearn.model_selection import train_test_split
df = pd.read_csv('Housing.csv')
df.info()

# encoding all categorical values
df['mainroad'] = df['mainroad'].map({'yes': 1, 'no': 0})
df['guestroom'] = df['guestroom'].map({'yes': 1, 'no': 0})
df['basement'] = df['basement'].map({'yes': 1, 'no': 0})
df['hotwaterheating'] = df['hotwaterheating'].map({'yes': 1, 'no': 0})
df['airconditioning'] = df['airconditioning'].map({'yes': 1, 'no': 0})
df['prefarea'] = df['prefarea'].map({'yes': 1, 'no': 0})
df['furnishingstatus'] = df['furnishingstatus'].map({
    'furnished': 1,
    'semi-furnished': 2,
    'unfurnished': 3
})

#creating train test split
X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

#Finding "Mean Absolute Error, Mean Squared Error, R-squared
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error: ${mae:,.2f}")
print(f"Mean Squared Error: ${mse:,.2f}")
print(f"R-squared: ${r2:.4f}")


plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) # Ideal line
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs. Predicted House Prices")
plt.grid(True)
plt.show()

coefficients = pd.DataFrame({'Feature': X_train.columns, 'Coefficient': model.coef_})
print("\nModel Coefficients:")
print(coefficients.to_markdown(index=False, numalign="left", stralign="left"))

print(f"\nModel Intercept: ${model.intercept_:,.2f}")
