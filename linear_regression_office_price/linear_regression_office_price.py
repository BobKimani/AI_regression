import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the dataset
data = pd.read_csv('Nairobi Office Price Ex.csv')

# 2. Extract the relevant columns ('SIZE' and 'PRICE')
size = data['SIZE'].values
price = data['PRICE'].values

# Normalize the data for better gradient descent performance
size_mean = np.mean(size)
size_std = np.std(size)
price_mean = np.mean(price)
price_std = np.std(price)

size_norm = (size - size_mean) / size_std
price_norm = (price - price_mean) / price_std

# 3. Initialize random values for slope (m) and intercept (c)
m = np.random.randn()  # Random initial slope
c = np.random.randn()  # Random initial intercept
learning_rate = 0.01
epochs = 10  # Train for 10 epochs as per criteria

# 4. Define Mean Squared Error (MSE) function
def compute_mse(size, price, m, c):
    n = len(size)
    predictions = m * size + c
    mse = (1 / n) * np.sum((price - predictions) ** 2)
    return mse

# 5. Define Gradient Descent function
def gradient_descent(size, price, m, c, learning_rate):
    n = len(size)
    predictions = m * size + c

    # Compute gradients
    dm = -(2 / n) * np.sum(size * (price - predictions))
    dc = -(2 / n) * np.sum(price - predictions)

    # Update parameters
    m = m - learning_rate * dm
    c = c - learning_rate * dc

    return m, c

# 6. Train the model for 10 epochs
for epoch in range(epochs):
    m, c = gradient_descent(size_norm, price_norm, m, c, learning_rate)
    mse = compute_mse(size_norm, price_norm, m, c)
    print(f"Epoch {epoch + 1}, Slope: {m}, Intercept: {c}, MSE: {mse}")

# 7. De-normalize the slope and intercept for original scale predictions
m_original = m * price_std / size_std
c_original = c * price_std + price_mean - m_original * size_mean

# 8. Make prediction for 100 sq. ft office
predicted_price_100 = m_original * 100 + c_original
print(f"Predicted price for 100 sq. ft office: {predicted_price_100}")

# 9. Plot the data and the line of best fit
plt.scatter(size, price, color='blue', label='Data Points')
plt.plot(size, m_original * size + c_original, color='red', label='Best Fit Line')
plt.xlabel('Office Size (sq. ft)')
plt.ylabel('Office Price')
plt.title('Linear Regression: Office Price vs Size')
plt.legend()
plt.show()
