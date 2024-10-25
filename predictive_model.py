import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('NairobiOfficePriceEx.csv')
X = df['SIZE'].values  
y = df['PRICE'].values  

# Normalize features
X = (X - np.mean(X)) / np.std(X)

def compute_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def gradient_descent_step(X, y, m, c, learning_rate):
    n = len(X)
    y_pred = m * X + c
    
    dm = (-2/n) * np.sum(X * (y - y_pred))
    dc = (-2/n) * np.sum(y - y_pred)
    
    m_new = m - learning_rate * dm
    c_new = c - learning_rate * dc
    
    return m_new, c_new

# Training parameters
learning_rate = 0.01
epochs = 10

# Initialize parameters randomly
np.random.seed(42)
m = np.random.randn()
c = np.random.randn()

print("Training Data Summary:")
print(f"Number of samples: {len(X)}")
print(f"Average office size: {np.mean(df['SIZE']):.2f} sq ft")
print(f"Average price: ${np.mean(df['PRICE']):.2f}")

print("\nInitial Parameters:")
print(f"Initial slope (m): {m:.4f}")
print(f"Initial y-intercept (c): {c:.4f}")
print("\nTraining Progress:")

# Training loop
error_history = []
for epoch in range(epochs):
    y_pred = m * X + c
    mse = compute_mse(y, y_pred)
    error_history.append(mse)
    
    print(f"Epoch {epoch + 1}: MSE = {mse:.2f}")
    
    m, c = gradient_descent_step(X, y, m, c, learning_rate)

print("\nFinal Parameters:")
print(f"Final slope (m): {m:.4f}")
print(f"Final y-intercept (c): {c:.4f}")

# Predict price for 100 sq ft
test_size = 100
test_size_normalized = (test_size - np.mean(df['SIZE'])) / np.std(df['SIZE'])
predicted_price = m * test_size_normalized + c
print(f"\nPredicted price for {test_size} sq ft: ${predicted_price:.2f}k")

# Plotting
plt.figure(figsize=(12, 6))

# Plot actual data and regression line
plt.subplot(1, 2, 1)
plt.scatter(df['SIZE'], df['PRICE'], color='blue', alpha=0.5, label='Actual Data')
plt.plot(df['SIZE'], m * X + c, color='red', label='Regression Line')
plt.xlabel('Office Size (sq ft)')
plt.ylabel('Price ($k)')
plt.title('Linear Regression: Office Size vs Price')
plt.legend()
plt.grid(True)

# Plot error history
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), error_history, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('Training Error Over Time')
plt.grid(True)

plt.tight_layout()
plt.show()

# Model Performance Metrics
final_mse = error_history[-1]
r2_score = 1 - (final_mse / np.var(y))
print("\nModel Performance Metrics:")
print(f"Final MSE: {final_mse:.2f}")
print(f"R² Score: {r2_score:.4f}")

# Additional Analysis
print("\nCorrelation Analysis:")
correlation = np.corrcoef(df['SIZE'], df['PRICE'])[0,1]
print(f"Correlation between Size and Price: {correlation:.4f}")