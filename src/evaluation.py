# Calculate and display metrics
mae = mean_absolute_error(actual_prices, predicted_prices)
rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
print(f'\nModel Performance Metrics:')
print(f'MAE: {mae:.4f}')
print(f'RMSE: {rmse:.4f}')

# Plot actual vs predicted prices
plt.figure(figsize=(12, 6))
plt.plot(actual_prices, label='Actual Prices')
plt.plot(predicted_prices, label='Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.xlabel('Time Steps')
plt.ylabel('Price')
plt.legend()
plt.show()

# Plot training history
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot delta between actual and predicted prices
delta = actual_prices - predicted_prices
plt.figure(figsize=(12, 6))
plt.plot(delta, label='Price Difference')
plt.title('Delta Between Actual and Predicted Prices')
plt.xlabel('Time Steps')
plt.ylabel('Price Difference')
plt.legend()
plt.show()

# Additional plot for momentum visualization
plt.figure(figsize=(12, 6))
plt.plot(all_data['Momentum'].iloc[-len(actual_prices):].values, label='Momentum')
plt.title('Cumulative Momentum')
plt.xlabel('Time Steps')
plt.ylabel('Momentum Value')
plt.legend()
plt.show()
