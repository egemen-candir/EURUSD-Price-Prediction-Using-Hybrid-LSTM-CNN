# Define features
features = ['close', 'high', 'low', 'SMA_5', 'SMA_20', 'RSI', 'MACD', 'ATR', 'Momentum']
sequence_length = 10

# Scale the features
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(all_data[features])

# Create sequences
X = []
y = []
for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])
    y.append(scaled_data[i, 0])  # 0 index corresponds to 'close' price
X, y = np.array(X), np.array(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Build the hybrid CNN-LSTM model
model = Sequential([
    # CNN layers
    Conv1D(filters=64, kernel_size=3, activation='relu', 
           input_shape=(sequence_length, len(features))),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),
    
    Conv1D(filters=128, kernel_size=3, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),
    
    # LSTM layers
    LSTM(100, return_sequences=True),
    BatchNormalization(),
    Dropout(0.2),
    
    LSTM(50),
    BatchNormalization(),
    Dropout(0.2),
    
    # Dense layers
    Dense(50, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(1)
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
             loss='huber',
             metrics=['mae'])

# Define callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
]

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# Make predictions
predicted_scaled = model.predict(X_test)

# Prepare for inverse transform
pred_full = np.zeros((len(predicted_scaled), len(features)))
pred_full[:, 0] = predicted_scaled.flatten()  # Put predictions in first column (close price)
y_test_full = np.zeros((len(y_test), len(features)))
y_test_full[:, 0] = y_test  # Put actual values in first column

# Inverse transform
predicted_prices = scaler.inverse_transform(pred_full)[:, 0]  # Get only the close price
actual_prices = scaler.inverse_transform(y_test_full)[:, 0]  # Get only the close price