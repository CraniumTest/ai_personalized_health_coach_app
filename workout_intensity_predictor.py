import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Simulated user data (age, fitness_level (1-10), avg_weekly_activity_hours)
user_data = np.array([
    [25, 5, 2],
    [30, 7, 4],
    [22, 6, 3],
    [35, 8, 5],
    [40, 2, 0.5],
    [28, 4, 2.5],
    # More data points...
])

# Simulated weekly recommended workout intensity (hours)
labels = np.array([
    5,
    7,
    6,
    8,
    3,
    5,
    # Corresponding labels...
])

# Preprocess data
X_train, X_test, y_train, y_test = train_test_split(user_data, labels, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define a simple feedforward neural network
model = Sequential([
    Dense(32, input_dim=X_train.shape[1], activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model
model.fit(X_train_scaled, y_train, epochs=50, batch_size=2, verbose=1)

# Evaluate the model
loss, mae = model.evaluate(X_test_scaled, y_test)
print(f"Test MAE: {mae}")

# Predict for a new user
new_user = np.array([[29, 5, 3]])  # Example age, fitness level, avg weekly hours
new_user_scaled = scaler.transform(new_user)
predicted_intensity = model.predict(new_user_scaled)
print(f"Recommended weekly workout intensity (hours): {predicted_intensity[0, 0]:.2f}")
