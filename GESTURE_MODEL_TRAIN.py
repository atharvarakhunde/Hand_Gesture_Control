import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import joblib

# Step 1: Load the dataset
print("Loading dataset...")
data = pd.read_csv("gesture_data_normalized.csv")  # Ensure this CSV is in the same directory

# Step 2: Extract features and labels
print("Preparing features and labels...")
X = data.drop('label', axis=1).values  # Feature matrix
y = data['label'].values               # Class labels

# Step 3: Encode labels into integers and then one-hot
print("Encoding labels...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)  # One-hot encoding

# Step 4: Split dataset into training and testing
print("Splitting data into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42
)

# Step 5: Build the neural network model
print("Building model...")
model = Sequential([
    Dense(128, input_shape=(X.shape[1],), activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(y_categorical.shape[1], activation='softmax')
])

# Step 6: Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 7: Train the model
print("Training the model...")
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.1)

# Step 8: Evaluate the model
print("Evaluating model on test data...")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {accuracy:.2f}")

# Step 9: Save the trained model
print("Saving model to 'gesture_control_model.h5'...")
model.save("gesture_control_model.h5")

# Step 10: Save the label encoder
print("Saving label encoder to 'label_encoder.pkl'...")
joblib.dump(label_encoder, "label_encoder.pkl")

print("\n🎉 Training complete. Model and label encoder saved successfully.")
