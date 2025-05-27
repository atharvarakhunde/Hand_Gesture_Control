import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Load data
data = pd.read_csv("gesture_data_normalized.csv")  # Adjust path if needed

# Split features and labels
X = data.drop('label', axis=1).values
y = data['label'].values

# Encode labels
label_encoder = joblib.load("label_encoder.pkl")
y_encoded = label_encoder.transform(y)
y_categorical = tf.keras.utils.to_categorical(y_encoded)

# Split into train/test (same ratio as used during training)
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Load model
model = tf.keras.models.load_model("gesture_control_model.h5")

# Predict
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
labels = label_encoder.classes_

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# Classification report
print(classification_report(y_true, y_pred, target_names=labels))
