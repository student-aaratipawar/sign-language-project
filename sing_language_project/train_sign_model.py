import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

print("🚀 Training Sign Language Model...")

# ✅ Load the fixed dataset
df = pd.read_csv('hand_landmarks_ready.csv')

print(f"📊 Dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns")

# Check for 'label' column
if 'label' not in df.columns:
    print("❌ ERROR: 'label' column not found! Please check your CSV file.")
    exit()

# Split data and labels
X = df.drop('label', axis=1)
y = df['label']

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
accuracy = model.score(X_test, y_test)
print(f"✅ Model trained successfully with {accuracy*100:.2f}% accuracy!")

# Save model and encoder
with open('sign_language_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

print("💾 Model saved as sign_language_model.pkl")
print("💾 Label encoder saved as label_encoder.pkl")
print("🎉 Training complete! You can now run live_sign_detection.py")
