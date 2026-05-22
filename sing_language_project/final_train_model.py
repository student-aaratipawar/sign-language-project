import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import os

print("🚀 Final Sign Language Model Trainer")

# ✅ STEP 1: Automatically find your latest dataset
for name in ["hand_landmarks_ready.csv", "hand_landmarks_final.csv", "hand_landmarks_labeled.csv"]:
    if os.path.exists(name):
        csv_file = name
        break
else:
    print("❌ No dataset found! Please make sure your CSV file is in this folder.")
    exit()

print(f"📂 Using dataset: {csv_file}")

# ✅ STEP 2: Load dataset (force header correction)
df = pd.read_csv(csv_file, header=None)

# Create proper headers
columns = []
for i in range(21):
    columns += [f'x{i}', f'y{i}', f'z{i}']
columns.append('label')
df.columns = columns

# ✅ STEP 3: Remove any accidental header row in data
if not str(df.iloc[0, -1]).replace('.', '', 1).isdigit():
    df = df.iloc[1:]

print(f"📊 Dataset ready: {df.shape[0]} samples, {df.shape[1]} columns")

# ✅ STEP 4: Split features and labels
if 'label' not in df.columns:
    print("❌ 'label' column missing — please check your CSV structure.")
    exit()

X = df.drop('label', axis=1)
y = df['label']

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# ✅ STEP 5: Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"✅ Model trained successfully with {accuracy*100:.2f}% accuracy!")

# ✅ STEP 6: Save model and encoder
with open('sign_language_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

print("💾 Model saved as sign_language_model.pkl")
print("💾 Label encoder saved as label_encoder.pkl")
print("🎉 Training complete! You can now run live_sign_detection.py")
