import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import joblib
import os

# -------------------------------------------------------------
# STEP 1: SYNTHETIC SENSOR DATA GENERATION
# -------------------------------------------------------------
# Purpose:
# We simulate accelerometer readings (ax, ay, az) for 3 activities:
# Walking, Running, Sitting.
# Each activity has unique motion patterns + noise.
# -------------------------------------------------------------

def generate_activity_data(label, samples, pattern_strength):
    """
    Generates synthetic accelerometer-like data.
    label: 'walking', 'running', 'sitting'
    samples: number of time steps
    pattern_strength: amplitude of the activity
    """

    t = np.linspace(0, 4*np.pi, samples)

    if label == "walking":
        ax = np.sin(t) * pattern_strength + np.random.normal(0, 0.2, samples)
        ay = np.cos(t) * pattern_strength + np.random.normal(0, 0.2, samples)
        az = np.ones(samples) + np.random.normal(0, 0.1, samples)

    elif label == "running":
        ax = np.sin(2*t) * (pattern_strength*1.5) + np.random.normal(0, 0.3, samples)
        ay = np.cos(2*t) * (pattern_strength*1.5) + np.random.normal(0, 0.3, samples)
        az = np.ones(samples) + np.random.normal(0, 0.15, samples)

    elif label == "sitting":
        ax = np.random.normal(0, 0.05, samples)
        ay = np.random.normal(0, 0.05, samples)
        az = np.ones(samples) + np.random.normal(0, 0.02, samples)

    df = pd.DataFrame({
        "ax": ax,
        "ay": ay,
        "az": az,
        "label": [label]*samples
    })
    return df


# -------------------------------------------------------------
# STEP 2: GENERATE FULL DATASET
# -------------------------------------------------------------
walking = generate_activity_data("walking", 800, 0.8)
running = generate_activity_data("running", 800, 1.2)
sitting = generate_activity_data("sitting", 800, 0.3)

df = pd.concat([walking, running, sitting], ignore_index=True)

print("Dataset created with shape:", df.shape)
print(df.head())


# -------------------------------------------------------------
# STEP 3: FEATURE EXTRACTION
# -------------------------------------------------------------
# Purpose:
# Convert time-series into ML-friendly features (mean, std, energy, etc.).
# Window-based approach: each window becomes one training example.
# -------------------------------------------------------------

WINDOW = 50  # 50 time-steps per window
STEP = 50    # non-overlapping windows

def extract_features(data):
    features = []
    labels = []

    for i in range(0, len(data) - WINDOW, STEP):
        window = data.iloc[i:i+WINDOW]

        ax = window["ax"].values
        ay = window["ay"].values
        az = window["az"].values

        feats = [
            np.mean(ax), np.std(ax), np.min(ax), np.max(ax),
            np.mean(ay), np.std(ay), np.min(ay), np.max(ay),
            np.mean(az), np.std(az), np.min(az), np.max(az),
            np.mean(np.sqrt(ax*ax + ay*ay + az*az)),  # magnitude
        ]

        features.append(feats)
        labels.append(window["label"].iloc[0])

    return np.array(features), np.array(labels)


X, y = extract_features(df)
print("Feature matrix shape:", X.shape)
print("Labels shape:", y.shape)


# -------------------------------------------------------------
# STEP 4: TRAIN ML MODEL
# -------------------------------------------------------------
model = RandomForestClassifier(n_estimators=120, random_state=42)
model.fit(X, y)

y_pred = model.predict(X)

acc = accuracy_score(y, y_pred)
print("\nTraining Accuracy:", acc)


# -------------------------------------------------------------
# STEP 5: CONFUSION MATRIX
# -------------------------------------------------------------
os.makedirs("outputs", exist_ok=True)

cm = confusion_matrix(y, y_pred, labels=["walking", "running", "sitting"])

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=["walking","running","sitting"],
            yticklabels=["walking","running","sitting"], cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png")
plt.close()

print("Confusion matrix saved in outputs/confusion_matrix.png")


# -------------------------------------------------------------
# STEP 6: SAVE MODEL
# -------------------------------------------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/har_levelc_model.joblib")

print("Model saved to models/har_levelc_model.joblib")
