Human Activity Recognition (HAR) – Synthetic Accelerometer Project



This project simulates how a wearable device (like a smartwatch) detects human activities using accelerometer signals.

Instead of using a large external dataset, the script generates synthetic but realistic time-series sensor data for three activities:



Walking



Running



Sitting



The goal is to understand the full ML workflow behind activity recognition systems — from raw signals → feature extraction → model training → evaluation.



~ What This Project Does



Generates 3-axis accelerometer data with realistic patterns



Splits data into fixed-size windows



Extracts meaningful statistical features from each window



Trains a Random Forest classifier



Evaluates performance using a confusion matrix



Saves the trained model (har\_levelc\_model.joblib)



Even though the dataset is synthetic, the overall process closely matches real HAR systems.



~ Project Structure

HAR\_LevelC/

│ har\_levelc.py                # Main ML pipeline (data → features → training)

│ README.md

│

├── outputs/

│   └── confusion\_matrix.png   # Saved evaluation plot

│

└── models/

&nbsp;   └── har\_levelc\_model.joblib   # Saved ML model



~ Results



The model typically achieves 95–100% accuracy, since the activities have distinct patterns.



A confusion matrix is saved here:



outputs/confusion\_matrix.png





It shows how well the model distinguishes between walking, running, and sitting.



~ Skills Demonstrated



This project helped me understand:



How accelerometer data looks for different activities



How to extract useful features from time-series data



How to train and evaluate ML classifiers



How to visualize correctness using a confusion matrix



How to save ML models



How to structure a clean, reproducible ML project



This builds foundational knowledge for real HAR systems and IoT/embedded ML work.



~ How to Run the Project

Install required packages:

pip install numpy pandas matplotlib seaborn scikit-learn joblib



Run:

python har\_levelc.py





The model and confusion matrix image will be generated automatically.



~ Why This Project Matters



Human Activity Recognition is widely used in:



Fitness trackers



Health monitoring



Gesture recognition



Wearable devices



Smart home systems



IoT and embedded ML applications



This project gives a beginner-friendly, end-to-end understanding of how such systems work internally — making it a strong addition to a portfolio or internship application.

