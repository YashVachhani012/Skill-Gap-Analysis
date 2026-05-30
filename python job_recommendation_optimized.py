# ==============================================
# 🧠 OPTIMIZED JOB RECOMMENDATION MODEL (.h5)
# ==============================================

# 1️⃣ Install dependencies (run once)
# pip install pandas openpyxl rapidfuzz pdfplumber tensorflow scikit-learn matplotlib seaborn

import os
import re
import pickle
import pdfplumber
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rapidfuzz import fuzz
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf

Sequential = tf.keras.models.Sequential
load_model = tf.keras.models.load_model

Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout

Adam = tf.keras.optimizers.Adam

# ----------------------------
# STEP 1: LOAD AND PREPARE DATA
# ----------------------------

JOB_DATA_PATH = "Jobdata.xlsx"  # Excel file

df_jobs = pd.read_excel(JOB_DATA_PATH, engine="openpyxl")
df_jobs = df_jobs.dropna(subset=["Job Title", "Skills Required"])

print("✅ Job data loaded successfully!")
print(df_jobs.head())

# Combine multiple columns to enrich features
cols_to_combine = [
    "Job Title", "Skills Required", "Experience Required", "Job Location", "Company Name"
]
for col in cols_to_combine:
    if col not in df_jobs.columns:
        df_jobs[col] = ""

df_jobs["text_data"] = df_jobs["Job Title"].astype(str) + " " + \
                       df_jobs["Skills Required"].astype(str) + " " + \
                       df_jobs["Experience Required"].astype(str) + " " + \
                       df_jobs["Job Location"].astype(str) + " " + \
                       df_jobs["Company Name"].astype(str)

X_text = df_jobs["text_data"]
y_labels = df_jobs["Job Title"].astype("category")

# Encode labels
y = y_labels.cat.codes
label_mapping = dict(enumerate(y_labels.cat.categories))
print("\n📘 Job Label Mapping:")
print(label_mapping)

# ----------------------------
# STEP 2: FEATURE EXTRACTION
# ----------------------------

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = vectorizer.fit_transform(X_text).toarray()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# STEP 3: BUILD MODEL
# ----------------------------

model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(len(set(y)), activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Compute class weights for imbalanced data
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

print("\n🚀 Training model...")
history = model.fit(
    X_train, y_train,
    epochs=40,
    batch_size=16,
    validation_data=(X_test, y_test),
    class_weight=class_weights,
    verbose=1
)

# ----------------------------
# STEP 4: EVALUATION
# ----------------------------

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Model Evaluation Results:")
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Predictions
y_pred = model.predict(X_test).argmax(axis=1)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=y_labels.cat.categories))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=y_labels.cat.categories,
            yticklabels=y_labels.cat.categories)
plt.title("Confusion Matrix for Job Prediction")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

# Plot training history
plt.figure(figsize=(8, 4))
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Model Accuracy per Epoch")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Model Loss per Epoch")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# ----------------------------
# STEP 5: SAVE MODEL & VECTORIZER
# ----------------------------

model.save("job_recommendation_model.h5")
pd.Series(label_mapping).to_csv("label_mapping.csv")
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\n✅ Model saved as 'job_recommendation_model.h5'")
print("✅ Label mapping saved as 'label_mapping.csv'")
print("✅ TF-IDF vectorizer saved as 'vectorizer.pkl'")

# ----------------------------
# STEP 6: RESUME SKILL EXTRACTION
# ----------------------------

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
    return text

def extract_skills(text):
    common_skills = [
        "python", "java", "c++", "excel", "sql", "machine learning",
        "data analysis", "communication", "leadership", "teamwork",
        "project management", "html", "css", "javascript", "power bi",
        "pandas", "numpy", "ai", "nlp", "deep learning", "react", "flask", "django"
    ]
    found = []
    text_lower = text.lower()
    for skill in common_skills:
        if re.search(r"\b" + re.escape(skill) + r"\b", text_lower):
            found.append(skill)
    return list(set(found))

RESUME_PATH = "Resume.pdf"

if os.path.exists(RESUME_PATH):
    resume_text = extract_text_from_pdf(RESUME_PATH)
    extracted_skills = extract_skills(resume_text)
    print("\n🧠 Extracted Skills from Resume:")
    print(extracted_skills)
else:
    print("\n⚠️ No Resume.pdf found. Skipping resume extraction.")
    extracted_skills = []

# ----------------------------
# STEP 7: PREDICT BEST JOB
# ----------------------------

if extracted_skills:
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    resume_text_combined = " ".join(extracted_skills)
    X_resume = vectorizer.transform([resume_text_combined]).toarray()

    model = load_model("job_recommendation_model.h5")
    pred = model.predict(X_resume)
    job_index = pred.argmax()

    print(f"\n🏆 Predicted Best Job Match: {label_mapping[job_index]}")
else:
    print("\n⚠️ No skills extracted to predict job match.")
