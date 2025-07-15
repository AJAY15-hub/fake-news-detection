# --- Import Libraries ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- Load Dataset ---
# You can merge Fake.csv and True.csv or just use Fake.csv with labels
fake_df = pd.read_csv("Fake.csv")
real_df = pd.read_csv("True.csv")

# Add labels
fake_df["label"] = 0  # Fake news
real_df["label"] = 1  # Real news

# Combine datasets
data = pd.concat([fake_df, real_df], axis=0)
data = data[['text', 'label']]  # Keep only relevant columns

# --- Preprocessing ---
# Shuffle the data
data = data.sample(frac=1).reset_index(drop=True)

# Train-test split
X = data['text']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# --- Train Model ---
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# --- Evaluate Model ---
y_pred = model.predict(X_test_vec)

print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n📈 Classification Report:\n", classification_report(y_test, y_pred))

# --- Optional: Predict on Custom Input ---
def predict_news(text):
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)
    return "REAL" if prediction[0] == 1 else "FAKE"

# Example:
print("\n📰 Prediction Example:")
test_text = "NASA discovers water on Mars in major breakthrough."
print("News:", test_text)
print("Prediction:", predict_news(test_text))