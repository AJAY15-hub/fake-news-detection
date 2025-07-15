# 📰 Fake News Detection using Machine Learning and NLP

This project detects whether a news article is **FAKE** or **REAL** using a machine learning pipeline with NLP techniques.

---

## 🚀 Project Summary

- ✅ Built with Python, Scikit-learn, Pandas, and TfidfVectorizer.
- 🧠 Model: **Logistic Regression**
- 📊 Achieved **98.35% accuracy**
- 📈 Evaluated with Confusion Matrix and Classification Report

---

## 🧰 Tech Stack

- Python 🐍
- Pandas & NumPy
- Scikit-learn
- TfidfVectorizer (NLP)
- Logistic Regression

---

## 📁 Dataset

- [Kaggle: Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

It contains:
- `Fake.csv` – Fake news articles
- `True.csv` – Real news articles

---

## 📸 Sample Output--->

Accuracy: 0.9835

Confusion Matrix:
[[4529 91]
[ 57 4303]]

Classification Report:
precision recall f1-score support
0 0.99 0.98 0.98 4620
1 0.98 0.99 0.98 4360

---

## 🔮 Example Prediction

```python
News: NASA discovers water on Mars in major breakthrough.
Prediction: FAKE
