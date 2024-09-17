# Sentiment-Analysis-on-Healthcare-Reviews
**Project Overview**
This project focuses on performing sentiment analysis on healthcare-related reviews. The goal is to automatically categorize the sentiment of reviews (positive, neutral, or negative) to gain insights into patient feedback and healthcare service quality.
Sentiment analysis can be a valuable tool for healthcare providers, helping them understand patient satisfaction, identify areas for improvement, and ultimately enhance the quality of care.

**In this project, we:**
1.Preprocess textual reviews
2.Extract features using TF-IDF
3.Train machine learning models to classify sentiments
4.Visualize the results
**Data**
The dataset used for this project contains healthcare reviews from various sources such as hospitals, clinics, or medical practices. Each review is accompanied by a sentiment label that indicates whether the review is positive, neutral, or negative.
**Features**
1.Text Preprocessing: Tokenization, stopword removal, and TF-IDF vectorization.
2.Classification Models: Support Vector Machines (SVM), Naive Bayes, and more.
3.Evaluation Metrics: Accuracy, Precision, Recall, F1 Score, Confusion Matrix, and ROC-AUC.
4.Visualization: Plot results using matplotlib and seaborn.

**REQUIRED LIBRARIES**
import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
nltk.download('punkt')
nltk.download('stopwords')

**Preprocessing**
The textual data is preprocessed using the following steps:

Lowercasing: Convert all text to lowercase.
Tokenization: Split the text into individual words.
Stopword Removal: Remove common words like "the", "is", "and", etc., using NLTK's stopword list.
TF-IDF Vectorization: Transform the text data into numerical features using TfidfVectorizer.

# Sample code for text preprocessing
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Remove special characters
    text = re.sub(r'\W', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize the text
    words = word_tokenize(text)
    # Remove stopwords
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

**Modeling**
1.Feature Extraction
We use the TF-IDF technique to transform text into feature vectors:

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, stop_words='english')
X_tfidf = tfidf.fit_transform(data['cleaned_review'])

2.Model Training
We train a SVM, but the project can be extended to other models such as Random Forest or Logistic Regression.

3.Hyperparameter Tuning
We use GridSearchCV to tune the hyperparameters of the models to achieve the best performance.

**Evaluation**
We evaluate the model's performance using the following metrics:

1.Accuracy
2.Precision
3.Recall
4.F1-Score
5.ROC-AUC
6.Confusion Matrix

from sklearn.metrics import classification_report, confusion_matrix

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

**Visualization**
We use seaborn and matplotlib to visualize the classification results.

**Confusion Matrix**
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
