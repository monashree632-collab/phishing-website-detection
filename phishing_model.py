import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import pickle
import warnings

from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')

# Load Dataset
try:
    phish_data = pd.read_csv('phishing_site_urls.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: The file 'phishing_site_urls.csv' was not found.")
    exit()

# Display dataset info
print(phish_data.info())
print(phish_data.head())

# Check missing values
print("Missing values:\n", phish_data.isnull().sum())

# Tokenization
tokenizer = RegexpTokenizer(r'[A-Za-z]+')

print("Getting words tokenized ...")
t0 = time.perf_counter()

phish_data['text_tokenized'] = phish_data['URL'].map(lambda x: tokenizer.tokenize(str(x)))

t1 = time.perf_counter() - t0
print("Time taken:", t1, "sec")

# Convert tokens to string
phish_data['text_clean'] = phish_data['text_tokenized'].apply(lambda x: ' '.join(x))

# Feature Extraction
cv = CountVectorizer()
feature = cv.fit_transform(phish_data['text_clean'])

# Train Test Split
trainX, testX, trainY, testY = train_test_split(
    feature, phish_data['Label'], test_size=0.2, random_state=42
)

# Dictionary to store model scores
Scores_ml = {}

# Logistic Regression
lr = LogisticRegression()
lr.fit(trainX, trainY)

train_acc = lr.score(trainX, trainY)
test_acc = lr.score(testX, testY)

Scores_ml['Logistic Regression'] = np.round(test_acc, 2)

print("\nLogistic Regression")
print("Training Accuracy:", train_acc)
print("Testing Accuracy:", test_acc)

# Confusion Matrix
con_mat = confusion_matrix(testY, lr.predict(testX))

plt.figure(figsize=(6, 4))
sns.heatmap(pd.DataFrame(con_mat,
                         columns=['Predicted: Bad', 'Predicted: Good'],
                         index=['Actual: Bad', 'Actual: Good']),
            annot=True, fmt='d', cmap="YlGnBu")

plt.title("Confusion Matrix - Logistic Regression")
plt.show()

print("\nClassification Report\n")
print(classification_report(testY, lr.predict(testX)))

# Save Model
with open('phishing_model.pkl', 'wb') as f:
    pickle.dump(lr, f)

print("Model saved as 'phishing_model.pkl'")

# Class Distribution Visualization
label_counts = phish_data['Label'].value_counts().reset_index()
label_counts.columns = ['Label', 'Count']

sns.set_style('white')

plt.figure(figsize=(6, 4))
sns.barplot(x=label_counts['Label'], y=label_counts['Count'], palette='coolwarm')

plt.title("Distribution of Good vs Bad Sites")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# Multinomial Naive Bayes
mnb = MultinomialNB()
mnb.fit(trainX, trainY)

train_acc = mnb.score(trainX, trainY)
test_acc = mnb.score(testX, testY)

Scores_ml['MultinomialNB'] = np.round(test_acc, 2)

print("\nMultinomial Naive Bayes")
print("Training Accuracy:", train_acc)
print("Testing Accuracy:", test_acc)

print("\nClassification Report\n")
print(classification_report(testY, mnb.predict(testX), target_names=['Bad', 'Good']))

con_mat = confusion_matrix(testY, mnb.predict(testX))

plt.figure(figsize=(6, 4))
sns.heatmap(con_mat, annot=True, fmt='d', cmap="YlGnBu")
plt.title("Confusion Matrix - Naive Bayes")
plt.show()

# TF-IDF + Random Forest (Bi-gram)
vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(2, 2))

feature = vectorizer.fit_transform(phish_data['text_clean'])

trainX, testX, trainY, testY = train_test_split(
    feature, phish_data['Label'], test_size=0.2, random_state=42
)

clf = RandomForestClassifier()
clf.fit(trainX, trainY)

predY = clf.predict(testX)

ngram_acc = accuracy_score(testY, predY)

Scores_ml['Bi-gram'] = np.round(ngram_acc, 2)

print("\nBi-gram Random Forest Model")
print("Accuracy:", ngram_acc)

# Training accuracy
train_pred = clf.predict(trainX)
train_acc = accuracy_score(trainY, train_pred)

print("Training Accuracy:", train_acc)

# Testing accuracy
test_pred = clf.predict(testX)
test_acc = accuracy_score(testY, test_pred)

print("Testing Accuracy:", test_acc)

print("\nClassification Report\n")
print(classification_report(testY, test_pred, target_names=['Bad', 'Good']))

# Confusion Matrix
con_mat = confusion_matrix(testY, test_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(con_mat, annot=True, fmt='d', cmap="YlGnBu")
plt.title("Confusion Matrix - Random Forest")
plt.show()

# Model Score Summary
print("\nModel Accuracy Comparison:")
print(Scores_ml)