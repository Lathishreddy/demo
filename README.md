
import pandas as pd
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download NLTK resources
nltk.download('stopwords')

# Load dataset
# Replace 'fake_news_dataset.csv' with your file name
# Expected columns: 'text' and 'label'
df = pd.read_csv('fake_news_dataset.csv')

# Drop missing values
df = df[['text', 'label']].dropna()

# Text preprocessing
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[.*?]', '', text)
    text = re.sub(r'https?://S+|www.S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'
', ' ', text)
    text = re.sub(r'w*dw*', '', text)

    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]

    return " ".join(words)

# Apply cleaning
df['clean_text'] = df['text'].apply(clean_text)

# Features and labels
X = df['clean_text']
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert text into TF-IDF features
vectorizer = TfidfVectorizer(max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Predictions
y_pred = model.predict(X_test_tfidf)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("
Classification Report:
")
print(classification_report(y_test, y_pred))
print("
Confusion Matrix:
")
print(confusion_matrix(y_test, y_pred))

# Predict on new article
def predict_news(news_text):
    cleaned = clean_text(news_text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    return prediction

sample_news = "Government announces a new policy to improve education."
print("
Sample Prediction:", predict_news(sample_news))