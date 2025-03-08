from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import ssl

# Download necessary NLTK data
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("vader_lexicon")

# Handle SSL certificate verification issues
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Raw Review Text
review_text = "The movie was absolutely fantastic! The plot and acting were top-notch."

# Step 1: Tokenization
word_tokens = word_tokenize(review_text)
print(f"Tokens: {word_tokens}")

# Step 2: Stop word removal
stop_words = set(stopwords.words("english"))
print(f"\nStop Words from English: {stop_words}")

filtered_tokens = [word for word in word_tokens if word.lower() not in stop_words]
print("\nFiltered Tokens (after stop-word removal):", filtered_tokens)

# Step 3: TF-IDF weighting
corpus = [
    "The movie was absolutely fantastic! The plot and acting were top-notch.",
    "The movie was okay, but the acting could have been better.",
    "I did not like the movie. The plot was too slow."
]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)
feature_names = vectorizer.get_feature_names_out()

tfidf_dict = {feature_names[i]: tfidf_matrix[0, i] for i in range(len(feature_names))}
print("\nTF-IDF Weight for the first review:", tfidf_dict)

# Step 4: Sentiment Detection
s_analyser = SentimentIntensityAnalyzer()
sentiment_score = s_analyser.polarity_scores(review_text)
print("\nSentiment Score:", sentiment_score)

# Final Statement
if sentiment_score['compound'] > 0:
    sentiment = "Positive"
elif sentiment_score['compound'] < 0:
    sentiment = "Negative"
else:
    sentiment = "Neutral"

print("Overall Sentiment:", sentiment)
