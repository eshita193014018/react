from flask import Flask, render_template, request
import pickle
import re
from tweepy import OAuthHandler, API, TweepyException
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import os
from dotenv import load_dotenv  # Import for .env file

# Load environment variables
load_dotenv()

# Download required NLTK data (do this only once, not every run in production)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

app = Flask(__name__)
            

# Load sentiment analysis model
sentiment_model, tfidf_vectorizer = None, None

model_path = os.path.join(os.path.dirname(__file__), 'ensemble_sentiment_model.pkl')
vectorizer_path = os.path.join(os.path.dirname(__file__), 'tfidf_vectorizer.pkl')

try:
    with open(model_path, 'rb') as model_file:
        sentiment_model = pickle.load(model_file)
    print("✅ Ensemble sentiment model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading sentiment model: {e}")

try:
    with open(vectorizer_path, 'rb') as vectorizer_file:
        tfidf_vectorizer = pickle.load(vectorizer_file)
    print("✅ TF-IDF vectorizer loaded successfully.")
except Exception as e:
    print(f"❌ Error loading TF-IDF vectorizer: {e}")


def preprocess_text(text):
    """Clean and preprocess text before prediction."""
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)


def predict_sentiment(text):
    """Predict sentiment using trained model."""
    if sentiment_model is None or tfidf_vectorizer is None:
        return "Error: Sentiment model/vectorizer not loaded."

    processed_text = preprocess_text(text)
    text_tfidf = tfidf_vectorizer.transform([processed_text])
    prediction = sentiment_model.predict(text_tfidf)[0]

    if prediction == 0:
        return "Negative"
    elif prediction == 1:
        return "Neutral"
    else:
        return "Positive"


class TwitterClient:
    """Twitter API client."""
    def __init__(self):
        consumer_key = os.getenv("CONSUMER_KEY")
        consumer_secret = os.getenv("CONSUMER_SECRET")
        access_token = os.getenv("ACCESS_TOKEN")
        access_token_secret = os.getenv("ACCESS_TOKEN_SECRET")

        if not all([consumer_key, consumer_secret, access_token, access_token_secret]):
            raise ValueError("❌ Missing Twitter API credentials in environment variables.")

        try:
            auth = OAuthHandler(consumer_key, consumer_secret)
            auth.set_access_token(access_token, access_token_secret)
            self.api = API(auth, wait_on_rate_limit=True)
        except TweepyException as e:
            print(f"❌ Twitter Authentication Failed: {e}")
            self.api = None

    def get_tweets(self, query, maxTweets=100):
        """Fetch tweets from Twitter API."""
        if self.api is None:
            print("❌ Twitter API not initialized.")
            return pd.DataFrame()

        try:
            fetched_tweets = self.api.search_tweets(q=query, count=maxTweets, tweet_mode="extended", lang="en")
            tweets = [{"tweets": tweet.full_text} for tweet in fetched_tweets]
            return pd.DataFrame(tweets)
        except Exception as e:
            print(f"❌ Tweepy error: {e}")
            return pd.DataFrame()


def con1(sentence):
    """Simple emotion analysis based on word list from emotions.txt."""
    emotion_list = []
    words = sentence.split()

    try:
        with open("emotions.txt", "r") as file:
            for line in file:
                clear_line = line.strip().replace("\n", "").replace(",", "").replace("'", "")
                if ":" in clear_line:
                    word, emotion = clear_line.split(":")
                    if word in words:
                        emotion_list.append(emotion.strip().lower())
    except FileNotFoundError:
        print("⚠️ emotions.txt not found.")
        return []

    return emotion_list


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/hello")
def hello():
    return "Hello, Flask is working!"


@app.route("/predict", methods=["POST"])
def predict():
    """Main prediction route."""
    if request.method == "POST":
        comment = request.form["Tweet"]
        twitter_client = TwitterClient()
        tweets_df = twitter_client.get_tweets(comment, maxTweets=100)

        if tweets_df.empty:
            return render_template(
                "result.html",
                outputs={},
                NU=0, N=0, P=0,
                happy=0, sad=0, angry=0, loved=0,
                powerless=0, surprise=0, fearless=0,
                cheated=0, attracted=0, singledout=0, anxious=0,
                error_message="No tweets found or error fetching tweets."
            )
        

        predicted_sentiments, cleaned_tweets, original_tweets = [], [], []

        for tweet_data in tweets_df["tweets"]:
            original_tweets.append(tweet_data)
            cleaned_tweet = preprocess_text(tweet_data)
            cleaned_tweets.append(cleaned_tweet)
            predicted_sentiments.append(predict_sentiment(tweet_data))

        output = dict(zip(original_tweets, predicted_sentiments))

        # Count sentiments
        Neucount = predicted_sentiments.count("Neutral")
        Negcount = predicted_sentiments.count("Negative")
        Poscount = predicted_sentiments.count("Positive")

        # Emotion analysis
        all_cleaned_text = " ".join(cleaned_tweets)
        emo = con1(all_cleaned_text)

        h = emo.count("happy")
        s = emo.count("sad")
        a = emo.count("angry")
        l = emo.count("loved")
        pl = emo.count("powerless")
        su = emo.count("surprise")
        fl = emo.count("fearless")
        c = emo.count("cheated")
        at = emo.count("attracted")
        so = emo.count("singledout")  # ✅ removed space, match with emotions.txt
        ax = emo.count("anxious")

        return render_template(
            "result.html",
            outputs=output,
            NU=Neucount, N=Negcount, P=Poscount,
            happy=h, sad=s, angry=a, loved=l,
            powerless=pl, surprise=su, fearless=fl,
            cheated=c, attracted=at, singledout=so, anxious=ax
        )

    return render_template("index.html")
    

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host="0.0.0.0")
        