import numpy as np
import pandas as pd
import os
import tweepy
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def load_data(bearer_token, query, count=100):
    """
    Loads tweets from the Twitter API v2.
    """
    client = tweepy.Client(bearer_token=bearer_token)


    response = client.search_recent_tweets(query=query, max_results=count, tweet_fields=["text", "created_at"])

    if response.data:
        tweets = [tweet.text for tweet in response.data]
        return tweets
    else:
        return []


def clean_data(data):
    """
    Cleans the data.
    """
    cleaned_data = []
    for tweet in data:
        tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
        tweet = re.sub(r"@\w+", "", tweet)
        tweet = re.sub(r"#\w+", "", tweet)
        tweet = re.sub(r"[^A-Za-z0-9 ]+", "", tweet)
        tweet = tweet.strip()

        tweet = tweet.lower()
        cleaned_data.append(tweet)
    return cleaned_data


def preprocess_data(data):
    """
    Preprocesses the data.
    """

    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X = vectorizer.fit_transform(data)

    return X, vectorizer


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Splits the data into training and testing sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test