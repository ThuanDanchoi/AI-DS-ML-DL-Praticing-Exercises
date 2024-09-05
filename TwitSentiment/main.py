import argparse
import pandas as pd
import matplotlib.pyplot as plt
from data_processing import load_data, clean_data, preprocess_data
from model_operations import build_model, train_model, save_model
from wordcloud import WordCloud


def main(args):

    tweets = load_data(args.bearer_token, args.query, args.count)
    cleaned_tweets = clean_data(tweets)


    X, vectorizer = preprocess_data(cleaned_tweets)

    input_shape = (X.shape[1], 1)
    model = build_model(input_shape, num_layers=args.num_layers, layer_type=args.layer_type, layer_size=args.layer_size,
                        dropout_rate=args.dropout_rate)
    y_train = pd.Series(
        [1 if "good" in tweet else 0 for tweet in cleaned_tweets])
    model = train_model(model, X, y_train, epochs=args.epochs, batch_size=args.batch_size)


    save_model(model, args.model_path)


    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(cleaned_tweets))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a sentiment analysis model using Twitter data.")
    parser.add_argument('--bearer_token', type=str, required=True, help="Bearer token for Twitter API v2")
    parser.add_argument('--query', type=str, required=True, help="Query to search for tweets")
    parser.add_argument('--count', type=int, default=100, help="Number of tweets to fetch")
    parser.add_argument('--model_path', type=str, required=True, help="Path to save the trained model")
    parser.add_argument('--num_layers', type=int, default=3, help="Number of layers in the RNN")
    parser.add_argument('--layer_type', type=str, default='LSTM', choices=['LSTM', 'GRU', 'RNN', 'BiLSTM', 'BiGRU'],
                        help="Type of RNN layer")
    parser.add_argument('--layer_size', type=int, default=50, help="Size of each RNN layer")
    parser.add_argument('--dropout_rate', type=float, default=0.2, help="Dropout rate")
    parser.add_argument('--epochs', type=int, default=25, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")

    args = parser.parse_args()
    main(args)
