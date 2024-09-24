from sklearn.metrics import accuracy_score


def evaluate_model(predictions, ground_truth):
    """
    Evaluate the model based on the generated predictions and the ground truth.
    This can include accuracy, F1 score, and other relevant metrics.
    """
    accuracy = accuracy_score(ground_truth, predictions)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    return accuracy


# Example usage
if __name__ == '__main__':
    # Dummy data for example
    predictions = ['The sky is blue.', 'Dogs are great.']
    ground_truth = ['The sky is blue.', 'Dogs are loyal.']

    evaluate_model(predictions, ground_truth)
