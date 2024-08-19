import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


TEST_SIZE = 0.2


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).
    """
    evidence = []
    labels = []

    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)

        for row in reader:
            evidence_row = [
                int(row[0]),  # Administrative
                float(row[1]),  # Administrative_Duration
                int(row[2]),  # Informational
                float(row[3]),  # Informational_Duration
                int(row[4]),  # ProductRelated
                float(row[5]),  # ProductRelated_Duration
                float(row[6]),  # BounceRates
                float(row[7]),  # ExitRates
                float(row[8]),  # PageValues
                float(row[9]),  # SpecialDay
                month_to_int(row[10]),  # Month (convert to integer)
                int(row[11]),  # OperatingSystems
                int(row[12]),  # Browser
                int(row[13]),  # Region
                int(row[14]),  # TrafficType
                1 if row[15] == "Returning_Visitor" else 0,  # VisitorType
                1 if row[16] == "TRUE" else 0  # Weekend
            ]

            label = 1 if row[17] == "TRUE" else 0

            evidence.append(evidence_row)
            labels.append(label)
    return evidence, labels


def month_to_int(month):
    """
    Convert a full or abbreviated month name to an integer.
    """
    months = {
        "January": 0, "February": 1, "March": 2, "April": 3, "May": 4, "June": 5,
        "July": 6, "August": 7, "September": 8, "October": 9, "November": 10, "December": 11,
        "Jan": 0, "Feb": 1, "Mar": 2, "Apr": 3, "May": 4, "Jun": 5,
        "Jul": 6, "Aug": 7, "Sep": 8, "Oct": 9, "Nov": 10, "Dec": 11
    }
    return months[month]


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)

    model.fit(evidence, labels)

    return model
def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """

    true_positive = 0
    true_negative = 0
    total_positive = 0
    total_negative = 0

    for label, prediction in zip(labels, predictions):
        if label == 1:
            total_positive += 1

            if prediction == 1:
                true_positive += 1

        elif label == 0:
            total_negative += 1

            if prediction == 0:
                true_negative += 1

    sensitivity = true_positive / total_positive
    specificity = true_negative / total_negative

    return (sensitivity, specificity)


if __name__ == "__main__":
    main()
