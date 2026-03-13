import numpy as np
import csv, os, math
import time
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score,accuracy_score, classification_report

# tokenizes a review based on the given vocabulary
def tokenize(review, vocabulary):
    # Splits the review into individual words
    tokens = review.split()
    # Creates a binary vector where each element corresponds to a word in the vocabulary
    # The value is 1 if the word is present in the review, and 0 otherwise
    vector = [1 if word in tokens else 0 for word in vocabulary]
    return vector

# creates a matrix for a set of reviews based on the specified vocabulary
def create_feature_matrix(reviews, vocabulary):
    return np.array([tokenize(review, vocabulary) for review in reviews])


# trains the Naive Bayes model
def train_naive_bayes(reviews, labels, vocabulary):
    X = create_feature_matrix(reviews, vocabulary)

    # separates positive and negative reviews (periexoun diadika dianismata)
    i=0
    positive_reviews=[]
    negative_reviews =[]
    for review in reviews:
        if labels[i] ==1:
            positive_reviews.append(X[i])
        else:
            negative_reviews.append(X[i])
        i+=1
   
    # Probability calculation
    prior_positive = len(positive_reviews) / len(X)
    prior_negative = len(negative_reviews) / len(X)

    # totals sums for each word(how many times they exist in reviews) 
    positive_counts = np.sum(positive_reviews, axis=0)
    negative_counts = np.sum(negative_reviews, axis=0)

    # total number of words in positive and negative reviews (how many words used in each category)
    total_positive_words = np.sum(positive_counts)
    total_negative_words = np.sum(negative_counts)

    # creates vocabulary: keys=words from voc and values=positions of those words in the list
    vocabulary_dict = {word: i for i, word in enumerate(vocabulary)}

    return prior_positive, prior_negative, positive_counts, negative_counts, total_positive_words, total_negative_words, vocabulary_dict

# predicts the sentiment of a review using Naive Bayes
def predict_naive_bayes(review, prior_positive, prior_negative, positive_counts, negative_counts, total_positive_words, total_negative_words, vocabulary_dict):
    #edit review - creates vector
    review_vector = tokenize(review, list(vocabulary_dict.keys()))

    #change into decimal numbers
    positive_counts = positive_counts.astype(np.float64)
    negative_counts = negative_counts.astype(np.float64)
    review_vector = np.array(review_vector, dtype=np.float64)

    # Probability calculation according to the Naive Bayes' positive and negative model 
    positive_probability = np.prod((positive_counts+1)**review_vector) * prior_positive / total_positive_words
    negative_probability = np.prod((negative_counts+1)**review_vector) * prior_negative / total_negative_words

    # Total Probability
    positive_score = prior_positive * positive_probability
    negative_score = prior_negative * negative_probability

    # returns prediction according to the higher score
    return 1 if positive_score > negative_score else 0

# evaluates the Naive Bayes model
def evaluate_model(reviews, labels, new_reviews, new_labels,voc, iterations=1):
    train_size =[]
    accuracy_train = []
    accuracy_test = []
    precision_train = []
    precision_test = []
    recall_train = []
    recall_test = []
    f1_train = []
    f1_test = []

    for iteration in range(iterations):
        #create different training data size for each experiment 
        start1, end1 = 0, 2500+(iteration*2500)
        start2, end2 = 12500, 15000+(iteration*2500)
        x_train= reviews[start1:end1] + reviews[start2:end2]
        y_train = labels[start1:end1] + labels[start2:end2]
        
        train_size.append(len(x_train))
        print("Training data size:" ,len(x_train))
        #calculate the model parameters based on the new train set
        prior_positive, prior_negative, positive_counts, negative_counts, total_positive_words, total_negative_words, vocabulary_dict = train_naive_bayes(x_train, y_train, voc)
        
        #predictions for train data
        train_predictions = [predict_naive_bayes(review,prior_positive, prior_negative, positive_counts, negative_counts, total_positive_words, total_negative_words, vocabulary_dict) for review in x_train]
        #calculate accuracy, precision, recall, f1 for train data
        accuracy_train.append(accuracy_score(y_train, train_predictions))
        train_classification_report = classification_report(y_train, train_predictions,zero_division=1)
        precision_train.append(precision_score(y_train, train_predictions))
        recall_train.append(recall_score(y_train, train_predictions))
        f1_train.append(f1_score(y_train, train_predictions))
        
        #predictions for test data
        test_predictions = [predict_naive_bayes(review,prior_positive, prior_negative, positive_counts, negative_counts, total_positive_words, total_negative_words, vocabulary_dict) for review in new_reviews]
        #calculate accuracy for test data
        accuracy_test.append(accuracy_score(new_labels, test_predictions))
        test_classification_report = classification_report(new_labels, test_predictions,zero_division=1)
        precision_test.append(precision_score(new_labels, test_predictions))
        recall_test.append(recall_score(new_labels, test_predictions))
        f1_test.append(f1_score(new_labels, test_predictions))

        print("Training Classification Report ", iteration +1, " :")
        print(train_classification_report) 
        print("Test Classification Report ", iteration +1, " :")
        print(test_classification_report)


    # Visualisation of learning curve
    plt.plot(train_size, accuracy_train, label='Training Accuracy', color='blue')
    plt.plot(train_size, accuracy_test, label='Test Accuracy', color='orange')
    plt.xlabel('Training Size')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Visualisation of precision curve
    plt.plot(train_size, precision_train, label='Training Precision',color='green')
    plt.plot(train_size, precision_test, label='Test Precision', color='red')
    plt.xlabel('Training Size')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()

    # Visualisation of recall curve
    plt.plot(train_size, recall_train, label='Training Recall', color='purple')
    plt.plot(train_size, recall_test, label='Test Recall', color='brown')
    plt.xlabel('Training Size')
    plt.ylabel('Recall')
    plt.legend()
    plt.show()

    # Visualisation of F1 curve
    plt.plot(train_size, f1_train, label='Training F1',color='cyan')
    plt.plot(train_size, f1_test, label='Test F1',color='magenta')
    plt.xlabel('Training Size')
    plt.ylabel('F1')
    plt.legend()
    plt.show()

# reads reviews and labels from a given folder path
def read_folder(folder_path):
    reviews= []
    labels= []
    # Iterate through "neg" and "pos" folders
    for sentiment in ["neg", "pos"]:
        folder = os.path.join(folder_path, sentiment)
        # Iterate through files in the current folder
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            # Read the content of the file
            with open(file_path, 'r', encoding='utf-8') as file:
                review_text = file.read()
            # Append the review to the list
            reviews.append(review_text)
            # Append the label (0 for "neg", 1 for "pos")
            labels.append(0 if sentiment == "neg" else 1)

    return reviews, labels 

# Main execution
start_time = time.time()
print('start')
np.seterr(over='ignore')

# defines the path of the csv file 
csv_file_path = 'C:\\Users\\joann\\Desktop\\ai 2nd\\vocab.csv'

# creates list for the vocabulary
voc = []

# opens and reads csv file 
with open(csv_file_path, 'r', newline='') as file:
    csv_reader = csv.reader(file)
    
    # accessing the lines of the CSV file
    for row in csv_reader:
        # adds word to the vocabulary
        voc.append(row[0])

print('read voc')

# defines the path of the csv file for train
folder_path = 'C:\\aclImdb\\train'

# reads folders and creates lists for the train reviews and their labels
reviews , labels = read_folder(folder_path)

print('read neg pos')

# defines the path of the csv file for test
folder_path = 'C:\\aclImdb\\test'

#reads folders and creates lists for the test reviews and their labels
new_reviews, new_labels = read_folder(folder_path)

print('read new reviews')

#evaluation of the model
evaluate_model(reviews, labels, new_reviews, new_labels, voc, iterations=5)

#prints execution time of the whole program
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution_time: {execution_time} seconds")

