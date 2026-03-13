import numpy as np
import os, csv
import time
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score,accuracy_score, classification_report

#visualization of data
def plot_metrics(train_size, train_metric, test_metric, metric_name, train_label, test_label, train_color, test_color):
    plt.plot(train_size, train_metric, label=train_label, color=train_color)
    plt.plot(train_size, test_metric, label=test_label, color=test_color)
    plt.xlabel('Training Size')
    plt.ylabel(metric_name)
    plt.legend()
    plt.show()

def bayes(x_train, y_train,new_reviews, new_labels, train_size, iteration,accuracy_train,accuracy_test,precision_train,precision_test,recall_train,recall_test ,f1_train ,f1_test):
    # train Bayes model
    nb = GaussianNB()
    nb.fit(x_train, y_train)

    #predictions for train data
    train_predictions = nb.predict(x_train)
    #calculate accuracy, precision, recall, f1 for train data
    accuracy_train.append(accuracy_score(y_train,train_predictions))
    train_classification_report = classification_report(y_train, train_predictions,zero_division=1)
    precision_train.append(precision_score(y_train, train_predictions))
    recall_train.append(recall_score(y_train, train_predictions))
    f1_train.append(f1_score(y_train, train_predictions))
    
    #predictions for test data
    test_predictions = nb.predict(new_reviews)
    #calculate accuracy, precision, recall, f1 for test data
    accuracy_test.append(accuracy_score(new_labels,test_predictions))
    test_classification_report = classification_report(new_labels,test_predictions,zero_division=1)
    precision_test.append(precision_score(new_labels,test_predictions))
    recall_test.append(recall_score(new_labels,test_predictions))
    f1_test.append(f1_score(new_labels,test_predictions))

    print('BAYES:')
    print("Training Classification Report ", iteration +1, " :")
    print(train_classification_report) 
    print("Test Classification Report ", iteration +1, " :")
    print(test_classification_report)

    if (iteration==4):
        plot_metrics(train_size, accuracy_train, accuracy_test, 'Accuracy', 'Training Accuracy', 'Test Accuracy', 'blue', 'orange')
        plot_metrics(train_size, precision_train, precision_test, 'Precision', 'Training Precision', 'Test Precision', 'green', 'red')
        plot_metrics(train_size, recall_train, recall_test, 'Recall', 'Training Recall', 'Test Recall', 'purple', 'brown')
        plot_metrics(train_size, f1_train, f1_test, 'F1', 'Training F1', 'Test F1', 'cyan', 'magenta')

    return accuracy_train,accuracy_test,precision_train,precision_test,recall_train,recall_test ,f1_train ,f1_test 

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

#Main execution
start_time = time.time()
print('start')
np.seterr(over='ignore')# defines the path of the csv file 
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


vectorizer = CountVectorizer(vocabulary=voc, binary=True)
#creates vector according to the vocabulary
new_reviews = vectorizer.transform(new_reviews).toarray()

train_size =[]
iterations=5
NB_accuracy_train = []
NB_accuracy_test = []
NB_precision_train = []
NB_precision_test = []
NB_recall_train = []
NB_recall_test = []
NB_f1_train = []
NB_f1_test = []

for iteration in range(iterations):
    #create different training data size for each experiment 
    start1, end1 = 0, 2500+(iteration*2500)
    start2, end2 = 12500, 15000+(iteration*2500)
    x_train= reviews[start1:end1] + reviews[start2:end2]
    y_train = labels[start1:end1] + labels[start2:end2]
    
    #creates vector according to the vocabulary
    x_train = vectorizer.transform(x_train).toarray()

    train_size.append(x_train.shape[0])
    print("Training data size:" , x_train.shape[0])
    #bayes train and 
    NB_accuracy_train,NB_accuracy_test,NB_precision_train,NB_precision_test,NB_recall_train,NB_recall_test ,NB_f1_train ,NB_f1_test = bayes(x_train, y_train,new_reviews, new_labels, train_size, iteration, NB_accuracy_train,NB_accuracy_test,NB_precision_train,NB_precision_test,NB_recall_train,NB_recall_test ,NB_f1_train ,NB_f1_test)

#prints execution time of the whole program
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution_time: {execution_time} seconds")