import numpy as np
import os, csv
import time
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score,accuracy_score, classification_report

def train_bayes(reviews,labels):
    nb = GaussianNB()
    nb.fit(reviews, labels)

    return nb 

def train_randomForest(reviews,labels):
    rf = RandomForestClassifier(criterion='entropy')
    rf.fit(reviews, labels)

    return rf 

def train_adaBoost(reviews,labels):
    ab = AdaBoostClassifier()
    ab.fit(reviews, labels)

    return ab

def evaluate(reviews, labels, new_reviews, new_labels,voc, iterations, algo):

    vectorizer = CountVectorizer(vocabulary=voc, binary=True)
    #creates vector according to the vocabulary
    new_reviews=vectorizer.transform(new_reviews).toarray()
    train_size =[]
    train_metrics={'accuracy':[],'precision':[],'recall':[],'f1':[]}
    test_metrics={'accuracy':[],'precision':[],'recall':[],'f1':[]}
    
        
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

        # train algo model
        if algo =='nb':
            m = train_bayes(x_train,y_train)
            name ='BAYES'
        elif algo == 'rf':
            m= train_randomForest(x_train,y_train)
            name='RANDOM FOREST'
        else:
            m=train_adaBoost(x_train,y_train)
            name='ADA BOOST'

        #predictions for train data
        train_predictions =m.predict(x_train)
        train_m =calculate_metrics(y_train,train_predictions)
        for metric, x in train_m.items():
            train_metrics[metric].append(x)

        test_predictions = m.predict(new_reviews)
        test_m =calculate_metrics(new_labels,test_predictions)
        for metric, x in test_m.items():
            test_metrics[metric].append(x)

        print(name)
        train_classification_report = classification_report(y_train, train_predictions,zero_division=1)
        print("Training Classification Report ", iteration +1, " :")
        print(train_classification_report) 
        test_classification_report = classification_report(new_labels,test_predictions,zero_division=1)
        print("Test Classification Report ", iteration +1, " :")
        print(test_classification_report)


    visualisation(train_size,train_metrics,test_metrics)
        
def calculate_metrics(y_data,y_pred):
    metrics={
        'accuracy':accuracy_score(y_data,y_pred),
        'precision':precision_score(y_data,y_pred),
        'recall':recall_score(y_data,y_pred),
        'f1': f1_score(y_data,y_pred)
    }
    return metrics

def visualisation(train_size,train_metrics,test_metrics):

        # Visualisation of learning curve
        plt.plot(train_size, train_metrics['accuracy'], label='Training Accuracy', color='blue')
        plt.plot(train_size, test_metrics['accuracy'], label='Test Accuracy', color='orange')
        plt.xlabel('Training Size')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

        # Visualisation of precision curve
        plt.plot(train_size, train_metrics['precision'], label='Training Precision',color='green')
        plt.plot(train_size, test_metrics['precision'], label='Test Precision', color='red')
        plt.xlabel('Training Size')
        plt.ylabel('Precision')
        plt.legend()
        plt.show()

        # Visualisation of recall curve
        plt.plot(train_size, train_metrics['recall'], label='Training Recall', color='purple')
        plt.plot(train_size, test_metrics['recall'], label='Test Recall', color='brown')
        plt.xlabel('Training Size')
        plt.ylabel('Recall')
        plt.legend()
        plt.show()

        # Visualisation of F1 curve
        plt.plot(train_size, train_metrics['f1'], label='Training F1',color='cyan')
        plt.plot(train_size, test_metrics['f1'], label='Test F1',color='magenta')
        plt.xlabel('Training Size')
        plt.ylabel('F1')
        plt.legend()
        plt.show()


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


start_time = time.time()
print('start')
np.seterr(over='ignore')

# defines the path of the csv file 
csv_file_path = 'C:\\vocab.csv'

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

evaluate(reviews, labels, new_reviews, new_labels, voc, iterations=5, algo ='nb')
evaluate(reviews, labels, new_reviews, new_labels, voc, iterations=5, algo ='rf')
evaluate(reviews, labels, new_reviews, new_labels, voc, iterations=5, algo ='ab')


#prints execution time of the whole program
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution_time: {execution_time} seconds")
