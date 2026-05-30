import math
import os
import csv
import time
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score,accuracy_score, classification_report

def read_folder(path):
    reviews = []
    labels = []

    for sentiment in ["neg", "pos"]:
        folder = os.path.join(path, sentiment)
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)

            # Read the content of the file
            with open(file_path, 'r', encoding='utf-8') as file:
                review_text = file.read()

            # Append the review to the list
            reviews.append(review_text)

            # Append the label (0 for "neg", 1 for "pos")
            labels.append(0 if sentiment == "neg" else 1)
    return reviews,labels

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

class AdaBoost:
    def __init__(self,m):
        self.m = m #arithmos ypothesewn pou dhmiourgountai
        self.h=[] #m ypotheseis pou mathainoume
        self.z=[]#varoi twn psifwn

    def fit(self,examples):
        n=len(examples) # number of examples
        w=np.ones(n) / n #Dianisma me ta varoi twn N paradeigmatwn ola 1/n
        
        for m in range(self.m):
            hm=DecisionTreeClassifier(max_depth=1)
            x_data=np.array([example[0] for example in examples])#features
            y_data=np.array([example[1] for example in examples])#svstes apanthseis
            hm.fit(x_data,y_data,sample_weight=w)# fit gia to decision tree

            error=0
            for j in range(n):
                if hm.predict([x_data[j]]) != y_data[j]:#check if to prediction diaferei apo to y_data
                    error= error + w[j]

            if error >= 0.5:
                break

            for j in range(n):
                if hm.predict([x_data[j]]) ==y_data[j]:
                    w[j]= w[j] * (error / (1 - error))

            w= w/(np.sum(w))#pali athroisma varwn 1
            self.h.append(hm) #Prosthetoume thn nea ypothesi pou mathame - adynamos taxinomiths

            self.z.append( 0.5 * np.log((1 - error) / error))

    def predict_adaBoost(self,x):#prediction for x_data
        result=[]
        x_data = np.array([example for example in x],dtype=int)
        for row in range(len(x_data)):
            sum0 = 0
            sum1 = 0
            index=0
            for h in self.h:
                prediction = h.predict([x_data[row]])
                if prediction == 0:
                    sum0 += self.z[index]
                else:
                    sum1 += self.z[index]
                index+=1
            result.append(0 if sum0>sum1 else 1)  

        return result
    
    def evaluate(self,ada_model, reviews,labels,new_reviews,new_labels,vocabulary,iterations):
        train_size =[]
        train_metrics={'accuracy':[],'precision':[],'recall':[],'f1':[]}
        test_metrics={'accuracy':[],'precision':[],'recall':[],'f1':[]}
        

        for iteration in range(iterations):

            #create different training data size for each experiment 
            start1, end1 = 0, 2500+(iteration*2500)
            start2, end2 = 12500, 15000+(iteration*2500)
            x_train= np.concatenate((reviews[start1:end1], reviews[start2:end2]))
            y_train = np.concatenate((labels[start1:end1], labels[start2:end2]))
            
            x_train = vectorizer.transform(x_train).toarray()
            training_data= [[x_train[i],y_train[i]] for i in range(len(x_train))] 
            ada_model.fit(training_data) #train the model


            #creates vector according to the vocabulary
            x_test = vectorizer.transform(new_reviews).toarray()

            train_size.append(len(x_train))
            print("Training data size:" , len(x_train))

            train_predictions = self.predict_adaBoost(x_train)
            train_m =calculate_metrics(y_train,train_predictions)
            for metric, x in train_m.items():
                train_metrics[metric].append(x)

            test_predictions = self.predict_adaBoost(x_test)
            test_m =calculate_metrics(new_labels,test_predictions)
            for metric, x in test_m.items():
                test_metrics[metric].append(x)
            
            #Print classification reports
            print("Training Classification Report:")
            print(classification_report(y_train,train_predictions,zero_division=1))
            print("Test Classification Report:")
            print(classification_report(new_labels, test_predictions,zero_division=1))

        return train_metrics,test_metrics,train_size

        
start_time = time.time()
print('start')

#Read the vocab file
csv_file_path = 'C:\\vocab.csv'

voc = []

with open(csv_file_path, 'r', newline='') as file:
    csv_reader = csv.reader(file)
    
    for row in csv_reader:
        voc.append(row[0])

#Read the train data
folder_path = 'C:\\aclImdb\\train'
train_reviews,train_labels=read_folder(folder_path)
print('train') 
 
#Vectorize train data      
vectorizer = CountVectorizer(vocabulary=voc, binary=True)

#create the model
M=100
ada_model = AdaBoost(M) 

#Read the test data
test_folder_path = 'C:\\aclImdb\\test'
test_reviews,test_labels=read_folder(test_folder_path)
print('test')

#Make predictions and metrics on train data
train_metrics , test_metrics,train_size = ada_model.evaluate(ada_model,train_reviews,train_labels,test_reviews,test_labels,voc,iterations=5)

#visualisation metrics
visualisation(train_size,train_metrics,test_metrics)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution_time: {execution_time} seconds")