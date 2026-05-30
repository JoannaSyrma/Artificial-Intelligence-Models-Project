from statistics import mode
import numpy as np
import math
import csv
import os
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report


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

class Node:
    def __init__(self, checking_feature=None, is_leaf=False, category=None):
        self.checking_feature = checking_feature
        self.left_child = None
        self.right_child = None
        self.is_leaf = is_leaf
        self.category = category
        


class ID3:
    def __init__(self, features):
        self.tree = None
        self.features = features
    
    def fit(self, x, y):
        '''
        creates the tree
        '''
        most_common = mode(y.flatten())
        self.tree = self.create_tree(x, y, features=np.arange(len(self.features)), category=most_common)
        return self.tree
    
    def create_tree(self, x_train, y_train, features, category):
        
        
        # check empty data
        if len(x_train) == 0:
            return Node(checking_feature=None, is_leaf=True, category=category)  # decision node
        
        # check all examples belonging in one category
        if np.all(y_train.flatten() == 0):
            return Node(checking_feature=None, is_leaf=True, category=0)
        elif np.all(y_train.flatten() == 1):
            return Node(checking_feature=None, is_leaf=True, category=1)
        
        if len(features) == 0:
            return Node(checking_feature=None, is_leaf=True, category=mode(y_train.flatten()))
        
        igs = list()
        for feat_index in features.flatten():
            igs.append(self.calculate_ig(y_train.flatten(), [example[feat_index] for example in x_train]))
        
        max_ig_idx = np.argmax(np.array(igs).flatten())
        m = mode(y_train.flatten())  # most common category 

        root = Node(checking_feature=max_ig_idx,category=m)

        # data subset with X = 0
        x_train_0 = x_train[x_train[:, max_ig_idx] == 0, :]
        y_train_0 = y_train[x_train[:, max_ig_idx] == 0].flatten()

        # data subset with X = 1
        x_train_1 = x_train[x_train[:, max_ig_idx] == 1, :]
        y_train_1 = y_train[x_train[:, max_ig_idx] == 1].flatten()

        new_features_indices = np.delete(features.flatten(), max_ig_idx)  # remove current feature

        root.left_child = self.create_tree(x_train=x_train_1, y_train=y_train_1, features=new_features_indices, 
                                           category=m)  # go left for X = 1
        
        root.right_child = self.create_tree(x_train=x_train_0, y_train=y_train_0, features=new_features_indices,
                                            category=m)  # go right for X = 0
        
        return root


    @staticmethod
    def calculate_ig(classes_vector, feature):
        classes = set(classes_vector)

        HC = 0
        for c in classes:
            PC = list(classes_vector).count(c) / len(classes_vector)  # P(C=c)
            HC += - PC * math.log(PC, 2)  # H(C)
            # print('Overall Entropy:', HC)  # entropy for C variable
            
        feature_values = set(feature)  # 0 or 1 in this example
        HC_feature = 0
        for value in feature_values:
            # pf --> P(X=x)
            pf = list(feature).count(value) / len(feature)  # count occurences of value 
            indices = [i for i in range(len(feature)) if feature[i] == value]  # rows (examples) that have X=x

            classes_of_feat = [classes_vector[i] for i in indices]  # category of examples listed in indices above
            for c in classes: 
                # pcf --> P(C=c|X=x)
                pcf = classes_of_feat.count(c) / len(classes_of_feat)  # given X=x, count C
                if pcf != 0: 
                    # - P(X=x) * P(C=c|X=x) * log2(P(C=c|X=x))
                    temp_H = - pf * pcf * math.log(pcf, 2)
                    # sum for all values of C (class) and X (values of specific feature)
                    HC_feature += temp_H
        
        ig = HC - HC_feature
        return ig    

        

    def predict(self, x):
        predicted_classes = list()

        for unlabeled in x.toarray():  # for every example 
            tmp = self.tree  # begin at root
            while not tmp.is_leaf:
                if unlabeled.flatten()[tmp.checking_feature] == 1:
                    tmp = tmp.left_child
                else:
                    tmp = tmp.right_child
            
            predicted_classes.append(tmp.category)
        
        return np.array(predicted_classes)

class RandomForest:
    def __init__(self,n_trees, features):
        self.n_trees = n_trees
        self.features = features
        self.trees= []
    
    def fit(self, x_train, y_train):
            x_dense= x_train.toarray()
            y_dense= np.array(y_train)

            for i in range(self.n_trees):
                #Choose a random subset of features for each tree-every feature is unique
                selected_features =np.random.choice(self.features,size=int(4),replace=False)
                #Train a tree on a selected subset
                
                self.trees.append(ID3(selected_features))
                self.trees[i].fit(x_dense,y_dense)

    def predict(self, x_data):
        #np.zeros:create an array of zeros(number of examples,number of decision trees)
        predictions=[]

        #Make predictions with each tree
        for tree in self.trees:#Iterates over each tree
            tree_predictions=tree.predict(x_data)
            predictions.append(tree_predictions) # Generate predictions for all examples in dataset x

        predictions=np.array(predictions)
        #the prediction array contains the predictions of each tree in a separate column
        final_predictions=[]
        for i in range(x_data.shape[0]):
            votes=predictions[:,i] # get predictions for the i-th example from all trees
            if np.sum(votes) > self.n_trees / 2 :
                final_predictions.append(1)
            elif np.sum(votes) ==self.n_trees / 2:
                final_predictions.append(1 if np.random.random() > 0.5 else 0)#dialegei tyxaia
            else:
                final_predictions.append(0)
            
        #Aggregate predictions using majority voting
       
        
        return final_predictions        

start_time = time.time()
print('start')

csv_file_path = 'C:\\vocab.csv'

voc = []

with open(csv_file_path, 'r', newline='') as file:
    csv_reader = csv.reader(file)
    
    for row in csv_reader:
        voc.append(row[0])

folder_path = 'C:\\aclImdb\\train'
train_reviews,train_labels=read_folder(folder_path)
print('train') 
 
        
vectorizer = CountVectorizer(vocabulary=voc, binary=True)
x_train = vectorizer.fit_transform(train_reviews)
y_train = np.array(train_labels).astype(int)


n_trees = 100
random_forest = RandomForest(n_trees=n_trees, features=voc)

random_forest.fit(x_train, y_train)

# Load and preprocess test data
test_folder_path = 'C:\\aclImdb\\test'
test_reviews,test_labels=read_folder(test_folder_path)
print('test')

# Vectorize test data
x_test = vectorizer.transform(test_reviews)
y_test = np.array(test_labels).astype(int)


# Make predictions on train data
train_predictions = random_forest.predict(x_train)
print(classification_report(y_train,train_predictions,zero_division=1))

# Make predictions on test data
test_predictions =random_forest.predict(x_test)
print(classification_report(y_test, test_predictions,zero_division=1))

#prints execution time of the whole program
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution_time: {execution_time} seconds")