import numpy as np
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL']='1' 
import tensorflow as tf
from tqdm import tqdm
from tensorflow import keras
from keras import utils
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

np.seterr(over='ignore')

# create tables & curves
def plot_results(training_accuracies, training_losses, testing_accuracies, testing_losses, testing_precision, testing_recall,testing_f1):

    # learning curve - accuracy TRAIN DATA
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    x_values = [((i+1) * 5000)for i in range(len(training_accuracies))]
    plt.plot(x_values, training_accuracies, label =None)
    plt.title('Training Accuracies')
    plt.xlabel('Training Data Size')
    plt.ylabel('Accuracy')


    # Loss curve TRAIN DATA
    plt.subplot(1, 2, 2)
    x_valuess = [((i+1)*2)for i in range(len(training_losses))]
    plt.plot(x_valuess,training_losses, label =None)
    plt.title('Training Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.show()
    
    # Curves: accuracy, loss, precision, recall, f1 TEST DATA
    plt.figure(figsize=(15, 10))

    metrics_names = ['Accuracy', 'Loss', 'Precision', 'Recall', 'F1']
    metrics_values = [testing_accuracies, testing_losses,testing_precision, testing_recall,testing_f1]

    for i, metric in enumerate(metrics_names):
        plt.subplot(2, 3, i+1)
        plt.plot(np.arange(1, len(metrics_values[i]) + 1), metrics_values[i], label =None)
        plt.title(f'Testing {metric}')
        plt.xlabel('Experiment')
        plt.ylabel(metric)

    plt.tight_layout()
    plt.show()

# Main execution
start_time = time.time()
warnings.filterwarnings("ignore")
#lists for results
training_accuracies = []
testing_accuracies = []
testing_losses = []
training_losses = []
testing_precision=[]
testing_recall=[]
testing_f1=[]
#number of epochs 
e=2
#loop for 5 experiments 
for k in range(0,5):

    (x_train_imdb, y_train_imdb), (x_test_imdb, y_test_imdb) = tf.keras.datasets.imdb.load_data()

    #edit size of training data
    start1, end1 = 0, 2500+(k*2500)
    start2, end2 = 12500, 15000+(k*2500)
    x_train_imdb= np.concatenate((x_train_imdb[start1:end1], x_train_imdb[start2:end2]))
    y_train_imdb = np.concatenate((y_train_imdb[start1:end1], y_train_imdb[start2:end2]))


    word_index = tf.keras.datasets.imdb.get_word_index()
    index2word = dict((i + 3, word) for (word, i) in word_index.items())
    index2word[0] = '[pad]'
    index2word[1] = '[bos]'
    index2word[2] = '[oov]'
    x_train_imdb = [' '.join([index2word[idx] for idx in text]) for text in x_train_imdb]
    x_test_imdb = [' '.join([index2word[idx] for idx in text]) for text in x_test_imdb]

    train_doc_length = 0
    for doc in tqdm(x_train_imdb):
        tokens = str(doc).split()
        train_doc_length += len(tokens)

    print('\nTraining data average document length =', (train_doc_length / len(x_train_imdb)))

    VOCAB_SIZE = 100000
    SEQ_MAX_LENGTH = 250
    vectorizer = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE,output_mode='int',ngrams=1, name='vector_text',output_sequence_length=SEQ_MAX_LENGTH)
    
    with tf.device('/CPU:0'):
        vectorizer.adapt(x_train_imdb)

    vector_model = tf.keras.models.Sequential([tf.keras.layers.Input(shape=(1,), dtype=tf.string),vectorizer])
    vector_model.predict([['awesome movie']])

    dummy_embeddings = tf.keras.layers.Embedding(1000, 5)
    dummy_embeddings(tf.constant([1, 2, 3])).numpy()

    def get_rnn(num_layers=1, emb_size=64, h_size=64):
        inputs = tf.keras.layers.Input(shape=(1,), dtype=tf.string, name='txt_input') # ['awesome movie']
        x = vectorizer(inputs) # [1189, 18, 0, 0, 0, 0, ...]
        x = tf.keras.layers.Embedding(input_dim=len(vectorizer.get_vocabulary()),output_dim=emb_size, name='word_embeddings',mask_zero=True)(x)
        for n in range(num_layers):
            if n != num_layers - 1:
                x = tf.keras.layers.SimpleRNN(units=h_size, name=f'rnn_cell_{n}', return_sequences=True)(x)
            else:
                x = tf.keras.layers.SimpleRNN(units=h_size, name=f'rnn_cell_{n}')(x)

        x = tf.keras.layers.Dropout(rate=0.5)(x)
        o = tf.keras.layers.Dense(units=1, activation='sigmoid', name='lr')(x)
        return tf.keras.models.Model(inputs=inputs, outputs=o, name='simple_rnn')

    imdb_rnn = get_rnn()

    imdb_rnn.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer=tf.keras.optimizers.SGD(),metrics=['binary_accuracy'])
    
    #train model for e epochs each time
    history = imdb_rnn.fit(x=np.array(x_train_imdb), y=np.array(y_train_imdb),epochs=e, verbose=1, batch_size=32) #thelei akeraious oxi string !!!
    
    #test results
    testloss, testaccuracy = imdb_rnn.evaluate(np.array(x_test_imdb),np.array(y_test_imdb))
    testing_accuracies.append(testaccuracy)
    testing_losses.append(testloss)
    # train results 
    training_accuracies.append(np.mean(history.history['binary_accuracy']))
    training_losses.append(np.mean(history.history['loss']))
    #print results for train data
    print('Training Data Experiment No', k+1, ' for', e,'epochs:')
    print('Accuracy(average):',round(training_accuracies[-1],2))
    print('Loss(average):',round(training_losses[-1],2))

    #save the true labels and the predictions for test data
    y_true = np.array(y_test_imdb)
    y_pred = np.array(imdb_rnn.predict(np.array(x_test_imdb))).squeeze()

    # print test 
    threshold=0.5
    y_pred_binary = (y_pred > threshold).astype(int)

    #print results for test data
    print('Testing Data Experiment No', k+1,':\n', classification_report(np.array(y_true),y_pred_binary,zero_division=1))
    #save results 
    testing_precision.append(precision_score(y_true, y_pred_binary))
    testing_recall.append(recall_score(y_true, y_pred_binary))
    testing_f1.append(f1_score(y_true, y_pred_binary))
    e += 2

    
plot_results(training_accuracies, training_losses, testing_accuracies, testing_losses, testing_precision, testing_recall, testing_f1)

#prints execution time of the whole program
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution_time: {execution_time} seconds")