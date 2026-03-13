import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
import re
from collections import Counter
import csv

with open('C:\\Users\\mpele\\Downloads\\Homework2\\amdb\\imdb.vocab','r',encoding='utf-8') as vocab_file:
    vocab = [word.strip() for word in vocab_file.readlines()]

vocab_set=set(vocab)
freq= Counter()


for folder in ['pos','neg']:
    path=os.path.join('C:\\Users\\mpele\\Downloads\\Homework2\\amdb\\train', folder)

    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
  
        with open(filepath,'r',encoding='utf-8') as wordfile:
            words = wordfile.read().lower().split()
            valid_words=[word for word in words if word in vocab_set]
            freq.update(valid_words)
            
sorted_words = sorted(freq.items(), key=lambda x: x[1])
csv_filename1 = "vocab_frequencies.csv"
with open(csv_filename1, 'w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)
    
    csv_writer.writerow(['Word', 'Frequency'])
    
    for word, frequency in sorted_words:
        csv_writer.writerow([word, frequency])

csv_filename2 = "vocab.csv"
with open(csv_filename2, 'w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)

    for word, frequency in sorted_words:
        if frequency >= 250 and frequency <= 12000:
            csv_writer.writerow([word])
