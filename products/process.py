import string
import math
from operator import methodcaller
import pandas as pd
from statistics import mean

import random
random.seed(11)

products_label_id = {   'pos': 1,
                        'neg': 0
                    }

def read_data(data_file):
    with open(data_file) as file:
        data = file.read().splitlines()
    return data

def split_data(data):
    # train/test : 80/20

    random.Random(11).shuffle(data)
    n = len(data)
    print('# of total: {}'.format(n))
    train_n = math.ceil(n * 0.8)

    train_set = data[:train_n]
    test_set = data[train_n:]

    print('# of trainset: {}'.format(len(train_set)))
    print('# of testset: {}'.format(len(test_set)))

    return train_set, test_set

def split_label(data):
    data_split = map(methodcaller("rsplit", "\t", 1), data)
    return map(list, zip(*data_split))

def preprocess(text):
    text = text.translate(str.maketrans({key: " {0} ".format(key) for key in string.punctuation}))

    tokens = text.split()
    tokens = [t.lower() for t in tokens if t.isalpha()]

    return ' '.join(tokens)

def write_csv(texts, labels, output_file):
    labels = [products_label_id[label] for label in labels]
    texts = [preprocess(text) for text in texts]
    L = [len(text.split()) for text in texts]
    print('avg # of words: {:.3f}'.format(mean(L)))

    df = pd.DataFrame(data={'label': labels, 'text': texts})

    df.to_csv(output_file, header=False, index=False)

def main():
    data = read_data('products.train.txt')
    train_data, test_data = split_data(data)

    train_texts, train_labels = split_label(train_data)
    test_texts, test_labels = split_label(test_data)

    write_csv(train_texts, train_labels, 'train.csv')
    write_csv(test_texts, test_labels, 'test.csv')

if __name__ == "__main__":
    main()
