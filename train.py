import json
import tensorflow as tf
import numpy as np

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def trainSarcasm(Model, trainFile):
    model = Model
    with open(trainFile, 'r') as R:
        data = [json.loads(l) for l in R.readlines()]

    labels = []
    comments = []
    for dicti in data:
        labels.append(dicti.get("label"))
        comments.append(dicti.get("comment"))

    labels1 = list(map(int, labels))
    labels2 = np.asarray(labels1)

    comments_vector = []
    Lem = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    for comment in comments:
        punctuation = RegexpTokenizer(r'\w+')
        tokens = punctuation.tokenize(comment)
        words = [Lem.lemmatize(word) for word in tokens if not word in stop_words]
        count_w = 1
        vector_word = np.zeros(300)
        for word in words:
            if words.index(word) > 1:
                count_w += 1
            if word in model:
                vector_word = np.add(vector_word, model[word])
            else:
                vector_word = np.add(vector_word, np.zeros(300))

        vector_words = np.divide(vector_word, count_w)
        comments_vector.append(vector_words)

    comment_n = np.asarray(comments_vector)

    # Tensorflow reference from https://www.tensorflow.org/api_docs/python/tf/keras/layers

    Model1 = tf.keras.models.Sequential()

    Model1.add(tf.keras.layers.Dense(300, kernel_initializer='glorot_uniform', activation='relu', input_shape=(300,)))
    Model1.add(tf.keras.layers.Dense(150, kernel_initializer='glorot_uniform', activation='relu'))
    Model1.add(tf.keras.layers.Dense(100, kernel_initializer='glorot_uniform', activation='relu'))
    Model1.add(tf.keras.layers.Dense(1, kernel_initializer='glorot_uniform', activation='sigmoid'))
    Model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    Model1.fit(comment_n, labels2, batch_size=400, epochs=5)

    return Model1
