import numpy as np

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def testSarcasm(vector, Model1, comment):

    Lem = WordNetLemmatizer()
    comments_vector1 = []
    comments = comment['comment']
    stop_words = set(stopwords.words('english'))
    punctuation1 = RegexpTokenizer(r'\w+')
    tokens1 = punctuation1.tokenize(comments)
    words1 = [Lem.lemmatize(word) for word in tokens1 if not word in stop_words]
    count_t = 1
    vector_word1 = np.zeros(300)
    for word in words1:
        if words1.index(word) > 1:
            count_t += 1
        if word in vector:
            vector_word1 = np.add(vector_word1, vector[word])
        else:
            vector_word1 = np.add(vector_word1, np.zeros(300))

    vector_words1 = np.divide(vector_word1, count_t)
    comments_vector1.append(vector_words1)

    comment_n1 = np.asarray(comments_vector1)
    label = Model1.predict(comment_n1)
    label = (label > 0.4)
    return label