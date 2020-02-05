import json

from gensim.models import KeyedVectors

from test_1 import testSarcasm
from train import trainSarcasm

if __name__ == "__main__":
    vector = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)  #
    # Google word2vec is loaded. For downloading word2vec
    # https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
    print('Word2vec loaded.')
    train_file = 'train.jsonlist'
    print('Training the model.')
    trained_model = trainSarcasm(vector, train_file)
    print('Model trained.')

    test_file = 'test.jsonlist'
    # Test file is used to predict the results
    with open(test_file, 'r') as R:
        test = [json.loads(l) for l in R.readlines()]
    result = []
    for comment in test:
        result.append(testSarcasm(vector, trained_model, comment))

    print(result)
