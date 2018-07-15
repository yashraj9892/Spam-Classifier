import nltk
import os
import random
from collections import Counter
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import NaiveBayesClassifier, classify

stoplist = stopwords.words('english')

def get_data(folder):
    a = []
    files = os.listdir(folder)
    for file in files:
        f = open(folder + file, 'r', encoding = "ISO-8859-1")  ## cause there where some latin words also
        a.append(f.read())
    f.close()
    return a

def preprocess(sentence):
    lemma = WordNetLemmatizer()
    return [lemma.lemmatize(word.lower()) for word in word_tokenize(sentence)]

def getfeatures(text):
     return {word: True for word in preprocess(text) if not word in stoplist}
    
def train(features, samples_proportion):
    trainsize = int(len(features) * samples_proportion)
   
    trainset, testset = features[:trainsize], features[trainsize:]
    print ('Train size = ' + str(len(trainset)) + ' emails')
    print ('Test  size = ' + str(len(testset)) + ' emails')
    
    classifier = NaiveBayesClassifier.train(trainset)
    return trainset, testset, classifier

def print_accuracy(train_set, test_set, classifier):
    
    print ('Accuracy on the training set = ' + str(classify.accuracy(classifier, train_set)))
    print ('Accuracy of the test set = ' + str(classify.accuracy(classifier, test_set)))
    classifier.show_most_informative_features(10)

if __name__ == "__main__":
    spam = get_data('enron1/spam/')
    ham  = get_data('enron1/ham/')
    
    allemails  = [(email, 'spam') for email in spam]
    allemails += [(email, 'ham') for email in ham]
    random.shuffle(allemails)
    
    print (str(len(allemails)) + ' emails')

    allfeatures = [(getfeatures(email), label) for (email, label) in allemails]
##    print(allfeatures[:10])
    print (str(len(allfeatures)) + ' feature sets')


    train_set, test_set, classifier = train(allfeatures, 0.8)


    print_accuracy(train_set, test_set, classifier)
