from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
import numpy as np
import numpy.linalg as LA

class Engine:
    def __init__(self):
        self.cosine_score = []
        self.train_set = [] #Documents
        self.test_set = [] #Query

    def addDocument(self, word):
        self.train_set.append(word)

    def setQuery(self, word):
        self.test_set.append(word)

    def process_score(self):
        stopWords = stopwords.words('english')
        # vectorizer = CountVectorizer(stop_words = stopWords)
        vectorizer = CountVectorizer()

        #print vectorizer
        transformer = TfidfTransformer()
        #print transformer

        trainVectorizerArray = vectorizer.fit_transform(self.train_set).toarray()
        testVectorizerArray = vectorizer.transform(self.test_set).toarray()
        # print 'Fit Vectorizer to train set', trainVectorizerArray
        # print 'Transform Vectorizer to test set', testVectorizerArray
        # print
        cx = lambda a, b : round(np.inner(a, b)/(LA.norm(a)*LA.norm(b)), 3)

        for vector in trainVectorizerArray:
            # print vector
            for testV in testVectorizerArray:
                # print testV
                cosine = cx(vector, testV)
                self.cosine_score.append(cosine)
                # print cosine
        return self.cosine_score

    def check_tag(self,tag,tags):
        # data_tag = tag.split(',')
        # data_tags = tags.split(',')
        stat = False
        if tag in tags:
            stat = True

        return stat
