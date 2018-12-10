from alex_classifier import AlexClassifier
import numpy as np

class AllanClassifier(object):

    def __init__(self, depth):
        self.ecg1_alex_classifier = AlexClassifier(depth)
        self.ecg2_alex_classifier = AlexClassifier(depth)

    def fit(self, ecg1, ecg2, Y):
        self.ecg1_alex_classifier.fit(ecg1, Y)
        self.ecg2_alex_classifier.fit(ecg2, Y)

    def predict(self, ecg1, ecg2):
        prob1 = self.ecg1_alex_classifier.predict_proba(ecg1)
        prob2 = self.ecg1_alex_classifier.predict_proba(ecg2)
        merge = (prob1 + prob2) / 2
        vote = np.argmax(merge, axis=1) + 1
        return vote

    def crossvalidate(self,X,Y):
        pass

