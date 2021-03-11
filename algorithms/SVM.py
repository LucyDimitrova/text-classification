from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from algorithms.abstract import Model


class SVM(Model):
    def __init__(self):
        super().__init__()
        self.clf = None
        self.parameters = {
            # 'clf__kernel': ('linear', 'rbf', 'poly', 'sigmoid'),
            'clf__C': (0.1, 1, 10, 100, 1000),
            'clf__tol': (0.0001, 0.000001)
            # 'clf__gamma': (0.1, 1, 10, 100)
        }
        self.name = 'SVM'

    def train(self, X_train, y_train, prob=False):
        clf = LinearSVC()
        if prob:
            clf = CalibratedClassifierCV(clf)
        self.clf = clf
        return super().fit(X_train, y_train, clf)

    def tune(self, X_train, y_train):
        params = self.parameters
        classifier = SVC()
        self.clf = super().optimize(X_train, y_train, params, classifier, search='rs')
        return self
