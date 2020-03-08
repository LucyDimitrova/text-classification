import datetime
from sklearn.linear_model import LogisticRegression, SGDClassifier, LogisticRegressionCV

from algorithms.abstract import Model

# memory monitoring
from memory_profiler import profile


class LogReg(Model):
    """Logistic Regression classifier, plus SGDClassifier and cross-validation"""
    def __init__(self):
        super().__init__()
        self.name = 'Logistic Regression'
        self.parameters = {
            'tfidf__use_idf': (True, False),
            'tfidf__norm': ('l1', 'l2'),
            'clf__penalty': ('l2', 'elasticnet', 'l1'),
            'clf__solver': ('saga', ),
            'clf__alpha': (0.00001, 0.0001, 0.001, 0.01, 0.1, 1),
            'clf__loss': ('log', ),
            'clf__C': (0.01, 1.0, 10, 100),
            'clf__max_iter': (10, 100, 1000),
        }
        self.clf = None

    @profile
    def train(self, X_train, y_train):
        """Fit classifier

        :param X_train: set of features
        :param y_train: set of labels
        :return Model instance
        """
        # log_clf = SGDClassifier(loss='log', alpha=0.00001)
        log_clf = LogisticRegression(solver='saga', C=10)
        return super().fit(X_train, y_train, log_clf)

    @profile
    def tune(self, X_train, y_train):
        """HyperParams tuning/optimization

        :param X_train: set of features
        :param y_train: set of labels
        :return Model instance
        """
        params = self.parameters
        classifier = SGDClassifier(loss='log', alpha=0.00001)
        # classifier = LogisticRegression()
        self.clf = super().optimize(X_train, y_train, params, classifier, search='rs')
        return self

    @profile
    def cross_validate(self, X_train, y_train):
        """Train LogisticRegression with cross-validation

        :param X_train: set of features
        :param y_train: set of labels
        :return: Model instance
        """
        start = datetime.datetime.now()
        print(f"Cross-validated {self.name} training start: {start}")
        log_clf = LogisticRegressionCV(cv=5, solver='saga', Cs=10)
        log_clf.fit(X_train, y_train)
        end = datetime.datetime.now()
        print(f"Cross-validated {self.name} training end: {end}")
        print(f"Cross-validated {self.name} training time: {end - start}")

        self.clf = log_clf
        return self
