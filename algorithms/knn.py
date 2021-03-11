from sklearn.neighbors import KNeighborsClassifier

from algorithms.abstract import Model


class KNN(Model):
    def __init__(self):
        super().__init__()
        self.params = {
            'clf__n_neighbors': (9, 11, 25),
            # 'clf__algorithm': ('ball_tree', 'kd_tree', 'brute', 'auto')
        }
        self.clf = None
        self.name = 'K Neighbors'

    def train(self, X_train, y_train):
        clf = KNeighborsClassifier(n_neighbors=25)
        return super().fit(X_train, y_train, clf)

    def tune(self, X_train, y_train):
        params = self.params
        clf = KNeighborsClassifier()
        self.clf = super().optimize(X_train, y_train, params, clf)
        return self

    def ensemble(self, X_train, y_train):
        clf = KNeighborsClassifier(n_neighbors=25)
        return super().train_ensemble(X_train, y_train, clf)

