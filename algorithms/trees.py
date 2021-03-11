from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

from algorithms.abstract import Model


class DecisionTree(Model):
    def __init__(self):
        super().__init__()
        self.clf = None
        self.parameters = {
            'clf__criterion': ('gini', 'entropy'),
            'clf__splitter': ('best', 'random')
        }
        self.name = 'Decision Tree'

    def train(self, X_train, y_train):
        clf = DecisionTreeClassifier(splitter='random')
        return super().fit(X_train, y_train, clf)
        # self.clf = clf.fit(X_train, y_train)
        # return self

    def plot(self):
        plot_tree(self.clf)

    def tune(self, X_train, y_train):
        clf = DecisionTreeClassifier()
        self.clf = super().optimize(X_train, y_train, self.parameters, clf, search='rs')
        return self


class RandomForest(Model):
    def __init__(self):
        super().__init__()
        self.clf = None
        self.parameters = {

        }
        self.name = 'Random Forest'

    def train(self, X_train, y_train):
        clf = RandomForestClassifier(criterion='entropy')
        return super().fit(X_train, y_train, clf)
