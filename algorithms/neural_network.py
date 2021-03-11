from sklearn.neural_network import MLPClassifier

from algorithms.abstract import Model


class NeuralNetwork(Model):
    def __init__(self):
        super().__init__()
        self.name = 'Multi-layer Perceptron'

    def train(self, X_train, y_train):
        clf = MLPClassifier()
        return super().fit(X_train, y_train, clf)
