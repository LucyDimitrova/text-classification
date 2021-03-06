import datetime
import joblib

# NLTK
import nltk
from nltk.corpus import stopwords

# scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# memory usage monitoring
from memory_profiler import profile

# download stopwords
nltk.download('stopwords')


class Model:
    """Abstract model to use as a baseline for testing out different classifiers"""
    def __init__(self):
        self.name = ''
        self.parameters = {}
        self.clf = None
        self.pipeline = None

    def train(self, X_train, y_train):
        pass

    @profile
    def fit(self, X_train, y_train, clf):
        """Train model using the given training data and classifier

        :param X_train: set of features
        :param y_train: set of labels
        :param clf: classifier instance
        :return Model instance
        """
        start = datetime.datetime.now()
        print(f"{self.name} training start: {start}")
        pipeline = Pipeline(
            [('tfidf',
              TfidfVectorizer(analyzer='word', sublinear_tf=True, stop_words=set(stopwords.words('english')))),
             ('clf', clf)])
        pipeline.fit(X_train, y_train)
        end = datetime.datetime.now()
        print(f"{self.name} training еnd: {end}")
        print(f"{self.name} training time: ", {end - start})

        self.pipeline = pipeline
        self.clf = pipeline.named_steps['clf']
        return self

    @profile
    def train_ensemble(self, X_train, y_train, clf):
        """Train model using the given training data and classifier in an AdaBoost ensemble

        :param X_train: set of features
        :param y_train: set of labels
        :param clf: classifier instance
        :return Model instance
        """
        start = datetime.datetime.now()
        print(f"Ensemble of {self.name} training start: {start}")
        clf = AdaBoostClassifier(clf)
        pipeline = Pipeline(
            [('tfidf',
              TfidfVectorizer(analyzer='word', sublinear_tf=True, stop_words=set(stopwords.words('english')))),
             ('clf', clf)])
        pipeline.fit(X_train, y_train)
        end = datetime.datetime.now()
        print(f"Ensemble of {self.name} training еnd: {end}")
        print(f"Ensemble of {self.name} ensemble training time: ", {end - start})

        self.pipeline = pipeline
        self.clf = pipeline.named_steps['clf']
        return self

    def predict(self, X_test, prob=False):
        """Predict labels for a given test set

        :param X_test: test set
        :param prob: Boolean, whether probabilities should be returned or not
        :return: array of labels
        """
        return self.pipeline.predict(X_test) if prob is False else self.pipeline.predict_proba(X_test)

    @profile
    def save(self, output_name):
        """Save pipeline into a file

        :param output_name: name to be used for the file
        :return: tuple, output path to saved file and file name
        """
        date = datetime.datetime.today()
        filename = f'{output_name}-{date}.joblib'
        output_path = f'models/{filename}'
        joblib.dump(self.pipeline, output_path)
        return output_path, filename

    def load(self, path):
        """Load pipeline from a file

        :param path: path to file
        :return: Model instance
        """
        self.pipeline = joblib.load(path)
        return self

    def metrics(self, y_test, y_pred):
        """Get classification report

        :param y_test: true labels
        :param y_pred: predicted labels
        :return: text report showing the main classification metrics
        """
        return classification_report(y_test, y_pred, digits=4)

    def accuracy(self, x, y):
        """Get accuracy score

        :param x: features
        :param y: labels
        :return: accuracy classification score
        """
        return accuracy_score(y, self.predict(x))

    def confusion_matrix(self, y_test, y_pred):
        """Get confusion matrix

        :param y_true: true labels
        :param y_pred: predicted labels
        :return: confusion matrix
        """
        return confusion_matrix(y_test, y_pred)

    def output_performance(self, X_train, y_train, X_test, y_test):
        """Print general performance metrics

        :param X_train: set of training example features
        :param y_train: set of training labels
        :param X_test: set of testing example features
        :param y_test: set of testing labels
        :return:
        """
        y_pred = self.predict(X_test)
        print(self.metrics(y_test, y_pred))
        print('Train Accuracy: ', self.accuracy(X_train, y_train))
        print('Test Accuracy: ', self.accuracy(X_test, y_test))

    def optimize(self, X_train, y_train, params, classifier, search='gs'):
        """Hyperparameter tuning optimization

        :param X_train: set of features
        :param y_train: set of labels
        :param params: hyperparameters to be optimized over
        :param classifier: classifier to be used
        :param search: method to be used - GridSearch (the default) or Randomized Search
        :return: Model instance
        """
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(analyzer='word', sublinear_tf=True, stop_words=set(stopwords.words('english')))),
            ('clf', classifier)
        ])
        start = datetime.datetime.now()
        print(f"Performing {search} optimization... {start}")
        print("pipeline:", [name for name, _ in pipeline.steps])

        search = RandomizedSearchCV(pipeline, params, n_iter=5) if search == 'rs' else GridSearchCV(pipeline, params, cv=5)
        search.fit(X_train, y_train)
        end = datetime.datetime.now()
        print(f"Optimization end: {end}")
        print(f"Optimization time: {end - start}")
        best_parameters = search.best_estimator_.get_params()
        print("Best parameters set:")
        for param_name in sorted(params.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))

        self.clf = search
        return self.clf

    def tune(self, X_train, y_train):
        pass

    def test(self, X_test, y_test):
        """Print performance for test set

        :param X_test: test features
        :param y_test: test labels
        :return:
        """
        y_pred = self.predict(X_test)
        print(self.metrics(y_test, y_pred))
        print('Test Accuracy = ', self.accuracy(X_test, y_test))

