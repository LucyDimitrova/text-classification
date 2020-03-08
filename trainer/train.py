import os
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split

from preprocessing.reader import load_dataframe
from preprocessing.visualize import label_distribution
from algorithms.linear import LogReg

load_dotenv()

dataset_path = os.getenv("DATASET")
test_case = os.getenv("TEST_CASE")

# load dataset
df = load_dataframe(dataset_path, chunksize=100000)
print(f'Dataset:\n {df}')

# visualize data
try:
    plot_path = f'visualization/distribution-{test_case}.png'
    label_distribution(df, name='Text examples per topic', output_path=plot_path)
except Exception as err:
    print('There was an error plotting distribution')

# split features from labelbs
X = df['text']
y = df['topic']

# split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('Train size: ', X_train.shape[0])
print('Test size: ', X_test.shape[0])

# train model
model = LogReg()
m = model.train(X_train, y_train)

# save model
model_name = 'Logistic Regression'
file_path, file_name = m.save(model_name)

# output performance metrics
m.output_performance(X_train, y_train, X_test, y_test)




