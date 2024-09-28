import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# open
with open('./data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
num_features_expected = x_train.shape[1]
if x_test.shape[1] != num_features_expected:
    raise ValueError(f"x_test has {x_test.shape[1]} features, but {num_features_expected} were expected.")

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_test, y_predict)

print(f'Accuracy of model: {score * 100}% ')


with open('model.p', 'wb') as f:
    pickle.dump(model, f)
