import numpy as np
from numpy import mean, std
from pandas import read_csv
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras import Sequential
from keras.layers import Dense

# location of the data
data_url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/haberman.csv'
# load the data
df = read_csv(data_url, header=None)

print("data set shape:", df.shape)
# print("data set summary:", df.describe())
#
# # plot the histograms
# df.hist()
# pyplot.show()

# split into input and output columns
# X -- first columns in the dataset without the last column
# y -- last column in the dataset
X, y = df.values[:, :-1], df.values[:, -1]

# ensure all data are floating point values
X = X.astype("float32")

# encode strings to integer
le = LabelEncoder()
y = le.fit_transform(y)

# prepare k-fold cross validation
kfold = StratifiedKFold(10)

# determine the number of input features
# number of columns of X
n_features = X.shape[1]

# define model
model = Sequential()
model.add(Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
model.add(Dense(1, activation='sigmoid'))

# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy')

# enumerate splits
scores = []
for train_ix, test_ix in kfold.split(X, y):
    # split data
    X_train, X_test, y_train, y_test = X[train_ix], X[test_ix], y[train_ix], y[test_ix]

    # fit the model
    model.fit(X_train, y_train, epochs=200, batch_size=16, verbose=0)

    # predict test set
    y_predict = np.argmax(model.predict(X_test), axis=-1)
    # evaluate the predictions
    score = accuracy_score(y_test, y_predict)
    print("Accuracy is: %.3f" % score)
    scores.append(score)

# print all scores
print('Mean accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
# current accuracy is 0.735

################################################
# Make a prediction
################################################
row = [30, 64, 1]  # a test data
y_predict_of_row = np.argmax(model.predict([row]), axis=-1)
y_predict_of_row = le.inverse_transform(y_predict_of_row)
# report prediction
print("Predicted: %s" % (y_predict_of_row[0]))

################################################
# MLP model without the k-fold cross-validation
################################################

# # split into train and test dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y, random_state=3)
#
# # determine the number of input features
# # number of columns of X
# n_features = X.shape[1]
#
# # define model
# model = Sequential()
# model.add(Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
# model.add(Dense(1, activation='sigmoid'))
#
# # compile the model
# model.compile(optimizer='adam', loss='binary_crossentropy')
#
# # fit the model
# history = model.fit(X_train, y_train, epochs=200, batch_size=16, verbose=0, validation_data=(X_test, y_test))
#
# # predict the test set
# y_predict = np.argmax(model.predict(X_test), axis=-1)
#
# # evaluate prediction
# score = accuracy_score(y_test, y_predict)
# print("Accuracy is: %.3f" % score)
#
# # plot learning curves
# pyplot.title('Learning Curves')
# pyplot.xlabel('Epoch')
# pyplot.ylabel('Cross Entropy')
# pyplot.plot(history.history['loss'], label='train')
# pyplot.plot(history.history['val_loss'], label='val')
# pyplot.legend()
# pyplot.show()
