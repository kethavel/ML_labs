import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
from tensorboard.plugins import projector
from sklearn.model_selection import train_test_split


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    :param clf: a classifier
    :param xx: meshgrid ndarray
    :param yy: meshgrid ndarray
    :param params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = plt.contourf(xx, yy, Z, **params)
    return out


def makePlotOfSVC(clf, X0, X1, y, title: str, accuracy: [float, None] = None):
    """Plot the decision boundaries for a classifier.

            Parameters
            ----------
            :param clf: a classifier
            :param X0: x0 coordinates
            :param X1: x1 coordinates
            :param y: classes of data
            :param title: title of the plot
            :param accuracy: accuracy. None if you don't want to print it in the plot
    """
    xx, yy = make_meshgrid(X0, X1)
    plot_contours(clf, xx, yy,
                  alpha=0.8)
    # plt.scatter(X0, X1, c=y, s=20, edgecolors='k')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.xticks(())
    plt.yticks(())
    if accuracy is None:
        plt.title(title)
    else:
        plt.title(title + '\n accuracy = {:.2f}'.format(accuracy))


nn_0_data = pd.read_csv('data/nn_0.csv')
plt.scatter(nn_0_data.loc[nn_0_data['class'] == -1]['X1'].values, nn_0_data.loc[nn_0_data['class'] == -1]['X2'].values,
            c='r')
plt.scatter(nn_0_data.loc[nn_0_data['class'] == 1]['X1'].values, nn_0_data.loc[nn_0_data['class'] == 1]['X2'].values,
            c='b')
# plt.show()

y = nn_0_data.loc[:, 'class':]
y = y.values.ravel()
X = nn_0_data.loc[:, 'X1':'X2']
X = X.values.tolist()
y = [0 if i == -1 else 1 for i in y]
X = np.array(X)
y = np.array(y)

model = Sequential()
model.add(Dense(1, input_dim=2, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.25, random_state=42)
model.fit(X_train, y_train, epochs=10, batch_size=2)

# print(X_test)
metrics = model.evaluate(X_test, y_test)
model.predict(X_test)
makePlotOfSVC(model, X, X, y, '')
plt.show()


