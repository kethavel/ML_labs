from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
import numpy as np
from random import sample
import matplotlib.pyplot as plt
from typing import Literal


def makeDataAndTarget(array: np.ndarray, dictionary: dict) -> (np.ndarray, np.ndarray):
    """Replace given str values in array with int ones"""
    target = array[:, -1].copy()
    data = array[:, 0:-1].copy()
    with np.nditer(target, op_flags=['readwrite']) as it:
        for i in it:
            if str(i) in dictionary:
                i[...] = dictionary[str(i)]
    with np.nditer(data, op_flags=['readwrite']) as it:
        for i in it:
            if str(i) in dictionary:
                i[...] = dictionary[str(i)]
    data = np.array(data).astype(float)
    target = np.array(target).astype(float)
    return data, target


def divideDataset(dataset: np.ndarray, targetDataset: np.ndarray, testPercent: float) -> \
        (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """Разделяем выборку на тренировочную и тестовую случайным образом
    :param dataset: данные, которые нужно разделить
    :param targetDataset: массив классов, соответствующий dataset
    :param testPercent: количество тестовых данных, выраженное в доле от общего количества данных
    :returns: trainingData, testData, trainingTarget, testTarget - соответствующие массивы разделденных данных
    """
    if testPercent > 1 or testPercent < 0:
        raise ValueError('testPercent has to be in between 0 and 1')
    if dataset.shape[0] != targetDataset.shape[0]:
        raise ValueError('dataset and targetDataset has different length')

    dataLength = dataset.shape[0]
    testLength = int(dataLength * testPercent)
    trainingLength = dataLength - testLength
    testNumberSequence = sample(range(dataLength), k=testLength)
    # создаем массивы под данные
    trainingData = np.ndarray(shape=(trainingLength,) + dataset.shape[1:], dtype=float)
    trainingTarget = np.ndarray(shape=(trainingLength,) + targetDataset.shape[1:], dtype=float)
    testData = np.ndarray(shape=(testLength,) + dataset.shape[1:], dtype=float)
    testTarget = np.ndarray(shape=(testLength,) + targetDataset.shape[1:], dtype=float)
    # заполняем
    testIndex = 0
    dataIndex = 0
    for i in range(dataLength):
        if i in testNumberSequence:
            testData[testIndex] = dataset[i]
            testTarget[testIndex] = targetDataset[i]
            testIndex += 1
        else:
            trainingData[dataIndex] = dataset[i]
            trainingTarget[dataIndex] = targetDataset[i]
            dataIndex += 1
    return trainingData, testData, trainingTarget, testTarget


def lab1_1(dataset: np.ndarray, targetDataset: np.ndarray, dataName: str) -> None:
    """1 пункт лабораторной работы: Исследуйте, как объем обучающей выборки и количество тестовых данных,
    влияет на точность классификации в датасетах с помощью наивного Байесовского классификатора. Постройте графики
    зависимостей точности на обучающей и тестовой выборках в зависимости от их соотношения. """
    listXPoints = []
    listYPoints = []
    for i in range(1, 10):
        testPercent = i / 10
        data, test, target, testTarget = divideDataset(dataset, targetDataset, testPercent)
        gnb = GaussianNB()
        gnb.fit(data, target)
        y_pred = gnb.predict(test)
        listXPoints.append(testPercent)
        listYPoints.append(float((testTarget != y_pred).sum()) / float(test.shape[0]))
    plt.plot(listXPoints, listYPoints, scalex=False, scaley=False)
    plt.xlabel('Test data part out of a total amount')
    plt.ylabel('Part of mislabeled points')
    plt.title('Accuracy of Bayes classifier for ' + dataName)
    plt.show()


def lab1_2(featuresMinus1: np.ndarray, featuresPlus1: np.ndarray) -> None:
    """2 пункт лабораторной работы: 2.	Сгенерируйте 100 точек с двумя признаками X1 и X2 в соответствии с нормальным
    распределением так, что одна и вторая часть точек (класс -1 и класс 1) имеют параметры: мат. ожидание X1,
    мат. ожидание X2, среднеквадратические отклонения для обеих переменных, соответствующие вашему варианту (указан в
    таблице). Построить диаграммы, иллюстрирующие данные. Построить Байесовский классификатор и оценить качество
    классификации с помощью различных методов (точность, матрица ошибок, ROС и PR-кривые). Является ли построенный
    классификатор «хорошим»? """
    # сторим диаграмму, иллюстрирующую данные
    plt.plot(featuresMinus1[:, 0], featuresMinus1[:, 1], 'ro')
    plt.plot(featuresPlus1[:, 0], featuresPlus1[:, 1], 'bo')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Classes -1 (red) and 1 (blue)')
    plt.show()

    # строим классификатор
    dataset = np.concatenate((featuresPlus1, featuresMinus1))
    targetDataset = np.array([1 for i in range(featuresPlus1.shape[0])] + [-1 for i in range(featuresMinus1.shape[0])])
    data, test, target, testTarget = divideDataset(dataset, targetDataset, 0.5)
    gnb = GaussianNB()
    gnb.fit(data, target)
    # оцениваем качество
    prediction: np.ndarray = gnb.predict(test)
    # точность
    print("Number of mislabeled points out of a total {} points : {} , percent of mistakes: {}".format(
        test.shape[0],
        (testTarget != prediction).sum(),
        float((testTarget != prediction).sum()) / float(test.shape[0])))
    # матрица ошибок
    metrics.plot_confusion_matrix(gnb, test, testTarget)
    plt.show()
    # ROС
    metrics.plot_roc_curve(gnb, test, testTarget)
    plt.show()
    # PR
    metrics.plot_precision_recall_curve(gnb, test, testTarget)
    plt.show()


if __name__ == '__main__':
    tick_tack_toe_txt = np.loadtxt('data/tic_tac_toe.txt', delimiter=',', dtype=str)
    tick_tack_toe_data, tick_tack_toe_target = makeDataAndTarget(tick_tack_toe_txt, dictionary={'positive': 1,
                                                                                                'negative': -1,
                                                                                                'x': 2,
                                                                                                'o': 3,
                                                                                                'b': 4})
    # lab1_1(tick_tack_toe_data, tick_tack_toe_target, 'tic_tac_toe.txt')

    spam_csv = np.loadtxt('data/spam.csv', delimiter=',', dtype=str)[1:, 1:]
    spam_data, spam_target = makeDataAndTarget(spam_csv, dictionary={'"spam"': 1,
                                                                     '"nonspam"': -1})
    # lab1_1(spam_data, spam_target, 'spam.csv')

    # Вариант	                        5
    # Матем. ожид. X1 (класс -1)	    14
    # Матем. ожид. X2 (класс -1)	    10
    # Дисперсия (класс -1)	            4
    # Матем. ожид. X1 (класс1)	        16
    # Матем. ожид. X2 (класс 1)	        10
    # Дисперсия (класс 1)	            1
    # Количество элементов (класс -1)	50
    # Количество элементов (класс 1)    50

    # генерируем признаки x1 и x2 с матожиданием loc 14 и 10, дисперсией scale 4 и 4 соответственно.
    # Вывод в массив размером size 50, 2
    featuresClassMinus1 = np.random.normal(loc=(14, 10), scale=(4, 4), size=(50, 2))
    featuresClassPlus1 = np.random.normal(loc=(16, 10), scale=(1, 1), size=(50, 2))

    lab1_2(featuresClassMinus1, featuresClassPlus1)
