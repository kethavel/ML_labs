from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt


def makeDataAndTarget(array: np.ndarray, dictionary: dict) -> (np.ndarray, np.ndarray):
    """Заменяет значения в соответствии со словарем dictionary, выделяет последний столбец в target,
    приводит значения к float

    :param array: массив исходных данных
    :param dictionary: словарь, в соответствии с которым заменяются значения
    :returns: data: массив значений (столбцы array без последнего)
              target: массив соответствующих классов (последний столбец array)
    """
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


def lab1_1(dataset: np.ndarray, targetDataset: np.ndarray, dataName: str) -> None:
    """1 пункт лабораторной работы: Исследуйте, как объем обучающей выборки и количество тестовых данных,
    влияет на точность классификации в датасетах с помощью наивного Байесовского классификатора. Постройте графики
    зависимостей точности на обучающей и тестовой выборках в зависимости от их соотношения.

    :param dataset: даные для обучения
    :param targetDataset: соответствующие классы
    :param dataName: имя данных, с которыми работаем"""
    listXPoints = []
    listYPoints = []
    for i in range(1, 10):
        testPercent = i / 10
        x_train, x_test, y_train, y_test = train_test_split(dataset, targetDataset, test_size=testPercent)
        gnb = GaussianNB()
        gnb.fit(x_train, y_train)
        y_pred = gnb.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        listXPoints.append(testPercent)
        listYPoints.append(accuracy)
    plt.plot(listXPoints, listYPoints, scalex=False, scaley=False)
    plt.xlabel('Test data part out of a total amount')
    plt.ylabel('Part of correct prediction')
    plt.title('Accuracy of Bayes classifier for ' + dataName + ' (more is better)')
    plt.show()


def lab1_2(featuresMinus1: np.ndarray, featuresPlus1: np.ndarray) -> None:
    """2 пункт лабораторной работы: Сгенерируйте 100 точек с двумя признаками X1 и X2 в соответствии с нормальным
    распределением так, что одна и вторая часть точек (класс -1 и класс 1) имеют параметры: мат. ожидание X1,
    мат. ожидание X2, среднеквадратические отклонения для обеих переменных, соответствующие вашему варианту (указан в
    таблице). Построить диаграммы, иллюстрирующие данные. Построить Байесовский классификатор и оценить качество
    классификации с помощью различных методов (точность, матрица ошибок, ROС и PR-кривые). Является ли построенный
    классификатор «хорошим»?

    :param featuresMinus1: массив признаков X1 и X2 для класса -1
    :param featuresPlus1: массив признаков X1 и X2 для класса +1"""
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
    x_train, x_test, y_train, y_test = train_test_split(dataset, targetDataset, test_size=0.33)
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    # оцениваем качество
    prediction: np.ndarray = gnb.predict(x_test)
    # точность
    print("accuracy = {}".format(accuracy_score(y_test, prediction)))
    # матрица ошибок
    metrics.plot_confusion_matrix(gnb, x_test, y_test)
    plt.title('Confusion matrix')
    plt.show()
    # ROС
    metrics.plot_roc_curve(gnb, x_test, y_test)
    plt.title('ROC curve')
    plt.show()
    # PR
    metrics.plot_precision_recall_curve(gnb, x_test, y_test)
    plt.title('PR curve')
    plt.show()


def lab1_3(dataset: np.ndarray, targetDataset: np.ndarray, dataName: str) -> None:
    """3 пункт:	Постройте классификатор на основе метода k ближайших соседей для обучающего множества Glass (glass.csv).
    a.	Постройте графики зависимости ошибки классификации от количества ближайших соседей.
    b.	Определите подходящие метрики расстояния и исследуйте, как тип метрики расстояния влияет на точность
    классификации.
    c.	Определите, к какому типу стекла относится экземпляр с характеристиками:
    RI =1.516 Na =11.7 Mg =1.01 Al =1.19 Si =72.59 K=0.43 Ca =11.44 Ba =0.02 Fe =0.1
    """

    # график зависимости точности от количества соседей
    listXPoints = []
    listYPoints = []
    for i in range(1, 10):
        classifier = KNeighborsClassifier(n_neighbors=i)
        x_train, x_test, y_train, y_test = train_test_split(dataset, targetDataset, test_size=0.33)
        classifier.fit(x_train, y_train)
        y_predicted = classifier.predict(x_test)
        accuracy = accuracy_score(y_test, y_predicted)
        listXPoints.append(i)
        listYPoints.append(accuracy)
    plt.plot(listXPoints, listYPoints, scaley=False)
    plt.xlabel('Amount of neighbors')
    plt.ylabel('Part of correct prediction')
    plt.title('Accuracy of K Neighbors Classifier for ' + dataName + '(more is better)')
    plt.show()

    # точность в зависимости от метрики
    print('Metrics accuracy:')
    metrics = ('euclidean', 'manhattan', 'chebyshev')
    for metric in metrics:
        classifier = KNeighborsClassifier(metric=metric)
        x_train, x_test, y_train, y_test = train_test_split(dataset, targetDataset, test_size=0.33)
        classifier.fit(x_train, y_train)
        y_predicted = classifier.predict(x_test)
        accuracy = accuracy_score(y_test, y_predicted)
        print('accuracy for {} metric: {}'.format(metric, accuracy))

    classifier = KNeighborsClassifier(metric='minkowski', p=3)
    x_train, x_test, y_train, y_test = train_test_split(dataset, targetDataset, test_size=0.33)
    classifier.fit(x_train, y_train)
    y_predicted = classifier.predict(x_test)
    accuracy = accuracy_score(y_test, y_predicted)
    print('accuracy for minkowski metric with p = 3: {}'.format(accuracy))

    # определить тип стекла
    RI = 1.516
    Na = 11.7
    Mg = 1.01
    Al = 1.19
    Si = 72.59
    K = 0.43
    Ca = 11.44
    Ba = 0.02
    Fe = 0.1
    classifier = KNeighborsClassifier()
    x_train, x_test, y_train, y_test = train_test_split(dataset, targetDataset, test_size=0.33)
    classifier.fit(x_train, y_train)
    y_predicted = classifier.predict(x_test)
    accuracy = accuracy_score(y_test, y_predicted)
    unknownGlass = np.array([RI, Na, Mg, Al, Si, K, Ca, Ba, Fe], ndmin=2)
    prediction = classifier.predict(unknownGlass)
    print('Unknown glass is {0[0]} class. Accuracy: {1}'.format(prediction, accuracy))
    return


if __name__ == '__main__':
    tick_tack_toe_txt = np.loadtxt('data/tic_tac_toe.txt', delimiter=',', dtype=str)
    tick_tack_toe_data, tick_tack_toe_target = makeDataAndTarget(tick_tack_toe_txt, dictionary={'positive': 1,
                                                                                                'negative': -1,
                                                                                                'x': 2,
                                                                                                'o': 3,
                                                                                                'b': 4})

    print('Lab1.1: tic tac toe')
    # lab1_1(tick_tack_toe_data, tick_tack_toe_target, 'tic_tac_toe.txt')

    spam_csv = np.loadtxt('data/spam.csv', delimiter=',', dtype=str)[1:, 1:]
    spam_data, spam_target = makeDataAndTarget(spam_csv, dictionary={'"spam"': 1,
                                                                     '"nonspam"': -1})
    print('Lab1.1: spam')
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
    print('Lab1.2:')
    # lab1_2(featuresClassMinus1, featuresClassPlus1)

    glass_csv = np.loadtxt('data/glass.csv', delimiter=',', dtype=str)[1:, 1:]
    glassData, glassTarget = makeDataAndTarget(glass_csv, dictionary={'"1"': 1,
                                                                      '"2"': 2,
                                                                      '"3"': 3,
                                                                      '"4"': 4,
                                                                      '"5"': 5,
                                                                      '"6"': 6,
                                                                      '"7"': 7,
                                                                      })
    print('Lab1.3:')
    lab1_3(glassData, glassTarget, 'glass.csv')

