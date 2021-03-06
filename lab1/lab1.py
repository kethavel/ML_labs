from typing import Tuple
from sklearn.metrics import accuracy_score, plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree


def makeDataAndTarget(array: np.ndarray, dictionary: dict) -> (np.ndarray, np.ndarray):
    """Создает новый массив данных и классов, заменяет значения в соответствии со словарем dictionary,
    выделяет последний столбец в target, приводит значения к float

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
        print('part of test data: {}, accuracy: {:.2f}'.format(testPercent, accuracy))
        listXPoints.append(testPercent)
        listYPoints.append(accuracy)
    plt.plot(listXPoints, listYPoints, scalex=False, scaley=False)
    plt.xlabel('Test data part out of a total amount')
    plt.ylabel('Part of correct predictions')
    plt.title('Accuracy of Bayes classifier for ' + dataName)
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
    targetDataset = np.array([1 for _ in range(featuresPlus1.shape[0])] + [-1 for _ in range(featuresMinus1.shape[0])])
    x_train, x_test, y_train, y_test = train_test_split(dataset, targetDataset, test_size=0.33)
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    # оцениваем качество
    prediction: np.ndarray = gnb.predict(x_test)
    # точность
    print("accuracy = {}".format(accuracy_score(y_test, prediction)))
    # матрица ошибок
    plot_confusion_matrix(gnb, x_test, y_test)
    plt.title('Confusion matrix (accuracy = {:.2f})'.format(accuracy_score(y_test, prediction)))
    plt.show()
    # ROС
    plot_roc_curve(gnb, x_test, y_test)
    plt.title('ROC curve (accuracy = {:.2f})'.format(accuracy_score(y_test, prediction)))
    plt.show()
    # PR
    plot_precision_recall_curve(gnb, x_test, y_test)
    plt.title('PR curve (accuracy = {:.2f})'.format(accuracy_score(y_test, prediction)))
    plt.show()


def lab1_3(dataset: np.ndarray, targetDataset: np.ndarray, dataName: str) -> None:
    """3 пункт:	Постройте классификатор на основе метода k ближайших соседей для обучающего множества Glass (glass.csv).
    a.	Постройте графики зависимости ошибки классификации от количества ближайших соседей.
    b.	Определите подходящие метрики расстояния и исследуйте, как тип метрики расстояния влияет на точность
    классификации.
    c.	Определите, к какому типу стекла относится экземпляр с характеристиками:
    RI =1.516 Na =11.7 Mg =1.01 Al =1.19 Si =72.59 K=0.43 Ca =11.44 Ba =0.02 Fe =0.1

    :param dataset: даные для обучения
    :param targetDataset: соответствующие классы
    :param dataName: имя данных, с которыми работаем
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
        print('amount of neighbours: {}, accuracy: {:.2f}'.format(i, accuracy))
        listXPoints.append(i)
        listYPoints.append(accuracy)
    plt.plot(listXPoints, listYPoints, scaley=False)
    plt.xlabel('Amount of neighbors')
    plt.ylabel('Part of correct prediction')
    plt.title('Accuracy of K Neighbors Classifier for ' + dataName)
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
        print('accuracy for {} metric: {:.2f}'.format(metric, accuracy))

    classifier = KNeighborsClassifier(metric='minkowski', p=3)
    x_train, x_test, y_train, y_test = train_test_split(dataset, targetDataset, test_size=0.33)
    classifier.fit(x_train, y_train)
    y_predicted = classifier.predict(x_test)
    accuracy = accuracy_score(y_test, y_predicted)
    print('accuracy for minkowski metric with p = 3: {:.2f}'.format(accuracy))

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
    classifier = KNeighborsClassifier(metric='manhattan')
    x_train, x_test, y_train, y_test = train_test_split(dataset, targetDataset, test_size=0.33)
    classifier.fit(x_train, y_train)
    y_predicted = classifier.predict(x_test)
    accuracy = accuracy_score(y_test, y_predicted)
    unknownGlass = np.array([RI, Na, Mg, Al, Si, K, Ca, Ba, Fe], ndmin=2)
    prediction = classifier.predict(unknownGlass)
    print('Unknown glass is {0[0]} class. Accuracy: {1:.2f}'.format(prediction, accuracy))


def lab1_4(dataTuple: Tuple[np.ndarray, ...], part='all') -> None:
    """
    a.	Постройте алгоритм метода опорных векторов с линейным ядром. Визуализируйте разбиение пространства
      признаков на области с помощью полученной модели. Выведите количество полученных опорных
      векторов, а также матрицу ошибок классификации на обучающей и тестовой выборках.
    b.	Постройте алгоритм метода опорных векторов с линейным ядром. Добейтесь нулевой ошибки сначала на обучающей
      выборке, а затем на тестовой, путем изменения штрафного параметра. Выберите оптимальное значение данного параметра
      и объясните свой выбор. Всегда ли нужно добиваться минимизации ошибки на обучающей выборке?
    c.	Постройте алгоритм метода опорных векторов, используя различные ядра (линейное, полиномиальное степеней 1-5,
      сигмоидальная функция, гауссово). Визуализируйте разбиение пространства признаков на области с помощью полученных
      моделей. Сделайте выводы.
    d.	Постройте алгоритм метода опорных векторов, используя различные ядра (полиномиальное степеней 1-5, сигмоидальная
      функция, гауссово). Визуализируйте разбиение пространства признаков на области с помощью полученных моделей.
      Сделайте выводы.
    e.	Постройте алгоритм метода опорных векторов, используя различные ядра (полиномиальное степеней 1-5, сигмоидальная
      функция, гауссово). Изменяя значение параметра ядра (гамма), продемонстрируйте эффект переобучения, выполните при
      этом визуализацию разбиения пространства признаков на области.

    :param dataTuple: Tuple из массивов данных для каждого пункта лабораторной в последовательности a, a_test, b, b_test
    , c, c_test, d, d_test, e, e_test
    :param part: Та часть лабораторной, которую следует запустить. Доступны 'a', 'b', 'c', 'd', 'e'. Чтобы запустить
    все и сразу - 'all'
    """

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
        plt.scatter(X0, X1, c=y, s=20, edgecolors='k')
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

    def partA() -> None:
        """Пункт a"""
        # пункт a
        print('part a:')
        x_train, y_train = makeDataAndTarget(dataTuple[0], dictConverter)
        x_test, y_test = makeDataAndTarget(dataTuple[1], dictConverter)
        dataset = np.concatenate((x_train.copy(), x_test.copy()), axis=0)
        targetDataset = np.concatenate((y_train.copy(), y_test.copy()), axis=0)
        classifier = SVC(kernel='linear')
        classifier.fit(x_train, y_train)
        y_predicted = classifier.predict(x_test)
        accuracy = accuracy_score(y_test, y_predicted)
        makePlotOfSVC(classifier, dataset[:, 0], dataset[:, 1], targetDataset, 'SVC with linear kernel', accuracy)
        plt.show()
        plot_confusion_matrix(classifier, x_test, y_test)
        plt.title('Confusion matrix for test data')
        plt.show()
        plot_confusion_matrix(classifier, x_train, y_train)
        plt.title('Confusion matrix for train data')
        plt.show()
        print('Number of support vectors for each class: {}'.format(classifier.n_support_))

    def partB() -> None:
        """Пункт b"""
        # пункт b
        print('part b:')
        x_train, y_train = makeDataAndTarget(dataTuple[2], dictConverter)
        x_test, y_test = makeDataAndTarget(dataTuple[3], dictConverter)
        dataset = np.concatenate((x_train.copy(), x_test.copy()), axis=0)
        targetDataset = np.concatenate((y_train.copy(), y_test.copy()), axis=0)

        # добиваемся нулевой погрешности в тренировочных данных
        # добиваемся нулевой погрешности в тестовых данных
        Cs = (1, 1000)
        for C in Cs:
            classifier = SVC(kernel='linear', C=C)
            classifier.fit(x_train, y_train)
            y_predicted = classifier.predict(x_train)
            # смотрим на тренировочные данные
            trainAccuracy = accuracy_score(y_train, y_predicted)
            y_predicted = classifier.predict(x_test)
            # смотрим на тренировочные данные
            testAccuracy = accuracy_score(y_test, y_predicted)
            makePlotOfSVC(classifier, dataset[:, 0], dataset[:, 1], targetDataset,
                          'SVC with linear kernel (C = {}). \n'
                          'Accuracy for train data: {} , for test data {}'.format(C, trainAccuracy, testAccuracy), None)
            plt.show()

    def partC() -> None:
        """Пункт c"""
        # пункт c
        print('part c:')
        x_train, y_train = makeDataAndTarget(dataTuple[4], dictConverter)
        x_test, y_test = makeDataAndTarget(dataTuple[5], dictConverter)
        dataset = np.concatenate((x_train.copy(), x_test.copy()), axis=0)
        targetDataset = np.concatenate((y_train.copy(), y_test.copy()), axis=0)
        kernels = ('linear', 'sigmoid', 'rbf')
        for kernel in kernels:
            classifier = SVC(kernel=kernel)
            classifier.fit(x_train, y_train)
            y_predicted = classifier.predict(x_train)
            accuracy = accuracy_score(y_train, y_predicted)
            makePlotOfSVC(classifier, dataset[:, 0], dataset[:, 1], targetDataset,
                          'SVC with {} kernel '.format(kernel), accuracy)
            plt.show()

        for degree in range(1, 6):
            classifier = SVC(kernel='poly', degree=degree)
            classifier.fit(x_train, y_train)
            y_predicted = classifier.predict(x_train)
            accuracy = accuracy_score(y_train, y_predicted)
            makePlotOfSVC(classifier, dataset[:, 0], dataset[:, 1], targetDataset,
                          'SVC with polynomial kernel (degree = {}) '.format(degree), accuracy)
            plt.show()

    def partD() -> None:
        """Пункт d"""
        # пункт d
        print('part d:')
        x_train, y_train = makeDataAndTarget(dataTuple[6], dictConverter)
        x_test, y_test = makeDataAndTarget(dataTuple[7], dictConverter)
        dataset = np.concatenate((x_train.copy(), x_test.copy()), axis=0)
        targetDataset = np.concatenate((y_train.copy(), y_test.copy()), axis=0)
        kernels = ('linear', 'sigmoid', 'rbf')
        for kernel in kernels:
            classifier = SVC(kernel=kernel)
            classifier.fit(x_train, y_train)
            y_predicted = classifier.predict(x_train)
            accuracy = accuracy_score(y_train, y_predicted)
            makePlotOfSVC(classifier, dataset[:, 0], dataset[:, 1], targetDataset,
                          'SVC with {} kernel '.format(kernel), accuracy)
            plt.show()

        for degree in range(1, 6):
            classifier = SVC(kernel='poly', degree=degree)
            classifier.fit(x_train, y_train)
            y_predicted = classifier.predict(x_train)
            accuracy = accuracy_score(y_train, y_predicted)
            makePlotOfSVC(classifier, dataset[:, 0], dataset[:, 1], targetDataset,
                          'SVC with polynomial kernel (degree = {}) '.format(degree), accuracy)
            plt.show()

    def partE() -> None:
        """Пункт e"""
        # пункт e
        print('part e:')
        x_train, y_train = makeDataAndTarget(dataTuple[8], dictConverter)
        x_test, y_test = makeDataAndTarget(dataTuple[9], dictConverter)
        dataset = np.concatenate((x_train.copy(), x_test.copy()), axis=0)
        targetDataset = np.concatenate((y_train.copy(), y_test.copy()), axis=0)
        kernels = ('linear', 'sigmoid', 'rbf')
        gammas = (0.1, 5)
        for gamma in gammas:
            for kernel in kernels:
                classifier = SVC(kernel=kernel, gamma=gamma)
                classifier.fit(x_train, y_train)
                y_predicted = classifier.predict(x_train)
                accuracy = accuracy_score(y_train, y_predicted)
                makePlotOfSVC(classifier, dataset[:, 0], dataset[:, 1], targetDataset,
                              'SVC with {} kernel (gamma = {})'.format(kernel, gamma), accuracy)
                plt.show()

            for degree in range(1, 6):
                classifier = SVC(kernel='poly', degree=degree, gamma=gamma)
                classifier.fit(x_train, y_train)
                y_predicted = classifier.predict(x_train)
                accuracy = accuracy_score(y_train, y_predicted)
                makePlotOfSVC(classifier, dataset[:, 0], dataset[:, 1], targetDataset,
                              'SVC with polynomial kernel (degree = {}, gamma = {}) '.format(degree, gamma), accuracy)
                plt.show()

        classifier = SVC(kernel='rbf', gamma=500)
        classifier.fit(x_train, y_train)
        y_predicted = classifier.predict(x_train)
        accuracy = accuracy_score(y_train, y_predicted)
        makePlotOfSVC(classifier, dataset[:, 0], dataset[:, 1], targetDataset,
                      'SVC with {} kernel (overfitting) (gamma = {})'.format('rbf', 500), accuracy)
        plt.show()

    if len(dataTuple) != 10:
        raise ValueError('dataTuple should contain 10 elements:a, a_test, b, b_test, c, c_test, d, d_test, e, e_test')

    dictConverter = {'red': 1, 'green': 2}

    chosenPart = part.lower()
    if chosenPart == 'all':
        partA()
        partB()
        partC()
        partD()
        partE()
    elif chosenPart == 'a':
        partA()
    elif chosenPart == 'b':
        partB()
    elif chosenPart == 'c':
        partC()
    elif chosenPart == 'd':
        partD()
    elif chosenPart == 'e':
        partE()
    else:
        raise ValueError("This lab doesn't have part 4.{}. Parts a, b, c, d and e are available. To make all parts at "
                         "one call use part='all'", chosenPart)


def lab1_5(datasets: Tuple[np.ndarray, np.ndarray], targetDatasets: Tuple[np.ndarray, np.ndarray],
           part: str = 'all') -> None:
    """
    5.	Постройте классификаторы для различных данных на основе деревьев решений:
    a.	Загрузите набор данных Glass
    из файла glass.csv. Постройте дерево классификации для модели, предсказывающей тип (Type) по остальным признакам.
    Визуализируйте результирующее дерево решения. Дайте интерпретацию полученным результатам. Является ли построенное
    дерево избыточным? Исследуйте зависимость точности классификации от критерия расщепления, максимальной глубины
    дерева и других параметров по вашему усмотрению.
    b.	Загрузите набор данных spam7 из файла spam7.csv. Постройте
    оптимальное, по вашему мнению, дерево классификации для параметра yesno. Объясните, как был осуществлён подбор
    параметров. Визуализируйте результирующее дерево решения. Определите наиболее влияющие признаки. Оцените качество
    классификации.

    :param part: пункт лабораторной
    :param datasets: даные для обучения
    :param targetDatasets: соответствующие классы
    """

    def partA() -> None:
        # пункт a
        classifier = DecisionTreeClassifier()
        x_train, x_test, y_train, y_test = train_test_split(datasets[0], targetDatasets[0], test_size=0.33)
        classifier.fit(x_train, y_train)
        y_predicted = classifier.predict(x_test)
        accuracy = accuracy_score(y_test, y_predicted)
        # вывод дерева
        plot_tree(classifier)
        plt.title('Tree classifier (accuracy = {:.2f})'.format(accuracy))
        plt.show()
        # зависимость точности от параметров
        # критерий расщепления
        criteria = ('gini', 'entropy')
        accuracies = []
        for criterion in criteria:
            classifier = DecisionTreeClassifier(criterion=criterion)
            classifier.fit(x_train, y_train)
            y_pred = classifier.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)
        x = np.arange(2)
        plt.bar(x=x, height=accuracies)
        plt.xticks(x, criteria)
        plt.ylabel('accuracy')
        plt.xlabel('split criteria')
        plt.title('Dependence of accuracy on split criteria')
        plt.show()
        # максимальная глубина расщепления
        max_depths = (1, 3, 5, 7, 9, 10, 15, 30, 100, 500)
        accuracies = []
        for max_depth in max_depths:
            classifier = DecisionTreeClassifier(max_depth=max_depth)
            classifier.fit(x_train, y_train)
            y_pred = classifier.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)
        x = np.arange(len(max_depths))
        plt.bar(x=x, height=accuracies)
        plt.xticks(x, max_depths)
        plt.ylabel('accuracy')
        plt.xlabel('max split depth')
        plt.title('Dependence of accuracy on max split depth')
        plt.show()
        # The minimum number of samples required to split an internal node
        min_samples_splits = (2, 3, 4, 5, 7, 9, 10, 15, 30, 100, 500)
        accuracies = []
        for min_samples_split in min_samples_splits:
            classifier = DecisionTreeClassifier(min_samples_split=min_samples_split)
            classifier.fit(x_train, y_train)
            y_pred = classifier.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)
        x = np.arange(len(min_samples_splits))
        plt.bar(x=x, height=accuracies)
        plt.xticks(x, min_samples_splits)
        plt.ylabel('accuracy')
        plt.xlabel('min samples splits')
        plt.title('Dependence of accuracy on min samples splits')
        plt.show()

    def partB() -> None:
        # пункт b
        # max_depth=10  3, 5, 20
        # min_samples_leaf=30  , 20, 40
        max_depths = (3, 10, 20)
        min_samples_leaves = (10, 30, 50)

        x_train, x_test, y_train, y_test = train_test_split(datasets[1], targetDatasets[1], test_size=0.33)
        for max_depth in max_depths:
            classifier = DecisionTreeClassifier(max_depth=max_depth)
            classifier.fit(x_train, y_train)
            y_predicted = classifier.predict(x_test)
            accuracy = accuracy_score(y_test, y_predicted)
            # матрица ошибок
            plot_confusion_matrix(classifier, x_test, y_test)
            plt.title('Confusion matrix (accuracy = {:.2f}) \n '
                      'max_depth={}'.format(accuracy, max_depth))
            plt.show()
            # ROС
            plot_roc_curve(classifier, x_test, y_test)
            plt.title('ROC curve \n '
                      'max_depth={}'.format(max_depth))
            plt.show()
            # PR
            plot_precision_recall_curve(classifier, x_test, y_test)
            plt.title('PR curve \n '
                      'max_depth={}'.format(max_depth))
            plt.show()

        for min_samples_leaf in min_samples_leaves:
            classifier = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf)
            classifier.fit(x_train, y_train)
            y_predicted = classifier.predict(x_test)
            accuracy = accuracy_score(y_test, y_predicted)
            # матрица ошибок
            plot_confusion_matrix(classifier, x_test, y_test)
            plt.title('Confusion matrix (accuracy = {:.2f}) \n '
                      'min_samples_leaf={}'.format(accuracy, min_samples_leaf))
            plt.show()
            # ROС
            plot_roc_curve(classifier, x_test, y_test)
            plt.title('ROC curve \n '
                      'min_samples_leaf={}'.format(min_samples_leaf))
            plt.show()
            # PR
            plot_precision_recall_curve(classifier, x_test, y_test)
            plt.title('PR curve \n '
                      'min_samples_leaf={}'.format(min_samples_leaf))
            plt.show()

        classifier = DecisionTreeClassifier(max_depth=10, min_samples_leaf=30)
        classifier.fit(x_train, y_train)
        y_predicted = classifier.predict(x_test)
        accuracy = accuracy_score(y_test, y_predicted)
        # матрица ошибок
        plot_confusion_matrix(classifier, x_test, y_test)
        plt.title('Confusion matrix (accuracy = {:.2f})'.format(accuracy))
        plt.show()
        # ROС
        plot_roc_curve(classifier, x_test, y_test)
        plt.title('ROC curve')
        plt.show()
        # PR
        plot_precision_recall_curve(classifier, x_test, y_test)
        plt.title('PR curve')
        plt.show()
        # вывод дерева
        plot_tree(classifier)
        plt.title('Optimal tree classifier (accuracy = {:.2f})'.format(accuracy))
        plt.show()

    chosenPart = part.lower()
    if chosenPart == 'all':
        partA()
        partB()
    elif chosenPart == 'a':
        partA()
    elif chosenPart == 'b':
        partB()
    else:
        raise ValueError("This lab doesn't have part 5.{}. Parts a, b are available. To make all parts at "
                         "one call use part='all'", chosenPart)


def lab1_6(dataset: np.ndarray, targetDataset: np.ndarray, test: np.ndarray, testTarget: np.ndarray) -> None:
    """Загрузите набор данных из файла bank_scoring_train.csv. Это набор финансовых данных, характеризующий
    физических лиц. Целевым столбцом является «SeriousDlqin2yrs», означающий, ухудшится ли финансовая ситуация у
    клиента. Постройте систему по принятию решения о выдаче или невыдаче кредита физическому лицу. Сделайте как
    минимум 2 варианта системы на основе различных классификаторов. Подберите подходящую метрику качества работы
    системы исходя из специфики задачи и определите, принятие решения какой системой сработало лучше на
    bank_scoring_test.csv.

    :param dataset: Набор исходных данных для обучения
    :param targetDataset: Соответствующий набор классов
    :param test: Набор данных для тестирования
    :param testTarget: Соответствующий набор классов для тестирования
    """
    x_train, x_test, y_train, y_test = train_test_split(dataset, targetDataset, test_size=0.33)
    # подбираем лучший параметр для DecisionTreeClassifier
    # критерий оценки - точность, ROC и AUC
    max_depths = (3, 7, 15)
    for max_depth in max_depths:
        classifier = DecisionTreeClassifier(max_depth=max_depth)
        classifier.fit(x_train, y_train)
        prediction = classifier.predict(x_test)
        accuracy = accuracy_score(y_test, prediction)
        # ROС
        plot_roc_curve(classifier, x_test, y_test)
        plt.title('ROC curve (accuracy = {:.2f}, max depth = {})'.format(accuracy, max_depth))
        plt.show()
    # для DecisionTreeClassifier лучший параметр max_depth = 7

    # смотрим на GaussianNB
    classifier = GaussianNB()
    classifier.fit(x_train, y_train)
    prediction = classifier.predict(x_test)
    accuracy = accuracy_score(y_test, prediction)
    # ROС
    plot_roc_curve(classifier, x_test, y_test)
    plt.title('ROC curve (accuracy = {:.2f})'.format(accuracy))
    plt.show()

    # проверяем два лучших варианта на тестовых данных
    bestClassifiers = (DecisionTreeClassifier(max_depth=7), GaussianNB())
    for bestClassifier in bestClassifiers:
        bestClassifier.fit(x_train, y_train)
        prediction = bestClassifier.predict(test)
        accuracy = accuracy_score(testTarget, prediction)
        # ROС
        plot_roc_curve(bestClassifier, test, testTarget)
        plt.title('ROC curve for best classifier (accuracy = {:.2f})'.format(accuracy))
        plt.show()


if __name__ == '__main__':
    # Lab 1.1
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

    # Lab 1.2
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

    # Lab 1.3
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
    # lab1_3(glassData, glassTarget, 'glass.csv')

    # Lab 1.4
    svmdata = (np.loadtxt('data/svmdata_a.txt', dtype=str, delimiter='	', skiprows=1)[:, 1:],
               np.loadtxt('data/svmdata_a_test.txt', dtype=str, delimiter='	', skiprows=1)[:, 1:],
               np.loadtxt('data/svmdata_b.txt', dtype=str, delimiter='	', skiprows=1)[:, 1:],
               np.loadtxt('data/svmdata_b_test.txt', dtype=str, delimiter='	', skiprows=1)[:, 1:],
               np.loadtxt('data/svmdata_c.txt', dtype=str, delimiter='	', skiprows=1)[:, 1:],
               np.loadtxt('data/svmdata_c_test.txt', dtype=str, delimiter='	', skiprows=1)[:, 1:],
               np.loadtxt('data/svmdata_d.txt', dtype=str, delimiter='	', skiprows=1)[:, 1:],
               np.loadtxt('data/svmdata_d_test.txt', dtype=str, delimiter='	', skiprows=1)[:, 1:],
               np.loadtxt('data/svmdata_e.txt', dtype=str, delimiter='	', skiprows=1)[:, 1:],
               np.loadtxt('data/svmdata_e_test.txt', dtype=str, delimiter='	', skiprows=1)[:, 1:])
    print('Lab1.4:')
    # lab1_4(svmdata, part='all')

    # Lab 1.5
    print('Lab1.5:')
    spam7_csv = np.loadtxt('data/spam7.csv', delimiter=',', dtype=str)[1:, :]
    spam7_data, spam7_target = makeDataAndTarget(spam7_csv, dictionary={'"y"': 1,
                                                                        '"n"': -1})
    # lab1_5((glassData, spam7_data), (glassTarget, spam7_target), part='all')

    # Lab 1.6
    bank_train = np.loadtxt('data/bank_scoring_train.csv', delimiter='	', dtype=float, skiprows=1)
    bank_train_data, bank_train_target = bank_train[:, 1:], bank_train[:, 0]
    bank_test = np.loadtxt('data/bank_scoring_test.csv', delimiter='	', dtype=float, skiprows=1)
    bank_test_data, bank_test_target = bank_test[:, 1:], bank_test[:, 0]
    print('Lab1.6:')
    # lab1_6(bank_train_data, bank_train_target, bank_test_data, bank_test_target)
