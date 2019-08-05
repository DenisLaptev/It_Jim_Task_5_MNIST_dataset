import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import datasets
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
import cv2


def change_shape_of_dataset_element(test):
    # чтобы использовать clf.predict(test) надо изменить размер рисунка, который мы тестируем.
    # print('test=', test)
    # print('test.shape=', test.shape)
    test = test.reshape(1, -1)
    # print('-----')
    # print('test=', test)
    # print('test.shape=', test.shape)
    return test

def sort_by_target(mnist):
    reorder_train = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[:60000])]))[:, 1]
    reorder_test = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[60000:])]))[:, 1]
    mnist.data[:60000] = mnist.data[reorder_train]
    mnist.target[:60000] = mnist.target[reorder_train]
    mnist.data[60000:] = mnist.data[reorder_test + 60000]
    mnist.target[60000:] = mnist.target[reorder_test + 60000]


def prepare_images_and_labels(digits):
    newdataset_data = []
    newdataset_labels = []
    newdataset_images = []
    digits_data_length = len(digits.data)
    # print(type(digits))
    for i in range(digits_data_length):
        # print(digits.target[i])
        if digits.target[i] == 0 or digits.target[i] == 1:
            newdataset_data.append(digits.data[i])
            newdataset_labels.append(digits.target[i])
            #newdataset_images.append(digits.images[i])

    # print("newdataset_labels:",newdataset_labels)
    # print("len(newdataset_labels):",len(newdataset_labels))
    return newdataset_data, newdataset_labels, newdataset_images


# get dataset (0-9 digits) from sklearn library
digits = datasets.load_digits()
# Load data from https://www.openml.org/d/554
mnist = fetch_openml('mnist_784', version=1, cache=True)
mnist.target = mnist.target.astype(np.int8)  # fetch_openml() returns targets as strings
sort_by_target(mnist)  # fetch_openml() returns an unsorted dataset
print(mnist)
print(len(mnist))

# number of elements in dataset
print(len(digits.data))

# the actual data (features) (рисунки цифр в пикселях)
print(digits.data)

# the actual label we've assigned to the digits data
# (лейблы, которые мы ставим в соответствие рисункам.)
# (в качестве лейблов - цифры от 0 до 9)
print(digits.target)

print('****************************************')
print(mnist.data.shape)
print(mnist.target.shape)
print(np.unique(mnist.target))
print('****************************************')

dataset_data, dataset_labels, newdataset_images = prepare_images_and_labels(mnist)


# First, we specify the classifier
clf = RandomForestClassifier(n_estimators=100)

# pair data-label
# training_set = всё, кроме последних 350 данных,
# потом мы можем использовать последние
# 350 данных для тестирования.
train_data, train_labels = dataset_data[:-4000], dataset_labels[:-4000]

print('initial_dataset_length=', len(mnist.data))
print('two_digit_dataset_length=', len(dataset_data))
print('train_dataset_length=', len(train_data))

print(np.unique(train_labels))

# тренировка алгоритма.
clf.fit(train_data, train_labels)

# введём переменные для подсчёта точности угадывания цифр
accuracy = 0
number_of_correct_elements = 0
number_of_elements = 0

for i in range(1, 4001):
    #print('~~~~~~~~~~~~~Element:', -i, '~~~~~~~~~~~~~')
    number_of_elements += 1
    test = dataset_data[-i]

    # чтобы использовать clf.predict(test) надо изменить размер рисунка, который мы тестируем.
    test = change_shape_of_dataset_element(test)

    # спрашиваем у машины предсказание, чем(какой цифрой) является i-й с конца элемент.
    prediction_of_the_machine = clf.predict(test)
    real_label_of_element = dataset_labels[-i]

    #print('Prediction:', prediction_of_the_machine)
    #print('Real label:', real_label_of_element)

    # Now we check the accuracy of classification
    # For that, compare the result with test_labels and check which are wrong
    if prediction_of_the_machine == real_label_of_element:
        number_of_correct_elements += 1

    accuracy = number_of_correct_elements * 100.0 / number_of_elements
    # print("accuracy=", accuracy)
    # print("number_of_correct_elements=", number_of_correct_elements)
    # print("number_of_elements=", number_of_elements)

    # визуализируем i-й с конца элемент, и сверяем с предсказанием.
    # plt.imshow(newdataset_images[-i], cmap=plt.cm.gray_r, interpolation="nearest")
    # plt.show()

print("accuracy=", accuracy)
print("number_of_correct_elements=", number_of_correct_elements)
print("number_of_elements=", number_of_elements)



'''initial_dataset_length= 70000
two_digit_dataset_length= 14780
train_dataset_length= 10780
accuracy= 24.5
number_of_correct_elements= 980
number_of_elements= 4000'''