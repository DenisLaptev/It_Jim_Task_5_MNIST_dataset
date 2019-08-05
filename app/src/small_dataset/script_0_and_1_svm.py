import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import datasets
from sklearn import svm
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
            newdataset_images.append(digits.images[i])

    # print("newdataset_labels:",newdataset_labels)
    # print("len(newdataset_labels):",len(newdataset_labels))
    return newdataset_data, newdataset_labels, newdataset_images


# get dataset (0-9 digits) from sklearn library
digits = datasets.load_digits()

# number of elements in dataset
print(len(digits.data))

# the actual data (features) (рисунки цифр в пикселях)
print(digits.data)

# the actual label we've assigned to the digits data
# (лейблы, которые мы ставим в соответствие рисункам.)
# (в качестве лейблов - цифры от 0 до 9)
print(digits.target)

dataset_data, dataset_labels, newdataset_images = prepare_images_and_labels(digits)

# First, we specify the classifier
clf = svm.SVC(gamma=0.001, C=100)

# pair data-label
# training_set = всё, кроме последних 350 данных,
# потом мы можем использовать последние
# 350 данных для тестирования.
train_data, train_labels = dataset_data[:-350], dataset_labels[:-350]

print('initial_dataset_length=', len(digits.data))
print('two_digit_dataset_length=', len(dataset_data))
print('train_dataset_length=', len(train_data))

# тренировка алгоритма.
clf.fit(train_data, train_labels)

# введём переменные для подсчёта точности угадывания цифр
accuracy = 0
number_of_correct_elements = 0
number_of_elements = 0

for i in range(1, 351):
    print('~~~~~~~~~~~~~Element:', -i, '~~~~~~~~~~~~~')
    number_of_elements += 1
    test = dataset_data[-i]

    # чтобы использовать clf.predict(test) надо изменить размер рисунка, который мы тестируем.
    test = change_shape_of_dataset_element(test)

    # спрашиваем у машины предсказание, чем(какой цифрой) является i-й с конца элемент.
    prediction_of_the_machine = clf.predict(test)
    real_label_of_element = dataset_labels[-i]

    print('Prediction:', prediction_of_the_machine)
    print('Real label:', real_label_of_element)

    # Now we check the accuracy of classification
    # For that, compare the result with test_labels and check which are wrong
    if prediction_of_the_machine == real_label_of_element:
        number_of_correct_elements += 1

    accuracy = number_of_correct_elements * 100.0 / number_of_elements
    print("accuracy=", accuracy)
    print("number_of_correct_elements=", number_of_correct_elements)
    print("number_of_elements=", number_of_elements)

    # визуализируем i-й с конца элемент, и сверяем с предсказанием.
    # plt.imshow(newdataset_images[-i], cmap=plt.cm.gray_r, interpolation="nearest")
    # plt.show()