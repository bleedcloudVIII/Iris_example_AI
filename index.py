import numpy as np
import random

INPUT_DIM = 4
OUT_DIM = 3
H_DIM = 10

def relu(t):
    return np.maximum(t, 0)
# relu: { x < 0, y = 0  }
#       { x >= 0, y = x }    

def softmax(t):
    out = np.exp(t)
    return out / np.sum(out)
# softmax: Вероятность
# e^(ti) / Σj(e^(tj))

def softmax_batch(t):
    out = np.exp(t)
    return out / np.sum(out, axis = 1, keepdims = True)

def sparce_cross_entropy(z, y):
    return -np.log(z[0, y])
# E = -Σi(yiLog(zi))

def sparce_cross_entropy_batch(z, y):
    return -np.log(np.array([z[j, y[j]] for j in range(len(y))]))

def to_full(y, num_classes):
    y_full = np.zeros((1, num_classes))
    y_full[0, y] = 1
    return y_full

def to_full_batch(y, num_classes):
    y_full = np.zeros((len(y), num_classes))
    for j, yj in enumerate(y):
        y_full[j, yj] = 1
    return y_full

def relu_deriv(t):
    return (t >= 0).astype(float)
# 1 - вход больше нуля
# 0 - вход меньше нуля


# Вектор-матрица
#x = np.random.randn(1, INPUT_DIM) 
# Ответы
#y = random.radint(0, OUT_DIM-1)

from sklearn import datasets
iris = datasets.load_iris()
#                       ?Чтобы были строки?
dataset = [(iris.data[i][None, ...], iris.target[i]) for i in range(len(iris.target))]
# INPUT_DIM*H_DIM - матрица
W1 = np.random.randn(INPUT_DIM, H_DIM) 
b1 = np.random.randn(1, H_DIM)
W2 = np.random.randn(H_DIM, OUT_DIM)
b2 = np.random.randn(1, OUT_DIM)

# Скорость обучения
ALPHA = 0.0001
NUM_EPOCHS = 200 # Количество эпох(кол-во прохождения цикла)
BATCH_SIZE = 50

loss_arr = []

for ep in range(NUM_EPOCHS):
    random.shuffle(dataset) # Перемешиваем входные данные
    for i in range(len(dataset) // BATCH_SIZE):
        
        batch_x, batch_y = zip(*dataset[i*BATCH_SIZE : i*BATCH_SIZE + BATCH_SIZE])
        x = np.concatenate(batch_x, axis = 0)
        y = np.array(batch_y)

        # Forward - Прямое распространение
        t1 = x @ W1 + b1 # @ - умножение матриц
        h1 = relu(t1)
        t2 = h1 @ W2 + b2
        z = softmax_batch(t2)
        # Ошибка/ Вычисление ошибки
        E = np.sum(sparce_cross_entropy_batch(z, y))

        # Backward - Обратное распространение
        # Вектор правильного ответа
        # y_full = to_full(y, OUT_DIM)
        # dE_dt2 = z - y_full
        # dE_dW2 = h1.T @ dE_dt2
        # dE_db2 = dE_dt2
        # dE_dh1 = dE_dt2 @ W2.T
        # dE_dt1 = dE_dh1 * relu_deriv(t1)
        # dE_dW1 = x.T @ dE_dt1
        # dE_db1 = dE_dt1

        y_full = to_full_batch(y, OUT_DIM)
        dE_dt2 = z - y_full
        dE_dW2 = h1.T @ dE_dt2
        dE_db2 = np.sum(dE_dt2, axis = 0, keepdims = True)
        dE_dh1 = dE_dt2 @ W2.T
        dE_dt1 = dE_dh1 * relu_deriv(t1)
        dE_dW1 = x.T @ dE_dt1
        dE_db1 = np.sum(dE_dt1, axis = 0, keepdims = True)

        # Update - Обновление
        # Смещение в сторону градиента, поэтому минус альфа на градиент(производную)
        W1 = W1 - ALPHA * dE_dW1
        b1 = b1 - ALPHA * dE_db1
        W2 = W2 - ALPHA * dE_dW2
        b2 = b2 - ALPHA * dE_db2
        
        # Для мониторинга изменения ошибок
        loss_arr.append(E)

def predict(x):
    t1 = x @ W1 + b1
    h1 = relu(t1)
    t2 = h1 @ W2 + b2
    z = softmax(t2)
    return z
    
# Рассчёт точности
def calc_accuracy():
    correct = 0
    for x, y in dataset:
        z = predict(x)
        y_pred = np.argmax(z)
        if y_pred == y:
            correct += 1
    acc = correct / len(dataset) # Точность
    return acc

accuracy = calc_accuracy()
# print("Accuracy: ", accuracy)

x1 = input()
x2 = input()
x3 = input()
x4 = input()
x_input = np.array([float(x1), float(x2), float(x3), float(x4)])

probs = predict(x_input)
pred_class = np.argmax(probs)
class_names = ['Setosa', 'Versicolor', 'Virginica']
print('Predicted class: ', class_names[pred_class])

# График падения ошибки
# import matplotlib.pyplot as plt 
# plt.plot(loss_arr)
# plt.show()