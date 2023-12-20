# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 19:31:29 2023

@author: IVAN
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('data.csv')

df = df.iloc[np.random.permutation(len(df))]
y = df.iloc[0:100, 4].values
y = np.where(y == "Iris-setosa", 1, -1)
X = df.iloc[0:100, [0, 2]].values
df.head(10)

inputSize = X.shape[1] # количество входных сигналов равно количеству признаков задачи 
hiddenSizes = 10 # число нейронов скрытого (А) слоя 
outputSize = 1 if len(y.shape) else y.shape[1] # количество выходных сигналов равно количеству классов задачи


# матрица весов скрытого слоя
Win = np.zeros((1+inputSize,hiddenSizes)) 
# пороги w0 задаем случайными числами
Win[0,:] = (np.random.randint(0, 3, size = (hiddenSizes))) 
# остальные веса задаем случайно -1, 0 или 1 
Win[1:,:] = (np.random.randint(-1, 2, size = (inputSize,hiddenSizes))) 

# случайно инициализируем веса выходного слоя
Wout = np.random.randint(0, 2, size = (1+hiddenSizes,outputSize)).astype(np.float64)

def predict(Xp):
    # выходы первого слоя = входные сигналы * веса первого слоя
    hidden_predict = np.where((np.dot(Xp, Win[1:,:]) + Win[0,:]) >= 0.0, 1, -1).astype(np.float64)
    # выходы второго слоя = выходы первого слоя * веса второго слоя
    out = np.where((np.dot(hidden_predict, Wout[1:,:]) + Wout[0,:]) >= 0.0, 1, -1).astype(np.float64)
    return out, hidden_predict
#Обучение с фиксированным количеством итераций
n_iter=5
eta = 0.01
for i in range(n_iter):
    for xi, target, j in zip(X, y, range(X.shape[0])):
        pr, hidden = predict(xi) 
        Wout[1:] += ((eta * (target - pr)) * hidden).reshape(-1, 1)
        Wout[0] += eta * (target - pr)
        
        
        #Обучение с граничным условием о сходимости
        eta = 0.01
converged = False  # Переменная для отслеживания сходимости
no_change_count = 0  # Переменная для отслеживания зацикливания

while not converged: # Эпоха
    epoch_weights = []
    for xi, target in zip(X, y):
        pr, hidden = predict(xi)
        Wout[1:] += ((eta * (target - pr)) * hidden).reshape(-1, 1)
        Wout[0] += eta * (target - pr)

    # Проверяем, все ли примеры правильно классифицированы
    all_correct = True
    for xi, target in zip(X, y):
        pr, _ = predict(xi)
        if pr != target:
            all_correct = False
            break

    if all_correct:
        converged = True

    # Если веса не изменились, это может быть признаком зацикливания
    if no_change_count > 0:
        print('Повтор весов')
        break

    weight = Wout.copy()
    weight.sort()
    weight = weight.reshape(1, -1)[0]
    
    epoch_weights.append(weight)
    
    # Сохраняем текущие веса
    for w in epoch_weights:
        if np.array_equal(w, weight):
            print(epoch_weights)
            no_change_count += 1

if converged:
    print('Отсутствуют ошибки')
        

