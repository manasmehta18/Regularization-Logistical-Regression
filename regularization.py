import arff
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import itertools as it

data = arff.load(open('chronic_kidney_disease_full.arff', 'r'))
data = data['data']
data = np.array(data)
data1 = pd.DataFrame(data)
data1.columns = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo',
                 'pcv', 'wbcc', 'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'class']

# sns.heatmap(data1.isnull(), yticklabels=False, cmap='viridis')
# plt.show()

data1['age'].fillna(data1['age'].mean(), inplace=True)
data1['bp'].fillna(data1['bp'].mean(), inplace=True)
data1.sg = data1.sg.map({'1.005': 1.005, '1.010': 1.010, '1.015': 1.015, '1.020': 1.020, '1.025': 1.025})
data1['sg'].fillna(1.015, inplace=True)
data1.al = data1.al.map({'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5})
data1['al'].fillna(2, inplace=True)
data1.su = data1.su.map({'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5})
data1['su'].fillna(2, inplace=True)
data1.rbc = data1.rbc.map({'normal': 0, 'abnormal': 1})
data1['rbc'].fillna(0, inplace=True)
data1.pc = data1.pc.map({'normal': 0, 'abnormal': 1})
data1['pc'].fillna(0, inplace=True)
data1.pcc = data1.pcc.map({'present': 0, 'notpresent': 1})
data1['pcc'].fillna(1, inplace=True)
data1.ba = data1.ba.map({'present': 0, 'notpresent': 1})
data1['ba'].fillna(1, inplace=True)
data1['bgr'].fillna(data1['bgr'].mean(), inplace=True)
data1['bu'].fillna(data1['bu'].mean(), inplace=True)
data1['sc'].fillna(data1['sc'].mean(), inplace=True)
data1['sod'].fillna(data1['sod'].mean(), inplace=True)
data1['pot'].fillna(data1['pot'].mean(), inplace=True)
data1['hemo'].fillna(data1['hemo'].mean(), inplace=True)
data1['pcv'].fillna(data1['pcv'].mean(), inplace=True)
data1['wbcc'].fillna(data1['wbcc'].mean(), inplace=True)
data1['rbcc'].fillna(data1['rbcc'].mean(), inplace=True)
data1.htn = data1.htn.map({'yes': 0, 'no': 1})
data1['htn'].fillna(1, inplace=True)
data1.dm = data1.dm.map({'yes': 0, 'no': 1})
data1['dm'].fillna(1, inplace=True)
data1.cad = data1.cad.map({'yes': 0, 'no': 1})
data1['cad'].fillna(1, inplace=True)
data1.appet = data1.appet.map({'good': 0, 'poor': 1})
data1['appet'].fillna(0, inplace=True)
data1.pe = data1.pe.map({'yes': 0, 'no': 1})
data1['pe'].fillna(1, inplace=True)
data1.ane = data1.ane.map({'yes': 0, 'no': 1})
data1['ane'].fillna(1, inplace=True)
data1['class'] = data1['class'].map({'ckd': 0, 'notckd': 1})
data1['class'].fillna(1, inplace=True)

# sns.heatmap(data1.isnull(), yticklabels=False, cmap='viridis')
# plt.show()


cols_to_norm = ['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc']

# data1[cols_to_norm] = data1[cols_to_norm].apply(lambda x: (x - x.mean()))
# data1[cols_to_norm] = data1[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
data1[cols_to_norm] = data1[cols_to_norm].apply(lambda x: (x - x.mean()) / x.std())


train = data1.sample(frac=0.8, random_state=200)
test = data1.drop(train.index)

X_train = train.iloc[:, :-1].values
y_train = train.iloc[:, -1].values

X_test = test.iloc[:, :-1].values
y_test = test.iloc[:, -1].values


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost_function(w, X, y, Lambda):

    m = float(len(y))
    y = y[:, np.newaxis]
    predictions = sigmoid(np.dot(X, w))

    error = (-y * np.log(predictions)) - ((1 - y) * np.log(1 - predictions))
    cost = 1 / m * sum(error)

    reg_cost = cost + (Lambda / (2 * m)) * sum(w ** 2)

    j_0 = (1 / m) * np.dot(X.transpose(), (predictions - y))[0]
    j_1 = (1 / m) * np.dot(X.transpose(), (predictions - y))[1:] + (Lambda / m) * w[1:]

    grad = np.vstack((j_0[:, np.newaxis], j_1))
    return reg_cost[0], grad


def gradient_descent(X, y, w_it, alpha, num_iter, Lambda):

    j_hist = []

    for i in range(num_iter):
        cost_it, grad_it = cost_function(w_it, X, y, Lambda)
        w_it = w_it - (alpha * grad_it)
        j_hist.append(cost_it)

    return w_it, j_hist


def predict(w, X):
    predictions = sigmoid(np.dot(X, w))
    return (predictions > 0.5).astype(int)


lambda_t = np.empty((0, 1))
f_mes_t = np.empty((0, 1))

for Lambda in np.arange(-2, 4, 0.2):

    init_w = np.zeros((X_train.shape[1], 1))

    cost_init, grad_init = cost_function(init_w, X_train, y_train, Lambda)
    # print("Cost at initial w (zeros):", cost_init)

    w, J_history = gradient_descent(X_train, y_train, init_w, 1, 800, Lambda)
    # print("regularized w:\n")
    # print w

    p = predict(w, X_test)

    TP, TN, FP, FN = 0.0, 0.0, 0.0, 0.0

    for pin, yin in it.izip(p, y_test):
        if pin == 0 and yin == 0:
            TP = TP + 1
        elif pin == 1 and yin == 1:
            TN = TN + 1
        elif pin == 0 and yin == 1:
            FP = FP + 1
        elif pin == 1 and yin == 0:
            FN = FN + 1

    pre = TP / (TP + FP)
    rec = TP / (TP + FN)

    f_mes = (2 * pre * rec) / (pre + rec)

    lambda_t = np.append(lambda_t, [Lambda])
    f_mes_t = np.append(f_mes_t, [f_mes])

plt.plot(lambda_t, f_mes_t)
plt.xlabel('lambda')
plt.ylabel('f-measure')
plt.show()
