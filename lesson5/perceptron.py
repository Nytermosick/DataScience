import numpy as np
import pandas as pd

data = pd.ExcelFile('/home/gleb/Документы/boats.xlsx')
data = data.parse('Лист1')

D = np.hstack((data.values[:, 2:5], data.values[:, 8:9])).astype(np.int32)

Y = data.values[:, 1:2].astype(np.int32)

for j in range(0,250,15):
    a = j
    b = j+15
    
    for i in range(len(Y)):
        if a < Y[i] < b:
            Y[i] = b
            
w0 = np.zeros((len(D[0])))
w1 = np.zeros_like(w0)
w2 = np.zeros_like(w0)
w3 = np.zeros_like(w0)
w4 = np.zeros_like(w0)
w5 = np.zeros_like(w0)
w6 = np.zeros_like(w0)
w7 = np.zeros_like(w0)
w8 = np.zeros_like(w0)
w9 = np.zeros_like(w0)

Y0 = np.zeros_like(Y)
Y0[np.where(Y == 15)] = 1

Y1 = np.zeros_like(Y)
Y1[np.where(Y == 30)] = 1

Y2 = np.zeros_like(Y)
Y2[np.where(Y == 45)] = 1

Y3 = np.zeros_like(Y)
Y3[np.where(Y == 60)] = 1

Y4 = np.zeros_like(Y)
Y4[np.where(Y == 90)] = 1

Y5 = np.zeros_like(Y)
Y5[np.where(Y == 120)] = 1

Y6 = np.zeros_like(Y)
Y6[np.where(Y == 135)] = 1

Y7 = np.zeros_like(Y)
Y7[np.where(Y == 150)] = 1

Y8 = np.zeros_like(Y)
Y8[np.where(Y == 210)] = 1

Y9 = np.zeros_like(Y)
Y9[np.where(Y == 255)] = 1

α =  0.2 
β = -0.4 
σ = lambda x: 1 if x > 0 else 0

def f(x, _w):
    s = β + np.sum(x @ _w)
    return σ(s)

def train(w, D, Y):
    _w = w.copy()
    for x, y in zip(D, Y):
        w += α * (y - f(x, w)) * x
    return (w != _w).any()

while train(w0, D, Y0) and \
      train(w1, D, Y1) and \
      train(w2, D, Y2) and \
      train(w3, D, Y3) and \
      train(w4, D, Y4) and \
      train(w5, D, Y5) and \
      train(w6, D, Y6) and \
      train(w7, D, Y7) and \
      train(w8, D, Y8) and \
      train(w9, D, Y9):
    print([ round(x) for x in w1 ])


for x in D:
    print(x, end=' > ')
    for w in [w0,w1,w2,w3,w4,w5,w6,w7,w8,w9]:
        print(f(x,w), end=', ')
    print()