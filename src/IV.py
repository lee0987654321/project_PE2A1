import xml.etree.ElementTree as etree
import matplotlib.pyplot as plt
import numpy as np
from numpy import exp
from lmfit import Model
import statsmodels.api as sm
import os

projectpath = os.path.dirname(os.getcwd())
datpath ='{}/dat'.format(projectpath)                                           # dat폴더의 경로입니다.
filepath=[]
for (root, directories, files) in os.walk(datpath):       # 분석해야하는 모든 파일의 경로를 알아냅니다.
    for file in files:
        if '.xml' in file:                                # xml파일만 골라냅니다.
            file_path = os.path.join(root, file)
            filepath.append(file_path)

xml = etree.parse(filepath[0])
root = xml.getroot()

# I-V graph
V = []
for v in root.iter('Voltage'):
    V.extend(list(map(float, v.text.split(','))))
I = []
for i in root.iter('Current'):
    I.extend(list(map(float, i.text.split(','))))
    I = list(map(abs, I))

plt.figure(1, [18, 8])
plt.subplot(2, 3, 1)
plt.plot(V, I, 'b.', label='data')
plt.yscale('log')
plt.title('I-V analysis')
plt.xlabel('Voltage[V]')
plt.ylabel('Current[A]')
plt.legend(loc='best')

x = np.array(V[:])
y = np.array(I[:])
fit1 = np.polyfit(x, y, 12)
fit1 = np.poly1d(fit1)

def IV_fit(X, Is, Vt):
    return (Is * (exp(X/Vt) - 1) + fit1(X))

model = Model(IV_fit)
result = model.fit(I, X=V, Is=10**-15, Vt=0.026)

initial_list = []
for i in V:
    x_value = IV_fit(i, 10e-16, 0.026)
    initial_list.append(x_value)

initial = sm.add_constant(np.abs(y))
result1 = sm.OLS(initial_list, initial).fit()

# R-squared
def IVR(y):
    yhat = result.best_fit
    ybar = np.sum(y)/len(y)
    sse = np.sum((yhat - ybar) ** 2)
    sst = np.sum((y - ybar) ** 2)
    return sse/sst

plt.plot(x, result.best_fit, label='best_fit')
plt.plot(x, result.best_fit, 'r-', label='R-squared ={}'.format(IVR(y)))
plt.legend(loc = 'best')

plt.show()