import numpy as np
import time
from src import BatchGradiantDescent
import matplotlib.pyplot as plt

# A tester file to find out at what point is gradient descent faster than
# solving the normal equation to find the parameters


def closedformregession(inputs, outputs):
    xtran = np.transpose(inputs)
    xTx = np.dot(xtran, inputs)
    xTxinv = np.linalg.pinv(xTx)
    xTy = np.dot(xtran, outputs)
    params = np.dot(xTxinv, xTy)
    return params


def runGradiant(inputs, outputs):
    BatchGradiantDescent.gradientdescentmatrix(inputs, outputs, 10000000, False)


def runClosed(inputs, outputs):
    closedformregession(inputs, outputs)


def graph(size, m1, m2):
    plt.plot(size, m1, 'bs', size, m2, 'ro')
    # plt.plot(size, m1, 'bs')
    plt.show()


test = np.arange(1, 1500, 100)
#test = [9500]
size = []
gradianttime = []
closedtime = []
prgmstart = time.time()

for i in test:
    size.append(i)
    inputs = np.random.rand(i, i)
    outputs = np.random.rand(i, 1)

    # running the gradient
    starttime = time.time()
    runGradiant(inputs, outputs)
    endtime = time.time()

    runtime = endtime - starttime
    gradianttime.append(runtime)

    # running the closed form
    starttime = time.time()
    runClosed(inputs, outputs)
    endtime = time.time()

    runtime = endtime - starttime
    closedtime.append(runtime)

    print("Iteration: " + str(i))
    print("Runtime: " + str(time.time() - prgmstart))
graph(size, gradianttime, closedtime)

