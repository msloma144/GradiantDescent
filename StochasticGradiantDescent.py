import numpy as np


def gradiantdecentiteration(inputmatrix, outputs, iterations):
    numofsamples = inputmatrix.shape[0]
    numoffeatures = inputmatrix.shape[1]
    # changing from np.ones(numoffeatures) to current fixed the error issue
    parameters = np.ones((numoffeatures, 1))
    learningrate = .01

    for a in range(0, iterations):
        sethypothesis = np.dot(inputmatrix, parameters)
        overallerror = sethypothesis - outputs
        sumofsquares = np.sum(overallerror ** 2) / (2 * numofsamples)  # sum of squares error
        print("Cycle: %s | SOS Error: %.5f" % (a, sumofsquares))

        for j in range(0, numofsamples):
            inputrow = inputmatrix[j, :]
            hypot = np.dot(inputrow, parameters)
            error = hypot - outputs[j]
            for i in range(0, numoffeatures):
                # update parameters
                # inputrow[i] = inputmatrix[j, i]
                gradient = (inputrow[i] * error) / numofsamples
                parameters[i] -= learningrate * gradient

    print(parameters)


def gradiantdecentmatrix(inputmatrix, outputs, iterations):
    numofsamples = inputmatrix.shape[0]
    numoffeatures = inputmatrix.shape[1]
    parameters = np.ones((numoffeatures, 1))
    learningrate = .01

    for a in range(0, iterations):
        sethypothesis = np.dot(inputmatrix, parameters)
        overallerror = sethypothesis - outputs
        sumofsquares = np.sum(overallerror ** 2) / (2 * numofsamples)  # sum of squares error
        print("Cycle: %s | SOS Error: %.5f" % (a, sumofsquares))

        for j in range(0, numofsamples):
            inputrow = inputmatrix[j, :]  # row used in this iteration
            hypot = np.dot(inputrow, parameters)
            error = hypot - outputs[j]
            # reshape input row to get into 1D matrix
            gradient = (np.transpose(inputrow.reshape((1, numoffeatures))) * error) / numofsamples
            parameters -= learningrate * gradient  # update parameters

    print(parameters)
