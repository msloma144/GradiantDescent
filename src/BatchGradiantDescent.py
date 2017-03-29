import numpy as np

# Batch gradient descent and implementation
# utilizing the normal equations to accelerate low feature count regression
# if var closedform = True.


def closedformregession(inputs, outputs):
    xtran = np.transpose(inputs)
    xTx = np.dot(xtran, inputs)
    xTxinv = np.linalg.pinv(xTx)
    xTy = np.dot(xtran, outputs)
    params = np.dot(xTxinv, xTy)
    return params


def paramatersupdate(inputmatrix, outputs, parameters):
    numofparameters = parameters.shape[0]
    numofsamples = inputmatrix.shape[0]
    learningrate = .001  # learning rate

    # The difference between stochastic and batch is that in batch the whole training set must be run
    # through before updating a parameter, where as in stochastic every example is run through and
    # parameters are updated based on that example. SO it depends on the order of the looping.

    sethypothesis = np.dot(inputmatrix, parameters)
    overallerror = sethypothesis - outputs
    sumofsquares = np.sum(overallerror ** 2) / (2 * numofsamples)  # sum of squares error

    for i in range(0, numofparameters):  # range of the parameters
        gradient = 0
        for j in range(0, numofsamples):  # range of the training examples
            inputrow = inputmatrix[j, :]
            hypothesis = np.dot(inputrow, parameters)
            error = hypothesis - outputs[j]
            # partial of loss in parameters
            gradient += (error * inputmatrix[j, i]) / numofsamples

        parameters[i] -= (learningrate * float(gradient))  # update parameters

    return parameters, sumofsquares


def gradiantdescentiteration(inputmatrix, outputs, parameters, iterations, closedform):
    errorinital = 1
    errorfinal = 0
    cyclecounter = 0

    numoffeatures = inputmatrix.shape[1]

    if (numoffeatures > 8000) or (closedform is False):
        while cyclecounter < iterations:
            if abs(errorfinal - errorinital) < .000001:
                print("Desired Error Achieved on cycle " + str(cyclecounter - 1) + "!")
                print("Parameters: " + str(parameters.reshape(1, 2)))
                break
            else:
                parameters, sumofsquares = paramatersupdate(inputmatrix, outputs, parameters)
                print("Cycle: %s | SOS Error: %.5f" % (cyclecounter, sumofsquares))
                print("Parameters: " + str(parameters.reshape(1, 2)))

            cyclecounter += 1

    else:
        parameters = closedformregession(inputmatrix, outputs)  # utilized closed form to find parameters

    return parameters


def gradientdescentmatrix(inputmatrix, outputs, numIterations, closedform):
    learningrate = .001
    numofsamples = inputmatrix.shape[0]  # number of samples
    numoffeatures = inputmatrix.shape[1]
    parameters = np.ones((numoffeatures, 1))

    if (numoffeatures > 8000) or (closedform is False):
        errorfinal = 0

        for i in range(0, numIterations):
            # parameter update
            hypothesis = np.dot(inputmatrix, parameters)
            error = hypothesis - outputs
            sumofsquares = np.sum(error ** 2) / (2 * numofsamples)  # sum of squares error
            errorinital = sumofsquares

            if abs(errorfinal - float(errorinital)) < .000001:  # if error is low, stop
                break

            #print("Cycle: %s | SOS Error: %.5f" % (i, sumofsquares))
            gradient = np.dot(np.transpose(inputmatrix), error) / numofsamples
            parameters -= learningrate * gradient  # update parameters
            errorfinal = errorinital
        #print(parameters)

    else:
        parameters = closedformregession(inputmatrix, outputs)  # utilized closed form to find parameters

    return parameters
