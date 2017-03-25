import numpy as np


def getnumoffeaturres(filename):
    # find the number of features using the first line of data
    line = open(filename).readline()
    numoffeaturers = 0
    for char in line:  # for all the chars in the first line
        if char == ",":
            numoffeaturers += 1  # add one to the features
    return numoffeaturers  # last line is the output so not counted


def getnumofexamples(filename):
    # find the number of examples in the data set
    infile = open(filename)
    examples = sum(1 for line in infile)
    infile.close()
    return examples


def examplecleanup(matrix, unknowns):
    for index in unknowns:
        matrix = np.delete(matrix, index, 0)
    return matrix


def inputfile(filename):
    infile = open(filename)  # opening the file of input data
    itrinfile = iter(infile)  # iterator for file

    numofexamples = getnumofexamples(filename)
    numoffeatures = getnumoffeaturres(filename)

    inputmaxtrix = np.empty((numofexamples, numoffeatures))  # whole input matrix
    outputmatrix = np.zeros((numofexamples, 1))

    unknownvals = []

    examplecounter = 0  # counter of the number of examples to place row in correct place
    for line in itrinfile:  # for the lines in open file

        temparray = np.zeros(numoffeatures)  # fill temp array with zeros
        featurecounter = 0  # set feature counter back to zero
        number = ""  # initialize number as empty, used to keep track of input

        for char in line:  # for characters in each line
            if char is "?":  # if char is ? then dump the example for clean up later
                temparray = np.zeros(numoffeatures)
                unknownvals.append(examplecounter)  # marks index for deletion
                break
            elif (char != ",") and (not char.isspace()):  # char is not a , and is not white space
                number += char  # add char to the number
            else:
                if featurecounter == numoffeatures:  # if the feature is the last one in the set then assign it to output
                    outputmatrix[examplecounter] = float(number)
                else:
                    temparray[featurecounter] = float(number)  # set the index of array to the value of that input
                    number = ""  # reset number to empty
                    featurecounter += 1  # increment the feature counter

        inputmaxtrix[examplecounter, :] = temparray
        examplecounter += 1

    unknownvals.reverse()  # reverse the order of the array b/c it passes the object to the method
    inputmaxtrix = examplecleanup(inputmaxtrix, unknownvals)
    outputmatrix = examplecleanup(outputmatrix, unknownvals)
    infile.close()  # close the file
    print(inputmaxtrix)
    return inputmaxtrix, outputmatrix
