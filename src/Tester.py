import ImportData
import numpy as np

from src import BatchGradiantDescent

inputmatrix, outputmatrix = ImportData.inputfile('data/Auto.Imports/1985.Auto.Imports.Database.data.refined.1var.csv')

parameters = np.ones((2, 1))

inputmatrix = np.c_[np.ones(203), inputmatrix]  # insert column of ones for param naught

#BatchGradiantDescent.gradiantdescentiteration(inputmatrix, outputmatrix, parameters, 10000, True)

BatchGradiantDescent.gradientdescentmatrix(inputmatrix, outputmatrix, 100, True)

#StochasticGradiantDescent.gradiantdecentiteration(inputmatrix, outputmatrix, 10000)

#StochasticGradiantDescent.gradiantdecentmatrix(inputmatrix, outputmatrix, 10000)