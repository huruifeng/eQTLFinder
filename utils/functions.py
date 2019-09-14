
import numpy as np


def acc_dist(y1, y0):
    '''
    calculate the differences between the true value and predicted values
    :param
        y_true: the true label value
        y_predict: values from model prediction
    :return:
        the accuracy of the predicted results
    '''

    ## for binary classification, 1 is best
    y0 = y0.flatten()
    x = y0 > 0.5
    return np.mean(x == y1)

    ## for continuous values, 0 is best
    #return K.sum(K.abs(y_true - y_pred))

