import tensorflow.keras.backend as K

def frobenius_norm(y_pred, y_true):
    y = K.square(y_pred - y_true)
    res =  K.sqrt( K.sum(y,-1) )
    return res


def leakage(y_pred, y_true):
    pred_sum = K.sum(K.abs(y_true-y_pred),-1)
    actual_sum = K.sum(y_true, -1)
    res = pred_sum/actual_sum
    
    # convert percent to number
    return res*100.


def mean_square_percent(y_true, y_pred):
    # Implement MSE directly to avoid TF version-specific imports
    return K.mean(K.square(y_pred * 100 - y_true * 100), axis=-1)


