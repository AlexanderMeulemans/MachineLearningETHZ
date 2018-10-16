import numpy as np

def class2sample_weights(y,class_weights):
    sample_weights = []
    for yi in y:
        sample_weights.append(class_weights[yi])
    return np.array(sample_weights)

def clip_resultfile(y_pred):
    result_file = []
    for yi in y_pred:
        label = np.argmax(yi)
        result_file.append(label)
    return np.array(result_file)

