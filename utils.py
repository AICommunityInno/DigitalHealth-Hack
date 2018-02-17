import numpy as np

def get_most_popular_diagnoses(data, percent=.95):
    cumsums = np.cumsum(data.value_counts())
    most_classes = data.value_counts()[cumsums <= percent * len(data)]
    return list(most_classes.index)
