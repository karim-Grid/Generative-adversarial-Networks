import numpy as np


def mov_avg(x, w):
    for m in range(len(x)-(w-1)):
        yield sum(np.ones(w) * x[m:m+w]) / w