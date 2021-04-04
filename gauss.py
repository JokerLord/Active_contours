import numpy as np
import math


def gaussian(value, sigma):
    return (np.exp(-(value ** 2) / (2 * sigma ** 2))
            / (math.sqrt(2 * math.pi) * sigma))


def gauss(img, sigma):
    res = img.copy()
    m = 6 * int(sigma) + 1
    kernel = gaussian(np.arange(-(m // 2), m // 2 + 1), sigma)
    num_of_rows, num_of_cols = img.shape
    tmp = np.concatenate((img, np.repeat(img[:, -1].reshape(-1, 1), m // 2,
                                         axis=1)), axis=1)
    tmp = np.concatenate((np.repeat(tmp[:, 0].reshape(-1, 1), m // 2,
                                    axis=1), tmp), axis=1)
    for col in range(num_of_cols):
        res[:, col] = tmp[:, col: col + m] @ kernel
    tmp = np.concatenate((np.repeat(res[0].reshape(1, -1), m // 2, axis=0),
                          res), axis=0)
    tmp = np.concatenate((tmp, np.repeat(tmp[-1].reshape(1, -1), m // 2,
                                         axis=0)), axis=0)
    for row in range(num_of_rows):
        res[row, :] = np.transpose(tmp[row: row + m, :]) @ kernel
    return res
