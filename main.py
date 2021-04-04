import argparse
import numpy as np
import imageio
import utils
from gauss import gauss
from math import sqrt
from scipy.interpolate import interp1d

K = 0.9
EPS = 0.007
SIGMA = 1


def mse(img1, img2):
    if img1.shape == img2.shape:
        return (np.sum((img1 - img2) ** 2)
                / (img1.shape[0] * img1.shape[1]))
    else:
        return None


def create_int_energy_mat(alpha, beta, n):
    a = alpha * n ** 2
    b = beta * n ** 4
    res = np.zeros((n, n))
    tmp = np.array([b, -a - 4 * b, 2 * a + 6 * b, -a - 4 * b, b])
    for i in range(n):
        for j in range(5):
            res[i, (j - 2 + i) % n] = tmp[j]
    return res


def gradient(matrix):
    kernel = np.array([-1, 0, 1])
    m = kernel.shape[0]
    tmp = np.concatenate((matrix, matrix[:, -1].reshape(-1, 1)), axis=1)
    tmp = np.concatenate((tmp[:, 0].reshape(-1, 1), tmp), axis=1)
    num_of_rows, num_of_cols = matrix.shape
    grad_x = matrix.copy()
    grad_y = matrix.copy()
    for col in range(num_of_cols):
        grad_x[:, col] = tmp[:, col: col + m] @ kernel
    tmp = np.concatenate((matrix[0].reshape(1, -1), matrix), axis=0)
    tmp = np.concatenate((tmp, tmp[-1].reshape(1, -1)), axis=0)
    for row in range(num_of_rows):
        grad_y[row, :] = np.transpose(tmp[row: row + m, :]) @ kernel
    return np.stack((grad_x, grad_y))


def gradient_magnitude(grad):
    num_of_rows = grad.shape[1]
    num_of_cols = grad.shape[2]
    res = np.zeros((num_of_rows, num_of_cols)).astype(np.float)
    for i in range(num_of_rows):
        for j in range(num_of_cols):
            res[i, j] = sqrt(grad[0, i, j] ** 2 + grad[1, i, j] ** 2)
    return res


def normalize(grad):
    res = grad.copy()
    magnitude = gradient_magnitude(grad)
    num_of_rows = grad.shape[1]
    num_of_cols = grad.shape[2]
    for i in range(num_of_rows):
        for j in range(num_of_cols):
            if magnitude[i, j] != 0:
                res[0, i, j] = res[0, i, j] / magnitude[i, j]
                res[1, i, j] = res[1, i, j] / magnitude[i, j]
    return res


def bilinear_interpolate(mat, x, y):
    x1 = np.floor(x).astype(int)
    x2 = x1 + 1
    y1 = np.floor(y).astype(int)
    y2 = y1 + 1

    x1 = min(mat.shape[1] - 1, max(0, x1))
    x2 = min(mat.shape[1] - 1, max(0, x2))
    y1 = min(mat.shape[0] - 1, max(0, y1))
    y2 = min(mat.shape[0] - 1, max(0, y2))

    f11 = mat[y1, x1]
    f12 = mat[y2, x1]
    f21 = mat[y1, x2]
    f22 = mat[y2, x2]

    a = (x2 - x) * (y2 - y)
    b = (x2 - x) * (y - y1)
    c = (x - x1) * (y2 - y)
    d = (x - x1) * (y - y1)

    return a * f11 + b * f12 + c * f21 + d * f22


def f_ot_snake(f, snake):
    res = np.zeros(snake.shape)
    for i in range(snake.shape[0]):
        res[i, 0] = bilinear_interpolate(f[0], snake[i, 0], snake[i, 1])
        res[i, 1] = bilinear_interpolate(f[1], snake[i, 0], snake[i, 1])
    return res


def reparametrization(snake):
    res = np.zeros(snake.shape)
    snake = np.concatenate((snake, snake[0].reshape(1, -1)), axis=0)
    dist = np.cumsum(np.sqrt(np.sum(np.diff(snake, axis=0) ** 2, axis=1)))
    dist = np.insert(dist, 0, 0)

    interpolator = interp1d(dist, snake[:, 0], kind='cubic')
    res[:, 0] = interpolator(np.linspace(0, dist[-1], res.shape[0]))

    interpolator = interp1d(dist, snake[:, 1], kind='cubic')
    res[:, 1] = interpolator(np.linspace(0, dist[-1], res.shape[0]))

    return res


def get_contour_normals(snake):
    res = np.zeros(snake.shape)
    num = 4
    xt = snake[:, 0]
    yt = snake[:, 1]
    n = xt.shape[0]

    a = np.arange(0, n) + num
    a[a > n - 1] -= n
    b = np.arange(0, n) - num
    b[b < 0] += n

    dx = xt[a] - xt[b]
    dy = yt[a] - yt[b]

    for i in range(n):
        length = sqrt(dx[0] ** 2 + dy[0] ** 2)
        res[i, 0] = -dy[i] / length
        res[i, 1] = dx[i] / length
    return res


parser = argparse.ArgumentParser()
parser.add_argument('input_image', type=str)
parser.add_argument('initial_snake', type=str)
parser.add_argument('output_image', type=str)
parser.add_argument('alpha', type=float)
parser.add_argument('beta', type=float)
parser.add_argument('tau', type=float)
parser.add_argument('w_line', type=float)
parser.add_argument('w_edge', type=float)
parser.add_argument('kappa', type=float)

args = parser.parse_args()

initial_snake = np.loadtxt(args.initial_snake)[: -1]
N = initial_snake.shape[0]
int_energy_mat = create_int_energy_mat(args.alpha, args.beta, N)
inverse_mat = np.linalg.inv(np.identity(N) + int_energy_mat * args.tau)

img = imageio.imread(args.input_image).astype(np.float)
filtered_img = gauss(img, SIGMA)
potential_matrix = args.w_line * filtered_img + args.w_edge * gradient_magnitude(gradient(filtered_img)) ** 2
ext_energy_mat = -K * normalize(gradient(potential_matrix))

prev_snake = initial_snake.copy()
new_snake = inverse_mat @ (prev_snake + args.tau * (f_ot_snake(ext_energy_mat, prev_snake) +
                                                    args.kappa * get_contour_normals(prev_snake)))
while mse(new_snake, prev_snake) > EPS:
    prev_snake = reparametrization(new_snake)
    new_snake = inverse_mat @ (prev_snake + args.tau * (f_ot_snake(ext_energy_mat, prev_snake) +
                                                        args.kappa * get_contour_normals(prev_snake)))
utils.save_mask(args.output_image, new_snake, img)
