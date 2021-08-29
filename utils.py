from cv2 import imread, imshow, namedWindow, createTrackbar
from cv2 import getTrackbarPos, setTrackbarPos, waitKey, imwrite
from cv2 import destroyAllWindows
from numpy import int16, real, ndarray, array, zeros, sqrt, arctan2
from numpy.fft import fft, fft2, ifft2
from scipy.signal import convolve2d
from math import pi, exp


def load_fig(file_name: str, **kwargs) -> ndarray:
    fig = imread(filename=file_name, flags=[-1])
    return fig


def show_figs(fig: ndarray, **kwargs) -> None:

    kernel_size = kwargs['gaussian_kernel_size']
    sigma = kwargs['gaussian_sigma']
    window_name = 'canny edge gradientect algorithm'
    processed_fig = fig.copy()

    def on_val_update() -> None:
        min_val = getTrackbarPos('min_Val', window_name)
        if min_val > getTrackbarPos('max_Val', window_name):
            setTrackbarPos('max_Val', window_name, min_val)
        max_val = getTrackbarPos('max_Val', window_name)
        process_fig(fig, processed_fig, kernel_size, sigma, min_val, max_val)
        return None

    def on_switch_update() -> None:
        if getTrackbarPos('0:OFF\n1:ON') == 1:
            imshow(processed_fig)
        else:
            imshow(fig)
        return

    namedWindow(window_name)
    createTrackbar('min_Val', window_name, 125, 255, on_val_update)
    min_val = getTrackbarPos('min_Val', window_name)
    createTrackbar('max_Val', window_name, min_val, 255, on_val_update)
    createTrackbar('0:OFF\n1:ON', window_name, 0, 1, on_switch_update)

    while(True):
        k = waitKey(1) & 0xFF
        if k == 27:
            break

    if 'output_file_name' in kwargs:
        imwrite(kwargs['output_file_name'], processed_fig)

    destroyAllWindows()

    return


def process_fig(fig: ndarray, target: ndarray, kernel_size: int, sigma: float, min_val: int, max_val: int, **kwargs) -> None:

    filtered_fig = gaussian_filter(fig, kernel_size, sigma)
    gradient, theta = calculate_gradient(filtered_fig)
    regression_fig = non_max_regression(gradient, theta)
    target = double_threshold_process(regression_fig, min_val, max_val)
    return


def gaussian_filter(fig: ndarray, kernel_size: int, sigma: float) -> ndarray:

    assert(kernel_size > 1)

    gaussian_kernel = ndarray([kernel_size, kernel_size])
    center = (kernel_size-1)/2
    for x in range(kernel_size):
        gaussian_kernel[x] = array(
            [(1/(2*pi*sigma**2)*exp(-((x-center)**2+(y-center)**2)/(2*sigma**2)))
             for y in range(kernel_size)])
    gaussian_kernel /= gaussian_kernel.sum()

    filtered_fig = convolve2d(fig, gaussian_kernel,
                              mode='same', boundary='fill', fillvalue=0)

    return filtered_fig.astype(int16)


def calculate_gradient(fig: ndarray) -> tuple(ndarray, ndarray):

    gradient = ndarray(fig.shape)
    theta = ndarray(fig.shape)

    operator_h = ndarray([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]])
    operator_v = ndarray([[-1, -2, -1],
                          [0,  0,  0],
                          [1,  2,  1]])

    kernel_h = zeros(fig.shape)
    kernel_h[:operator_h.shape[0], :operator_h.shape[1]] = operator_h
    kernel_h = fft2(kernel_h)
    kernel_v = zeros(fig.shape)
    kernel_v[:operator_v.shape[0], :operator_v.shape[1]] = operator_v
    kernel_v = fft(kernel_v)

    fim = fft2(fig)
    gradient_x = real(ifft2(kernel_h * fim)).astype(float)
    gradient_y = real(ifft2(kernel_v * fim)).astype(float)

    gradient = sqrt(gradient_x**2 + gradient_y**2)
    theta = arctan2(gradient_y, gradient_x) * 180 / pi

    return gradient, theta


def non_max_regression(gradient: ndarray, theta: ndarray) -> ndarray:
    regression_fig = ndarray(gradient.shape)
    for i in range(regression_fig.shape[0]):
        for j in range(regression_fig.shape[1]):
            if theta[i][j] < 0:
                theta[i][j] += 360

            if ((j+1) < regression_fig.shape[1]) and ((j-1) >= 0) and ((i+1) < regression_fig.shape[0]) and ((i-1) >= 0):
                if (theta[i][j] >= 337.5 or theta[i][j] < 22.5) or (theta[i][j] >= 157.5 and theta[i][j] < 202.5):
                    if gradient[i][j] >= gradient[i][j + 1] and gradient[i][j] >= gradient[i][j - 1]:
                        regression_fig[i][j] = gradient[i][j]
                if (theta[i][j] >= 22.5 and theta[i][j] < 67.5) or (theta[i][j] >= 202.5 and theta[i][j] < 247.5):
                    if gradient[i][j] >= gradient[i - 1][j + 1] and gradient[i][j] >= gradient[i + 1][j - 1]:
                        regression_fig[i][j] = gradient[i][j]
                if (theta[i][j] >= 67.5 and theta[i][j] < 112.5) or (theta[i][j] >= 247.5 and theta[i][j] < 292.5):
                    if gradient[i][j] >= gradient[i - 1][j] and gradient[i][j] >= gradient[i + 1][j]:
                        regression_fig[i][j] = gradient[i][j]
                if (theta[i][j] >= 112.5 and theta[i][j] < 157.5) or (theta[i][j] >= 292.5 and theta[i][j] < 337.5):
                    if gradient[i][j] >= gradient[i - 1][j - 1] and gradient[i][j] >= gradient[i + 1][j + 1]:
                        regression_fig[i][j] = gradient[i][j]
    return regression_fig


def double_threshold_process(gradient: ndarray, theta: ndarray, min_val: int, max_val: int) -> ndarray:
    result = ndarray(gradient.shape)
    # TODO
    return result
