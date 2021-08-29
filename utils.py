from cv2 import imread, imshow, namedWindow, createTrackbar
from cv2 import getTrackbarPos, setTrackbarPos, waitKey, imwrite
from cv2 import destroyAllWindows
from numpy import int16, real, ndarray, array, zeros, sqrt, arctan2
from numpy.fft import fft, fft2, ifft2
from scipy.signal import convolve2d
from math import pi, exp


def load_fig(input_file_name: str, **kwargs) -> ndarray:
    """
    This is the function who will load the picture.
    @input:
        file_name:a string indicates where the figure is.
    @output:
        fig: a figure in 2-dimension matrix form.
        (Only grayscale figure supported now)
    """
    fig = imread(filename=input_file_name, flags=0)
    return fig


def show_figs(fig: ndarray,gaussian_kernel_size:int,gaussian_sigma:float, **kwargs) -> None:
    """
    This is the function define how window interact to user and
    how the origin figure will be shown.
    @input:
        fig: a figure in 2-dimensional matrix form.
        gaussian_kernel_size: a integer indicates the shape of the gaussian smoothing kernel.
        gaussian_sigma: a float used in gaussian smoothing, a bigger number means more close to average smoothing.
        (optional) output_file_name:this is a string containing the path where the program should storage the processed figure.
            If given the program will automatically save the figure in the path given or thr program will ignore it.
    @output:
        None
    """
    kernel_size = gaussian_kernel_size
    sigma = gaussian_sigma

    window_name = 'canny edge detect algorithm'

    # Define the callbacks of different controls seperately to import experience 
    # and make it more clear.
    def on_max_val_update(*args, **kwargs) -> None:
        max_val = getTrackbarPos('max_Val', window_name)
        if max_val < getTrackbarPos('min_Val', window_name):
            setTrackbarPos('min_Val', window_name, max_val)
        setTrackbarPos('0:OFF 1:ON', window_name, 0)
        imshow(window_name, fig)
        return None

    def on_min_val_update(*args, **kwargs) -> None:
        min_val = getTrackbarPos('min_Val', window_name)
        if min_val > getTrackbarPos('max_Val', window_name):
            setTrackbarPos('max_Val', window_name, min_val)
        setTrackbarPos('0:OFF 1:ON', window_name, 0)
        imshow(window_name, fig)
        return None

    def on_switch_update(*args, **kwargs) -> None:
        if getTrackbarPos('0:OFF 1:ON', window_name) == 1:
            processed_fig = process_fig(fig, kernel_size, sigma, getTrackbarPos(
                'min_Val', window_name), getTrackbarPos('max_Val', window_name))
            imshow(window_name, processed_fig)
        else:
            imshow(window_name, fig)
        return

    namedWindow(window_name)
    createTrackbar('min_Val', window_name, 125, 255, on_min_val_update)
    min_val = getTrackbarPos('min_Val', window_name)
    createTrackbar('max_Val', window_name, min_val, 255, on_max_val_update)
    createTrackbar('0:OFF 1:ON', window_name, 0, 1, on_switch_update)
    imshow(window_name, fig)
    while(True):
        k = waitKey(1) & 0xFF
        if k == 27:
            break

    processed_fig = process_fig(fig, kernel_size, sigma, getTrackbarPos(
        'min_Val', window_name), getTrackbarPos('max_Val', window_name))
    if 'output_file_name' in kwargs:
        imwrite(kwargs['output_file_name'], processed_fig)

    destroyAllWindows()

    return


def process_fig(fig: ndarray, kernel_size: int, sigma: float, min_val: int, max_val: int, **kwargs) -> ndarray:
    """
    This is the function containing the main framework in canny edge detect.
    @input:
        fig: a fig in 2-dimensional matrix form
        kernel_size: a integer indicates the shape of the gaussian smoothing kernel.
        sigma: a float used in gaussian smoothing, a bigger number means more close to average smoothing.
        min_val: a integer indicates that all edge with gradient under it should be discarded.
        max-val: a integer indicates that all edge with gradient above it should be preserved.
    @output:
        target: the processed figure in 2-dimensional matrix form.
    """
    filtered_fig = gaussian_filter(fig, kernel_size, sigma)
    gradient, theta = calculate_gradient(filtered_fig)
    regression_fig = non_max_regression(gradient, theta)
    target = double_threshold_process(regression_fig, min_val, max_val)
    return target


def gaussian_filter(fig: ndarray, kernel_size: int, sigma: float) -> ndarray:
    """
    This is the function that do gaussian smoothing with given parameters.
    @input:
        fig: a figure in 2-dimensional matrix form
        kernel_size:a integer indicates the shape of the gaussian smoothing kernel.
        sigma: a float used in gaussian smoothing, a bigger number means more close to average smoothing.
    @output:
        filtered_fig: the figure after gaussain smoothing process in 2-dimensional matrix form.
    """
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


def calculate_gradient(fig: ndarray) -> tuple:
    """
    This is hte function calculating the gradient using sobel operator.
    @input:
        fig: a figure in 2-dimensional matrix form.
    @output:
        gradient: the absolute value of gradient in matrix form
        theta: the direction of gradient in angel form, all elements varing in [-180,180]    
    """
    gradient = ndarray(fig.shape)
    theta = ndarray(fig.shape)

    operator_h = array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    operator_v = array([[-1, -2, -1],
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
    """
    This is the function where non max regression process.
    @input:
        gradient: the absolute value of gradient in matrix form
        theta: the direction of gradient in angel form, all elements varing in [-180,180]
    @output:
        regression-fig: the figure after non max regression process.
    """
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


def double_threshold_process(gradient: ndarray, min_val: int, max_val: int) -> ndarray:
    """
    This is the function containing the double threshold process.
    @input:
        gradient: the absolute value of gradient in matrix form
        min_val: a integer indicates that all edge with gradient under it should be discarded.
        max-val: a integer indicates that all edge with gradient above it should be preserved.
    @output:
        result:  the figure after all process of canny edge detect. 
        """
    result = ndarray(gradient.shape)
    strong = 255
    weak = 0
    centre = 126

    def find_local_edge(i: int, j: int) -> int:
        result = weak
        for x in range(3):
            for y in range(3):
                if i+x-1 > 0 and j + y - 1 > 9:
                    if gradient[i+x-1][j+y-1] > max_val:
                        result = centre
        return result

    for i in range(gradient.shape[0]):
        for j in range(gradient.shape[1]):
            px = gradient[i][j]
            if px >= max_val:
                result[i][j] = strong
            elif px <= min_val:
                result[i][j] = weak
            else:
                result[i, j] = find_local_edge(i, j)

    return result
