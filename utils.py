from cv2 import imread,imshow,namedWindow,createTrackbar
from cv2 import getTrackbarPos,setTrackbarPos,waitKey,imwrite
from cv2 import destroyAllWindows
from numpy import int16, ndarray,array
from scipy.signal import convolve2d
from math import pi,exp

def load_fig(file_name: str, **kwargs) -> ndarray:
    fig = imread(filename=file_name, flags=[-1])
    return fig


def show_figs(fig: ndarray, **kwargs) -> None:

    kernel_size = kwargs['gaussian_kernel_size']
    sigma = kwargs['gaussian_sigma']
    window_name = 'canny edge detect algorithm'
    processed_fig = fig.copy()

    def on_val_update()->None:
        min_val = getTrackbarPos('min_Val', window_name)
        if min_val > getTrackbarPos('max_Val', window_name):
            setTrackbarPos('max_Val', window_name, min_val)
        max_val = getTrackbarPos('max_Val',window_name)
        process_fig(fig,processed_fig,kernel_size,sigma,min_val,max_val)
        return None

    def on_switch_update()-> None:
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
        imwrite(kwargs['output_file_name'],processed_fig)
    
    destroyAllWindows()

    return


def process_fig(fig: ndarray, target: ndarray,kernel_size:int,sigma:float,min_val:int,max_val:int, **kwargs) -> None:
    filtered_fig = gaussian_filter(fig)
    gradient_x,gradient_y = calculate_gradient(filtered_fig)
    regressioned_fig = non_max_regression(filtered_fig,gradient_x,gradient_y)
    target = double_threshold_process(regressioned_fig,min_val,max_val)
    return


def gaussian_filter(fig:ndarray,kernel_size:int,sigma:float)->ndarray:
    
    assert(kernel_size>1)

    gaussian_kernel = ndarray([kernel_size,kernel_size])
    center = (kernel_size-1)/2
    for x in range(kernel_size):
        gaussian_kernel[x] = array(
            [(1/(2*pi*sigma**2)*exp(-((x-center)**2+(y-center)**2)/(2*sigma**2)))
             for y in range(kernel_size)])
    gaussian_kernel/=gaussian_kernel.sum()
    
    filtered_fig = convolve2d(fig,gaussian_kernel,mode='same',boundary='fill',fillvalue=0)
    
    return filtered_fig.astype(int16)

def calculate_gradient(fig:ndarray)->tuple(ndarray,ndarray):
    return

def non_max_regression(fig:ndarray,gradient_x:ndarray,gradient_y:ndarray)->ndarray:
    return

def double_threshold_process(fig:ndarray,min_val:int,max_val:int)->ndarray:
    return