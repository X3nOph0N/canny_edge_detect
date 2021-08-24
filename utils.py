import cv2 as cv
from cv2 import imread, imshow
from numpy import ndarray


def load_fig(file_name: str, **kwargs) -> tuple(ndarray, str):
    fig = imread(filename=file_name, flags=[-1])
    fig_type = 'GRB' if len(fig.shape) > 2 else 'GREY'
    return fig, fig_type


def show_figs(fig: ndarray, **kwargs) -> None:

    window_name = 'canny edge detect algorithm'
    processed_fig = fig.copy()

    def on_val_update():
        min_val = cv.getTrackbarPos('min_Val', window_name)
        if min_val > cv.getTrackbarPos('max_Val', window_name):
            cv.setTrackbarPos('max_Val', window_name, min_val)
        processed_fig = process_fig(fig)
        return

    def on_switch_update():
        if cv.getTrackBarPos('0:OFF\n1:ON') == 1:
            cv.imshow(processed_fig)
        else:
            cv.imshow(fig)
        return

    cv.namedWindow(window_name)
    cv.createTrackbar('min_Val', window_name, 125, 255, on_val_update)
    min_val = cv.getTrackbarPos('min_Val', window_name)
    cv.createTrackbar('max_Val', window_name, min_val, 255, on_val_update)
    cv.createTrackbar('0:OFF\n1:ON', window_name, 0, 1, on_switch_update)

    while(True):
        k = cv.waitKey(1) & 0xFF
        if k == 27:
            break
    # TODO well I hear that this function may lead to a failure for unable to update picture, I will do that later
    cv.destroyAllWindows()

    return


def process_fig(fig: ndarray, fig_type: str, **kwargs) -> ndarray:
    # TODO
    return


def canny_edge_detect(**kwargs):
    # TODO
    return
