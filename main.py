import argparse
from utils import load_fig, show_figs


def main(args):
    fig = load_fig(**vars(args))
    show_figs(fig, **vars(args))
    return


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--file-name', type=str)
    argparser.add_argument('--output-file-name', type=str, required=False)
    argparser.add_argument('--gaussian-kernel-size',type=int,default =5)
    argparser.add_argument('--gaussian-sigma',type=float,default=1)
    args = argparser.parse_args()
    main(args)
