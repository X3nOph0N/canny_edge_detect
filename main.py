import argparse
from utils import load_fig, show_figs


def main(args):
    fig, fig_type = load_fig(**vars(args))
    show_figs(fig,fig_type, **vars(args))
    return


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--file-name', type=str)
    argparser.add_argument('--filter-type', type=str)
    argparser.add_argument('--output-file-name', type=str, required=False)
    argparser.add_argument('--threshold', type=float)
    args = argparser.parse_args()
    assert(0 < args.threshold < 1)
    main(args)
