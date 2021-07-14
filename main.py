from experiment import SegInprintExp
from nntools.utils import Config

if __name__ == '__main__':
    c = Config('config.yaml')
    exp = SegInprintExp(c, '32b7e1a3c36c44d0b1848aff07e8d59a')
    exp.eval()