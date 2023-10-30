import os

from utils import ToolGeneral

pwd = os.path.dirname(os.path.abspath(__file__))
tool = ToolGeneral()


class Hyperparams:
    '''Hyper parameters'''
    # 加载情感字典
    deny_word = tool.load_dict(os.path.join(pwd, 'dict', 'not.txt'))
    posdict = tool.load_dict(os.path.join(pwd, 'dict', 'positive.txt'))
    negdict = tool.load_dict(os.path.join(pwd, 'dict', 'negative.txt'))
    pos_neg_dict = posdict | negdict
    # 加载副词字典
    mostdict = tool.load_dict(os.path.join(pwd, 'dict', 'most.txt'))
    verydict = tool.load_dict(os.path.join(pwd, 'dict', 'very.txt'))
    moredict = tool.load_dict(os.path.join(pwd, 'dict', 'more.txt'))
    ishdict = tool.load_dict(os.path.join(pwd, 'dict', 'ish.txt'))
    insufficientlydict = tool.load_dict(
        os.path.join(pwd, 'dict', 'insufficiently.txt'))
    overdict = tool.load_dict(os.path.join(pwd, 'dict', 'over.txt'))
    inversedict = tool.load_dict(os.path.join(pwd, 'dict', 'inverse.txt'))
