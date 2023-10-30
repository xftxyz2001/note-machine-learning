import re


class ToolGeneral():
    """
    Tool function
    """

    def load_dict(self, file):
        """
        Load dictionary
        """
        with open(file, encoding='utf-8', errors='ignore') as fp:
            lines = fp.readlines()
            lines = [l.strip() for l in lines]
            print("Load data from file (%s) finished !" % file)
            dictionary = [word.strip() for word in lines]
        return set(dictionary)

    # 句子分割
    def sentence_split_regex(self, sentence):
        if sentence is not None:
            sentence = re.sub(r"&ndash;+|&mdash;+", "-", sentence)
            sub_sentence = re.split(
                r"[。,，！!？?;；\s…~～]+|\.{2,}|&hellip;+|&nbsp+|_n|_t", sentence)
            sub_sentence = [s for s in sub_sentence if s != '']
            if sub_sentence != []:
                return sub_sentence
            else:
                return [sentence]
        return []


if __name__ == "__main__":
    #
    tool = ToolGeneral()
    #
    s = '我今天。昨天上午，还有现在'
    ls = tool.sentence_split_regex(s)
    print(ls)
