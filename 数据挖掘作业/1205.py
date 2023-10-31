from pprint import pprint

# Python 版本


def levenshtein_distance(string1, string2):
    m = len(string1)
    n = len(string2)
    d = [[0] * (n + 1) for _ in range(m + 1)]
    # 初始化第 1 列
    for i in range(m + 1):
        d[i][0] = i
    # 初始化第 1 行
    for j in range(n + 1):
        d[0][j] = j
    # 自底向上递推计算每个 d[i][j] 的值
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if string1[i - 1] == string2[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1]) + 1
    pprint(d)
    return d[m][n]


# S1=XGYXYYX，S2=XYXYXTX
levenshtein_distance('XGYXYYX', 'XYXYXTX')
