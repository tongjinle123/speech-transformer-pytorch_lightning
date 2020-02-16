

def is_chinese(char):
    if char >= '\u4e00' and char <= '\u9fa5':
        return True
    else:
        return False


def is_english(char):
    if (char >= '\u0041' and char <= '\u005a') or (char >= '\u0061' and char <= '\u007a') or char in ['[', ']', '@', '~', '\'', '-']:
        return True
    else:
        return False


def is_number(char):
    if char >= '\u0030' and char <= '\u0039':
        return True
    else:
        return False


def strQ2B(ustring):
    ss = []
    for s in ustring:
        rstring = ""
        for uchar in s:
            inside_code = ord(uchar)
            if inside_code == 12288:  # 全角空格直接转换
                inside_code = 32
            elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
                inside_code -= 65248
            rstring += chr(inside_code)
        ss.append(rstring)
    return ss

def tokenize(sentence, lower=False):
    """
    cut by chinese char, number, english word
    :param sentence:
    :return:
    """
    sentence = strQ2B(sentence)
    ret = []
    temp = ''
    for index, char in enumerate(sentence):
        is_e = is_english(char)
        is_b = temp == ''

        if is_e and is_b:
            temp += char
            continue

        elif is_e and not is_b:
            temp += char
            if temp in ['[N]', '[S]', '[T]', '[P]']:
                ret.append(temp)
                temp = ''
            if temp[-3:] in ['[N]', '[S]', '[T]', '[P]'] and len(temp) > 3:
                ret.append(temp[:-3])
                ret.append(temp[-3:])
                temp = ''
            continue

        elif not is_e and not is_b:
            ret.append(temp)
            temp = ''

        elif not is_e and is_b:
            pass

        if is_chinese(char) or is_number(char):
            ret.append(char)

    if temp != '':
        ret.append(temp)
    nret = []
    for i in ret:
        if i.startswith('@'):
            nret.append(i[1:])
        elif i.startswith('~'):
            nret.extend([i[1:]])
        else:
            nret.append(i)
    if lower:
        nret = [i.lower() for i in nret]
    nret = [i for i in nret if i not in ['，', '。', ""]]
    return nret

def combine(tokenized):
    new_str = ''
    for i in tokenized:
        if i[0] == '[' and i[2] == ']':
            new_str += i
            continue
        if is_chinese(i[0]):
            new_str += i
        elif is_english(i[0]):
            new_str += ' '+i
    return new_str


if __name__ == '__main__':
    s = '[S]我有一个iphone7p,你有[n],me[T]'
    print(tokenize(s))