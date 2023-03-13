# -*- coding:utf-8 -*-
import os
import re
import codecs

from data_utils import create_dico, create_mapping, zero_digits
from data_utils import iob2, iob_iobes, get_seg_features, iobes_iob


def load_sentences(path, lower=True, zeros=False):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    num = 0
    # codecs.open 用于解决编码问题，推荐使用codecs.open读写文件
    for line in codecs.open(path, 'r', 'utf8'):
        num+=1
        # line = zero_digits(line) if zeros else line       # .rstrip()删除字符串尾部空格? 是否删除\n
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()       # .rstrip()删除字符串尾部空格? 是否删除\n 确认删除\n
        # .strip()删除行首、行尾的空白符 包含'\n' '\t' '\r' ' '  .rstrip()表示上次行尾的
        # zero_digits 实现将所有的数字全变成0，消除数字的影响。默认为False 不操作这步
        # print(list(line))    # 打e 印结果没有\n，代表.rstrip已经去除尾部空格、\n
        if not line:       # 对于这行，如果line是\n空  则会执行，我们将前面所有的sentence中的值保存到sentences中并空置，为下一行做准备
            # print(line, 'i love china!')
            if len(sentence) > 0:
                sentences.append(sentence)
                sentence = []
        else:
            if line[0] == " ":      # 如果出现第一个词为空格，则用$代替。 某些数据不规范
                line = "$" + line[1:]
                word = line.split()
            else:
                word = line.split()
            assert len(word) >= 2, print([word[0]])
            sentence.append(word)
    if len(sentence) > 0:
        sentences.append(sentence)
    return sentences


def update_tag_scheme(sentences, tag_scheme):
    """
    Check and update sentences tagging scheme to IOB2.
    Only IOB1 and IOB2 schemes are accepted.
    """
    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]
        # Check that tags are given in the IOB format
        # iob2 检测是否为标准的BIO格式    若不为标准模式，则输出错误信息
        # tags = iobes_iob(tags)
        if not iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in s)
            raise Exception('Sentences should be given in IOB format! ' +
                            'Please check sentence %i:\n%s' % (i, s_str))
        if tag_scheme == 'iob':
            # If format was IOB1, we convert to IOB2
            # 证实操作word会更新sentences
            for word, new_tag in zip(s, tags):
                word[-1] = new_tag
        elif tag_scheme == 'iobes':
            new_tags = iob_iobes(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        else:
            raise Exception('Unknown tagging scheme!')


def char_mapping(sentences, lower):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    chars = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]
    # 构建字典，记录出现的字符以及出现次数 20191009 读到这里
    dico = create_dico(chars)
    dico["<PAD>"] = 10000001
    dico['<UNK>'] = 10000000
    char_to_id, id_to_char = create_mapping(dico)
    print("Found %i unique words (%i in total)" % (
        len(dico), sum(len(x) for x in chars)
    ))
    return dico, char_to_id, id_to_char


def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = [[char[-1] for char in s] for s in sentences]
    dico = create_dico(tags)
    tag_to_id, id_to_tag = create_mapping(dico)
    print("Found %i unique named entity tags" % len(dico))
    return dico, tag_to_id, id_to_tag


def prepare_dataset(sentences, char_to_id, tag_to_id, lower=False, train=True):
    """
    输出data 用来记录1.句子，2.本句字符在char_to_id的值(字典中的数)，3.分词结果，4.每个字符标签在tag_to_id的值(标签字典中的数)
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - tag indexes
    """

    none_index = tag_to_id["O"]

    def f(x):
        return x.lower() if lower else x
    data = []
    for s in sentences:
        # string 记录数据的第一个字符，20191011
        string = [w[0] for w in s]
        # 如果f(w)在 则记录char不在字典则记录'<UNK>'，返回值为char_to_id 里面的值（数字）
        chars = [char_to_id[f(w) if f(w) in char_to_id else '<UNK>']
                 for w in string]
        segs = get_seg_features("".join(string))
        # segs 记录数据中分词后的结果，单个词用0表示，其他首字1，尾字3，中间2
        # tags 记录tag_to_id 里面的结果，记录sentences里面标签的字典值（数字）
        if train:
            tags = [tag_to_id[w[-1]] for w in s]
        else:
            # 全标记0
            tags = [none_index for _ in chars]
        data.append([string, chars, segs, tags])

    return data


def augment_with_pretrained(dictionary, ext_emb_path, chars):
    """
    Augment the dictionary with words that have a pretrained embedding.
    If `words` is None, we add every word that has a pretrained embedding
    to the dictionary, otherwise, we only add the words that are given by
    `words` (typically the words in the development and test sets.)
    """
    print('Loading pretrained embeddings from %s...' % ext_emb_path)
    assert os.path.isfile(ext_emb_path)

    # Load pretrained embeddings from file
    # pretrained 获得ext_emb_path 中的所有字符。
    pretrained = set([
        line.rstrip().split()[0].strip()
        for line in codecs.open(ext_emb_path, 'r', 'utf-8')
        if len(ext_emb_path) > 0
    ])

    # We either add every word in the pretrained file,
    # or only words given in the `words` list to which
    # we can assign a pretrained embedding
    if chars is None:
        for char in pretrained:
            if char not in dictionary:
                dictionary[char] = 0
    else:
        for char in chars:
            # 前一行判断char， char.lower 和存在一个则为True，char在预训练的字表中，且char 不在字典中，那么在字典中增加这个字符
            if any(x in pretrained for x in [
                char,
                char.lower(),
                re.sub('\d', '0', char.lower())
            ]) and char not in dictionary:
                dictionary[char] = 0

    word_to_id, id_to_word = create_mapping(dictionary)
    return dictionary, word_to_id, id_to_word


def save_maps(save_path, *params):
    """
    Save mappings and invert mappings
    """
    pass
    # with codecs.open(save_path, "w", encoding="utf8") as f:
    #     pickle.dump(params, f)


def load_maps(save_path):
    """
    Load mappings from the file
    """
    pass
    # with codecs.open(save_path, "r", encoding="utf8") as f:
    #     pickle.load(save_path, f)

