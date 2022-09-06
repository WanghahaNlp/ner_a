# encoding=utf-8
import codecs
import jieba
import json
import numpy as np
import os
import re
from loguru import logger


def load_sentences(path, lower: bool=True, zeros: bool=True):
    """读取数据集，是否大写字母转成小写字母，数字是否都替换为0"""
    sentences = []
    sentence = []
    num = 0
    for line in codecs.open(path, "r", "utf8"):
        num += 1
        # 数字是否替换为0
        line = zero_digits(line.strip()) if zeros else line.rstrip()
        if line is None or line == "":
            if len(sentence) == 0:
                continue
            sentences.append(sentence)
            sentence = []
        else:
            if line[0] == " ":
                cell = line.strip().split()
                assert len(cell) != 1, logger.error(f"标注有问题：{line}")
                sentence.append(["$", cell[0]])
            else:
                cell = line.strip().split()
                assert len(cell) == 2, logger.error(f"标注有问题：{line}")
                # 英文是否转换成小写
                sentence.append([cell[0].lower() if lower else cell[0], cell[1]])
    return sentences


def update_tag_schema(sentences: list, tag_schema):
    for ind, cell in enumerate(sentences):
        tags = [i[1] for i in cell]
        if not iob(tags):
            s_str = "\n".join(" ".join(i) for i in cell)
            raise Exception(f"标签标注错误，请检查{ind}\n{s_str}")

        if tag_schema == "iob":
            for word, new_tag in zip(cell, tags):
                word[-1] = new_tag
        elif tag_schema == "iobes":
            new_tags = iob_iobes(tags)
            for word, new_tag in zip(cell, new_tags):
                word[-1] = new_tag
        else:
            raise Exception("\"tag_schema\" in [\"iob\", \"iobes\"]")


def create_dico(sentences):
    """统计词频"""
    assert type(sentences) is list
    dico = {}
    for line in sentences:
        for word in line:
            dico[word] = dico.get(word, 0) + 1
    return dico


def create_mapping(dico):
    """按照词频排序，创建id:word 和 word:id"""
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {j: i for i, j in id_to_item.items()}
    return item_to_id, id_to_item


def char_mapping(sentences):
    # 生成序列
    sequences = [[cell[0] for cell in line] for line in sentences]
    # 统计词频
    dico = create_dico(sequences)
    dico["<PAD>"] = int(1e+7 + 1)
    dico["<UNK>"] = int(1e+7)
    char_to_id, id_to_char = create_mapping(dico)
    logger.debug(f"找到{len(dico)}个独特的单词，总数为{sum(len(i) for i in sequences)}个")
    return dico, char_to_id, id_to_char


def augment_with_pretrained(dictionary, ext_emb_path, chars):
    logger.debug(f"加载来自{ext_emb_path}的预训练的embedding...")
    assert os.path.isfile(ext_emb_path)
    pretrained = set([line.rstrip().split()[0].strip() for line in codecs.open(ext_emb_path, "r", "utf-8") if len(line.rstrip()) > 0])

    if chars is None:
        for char in pretrained:
            if char not in dictionary:
                dictionary[char] = 0
    else:
        for char in chars:
            # 如果测试序列中字或词在预训练的向量中 和 字或词 不在训练集中 
            if any(x in pretrained for x in [
                char,
                char.lower(),
                re.sub('\d', '0', char.lower())
            ]) and char not in dictionary:
                # 在词典中添加 字:数量0
                dictionary[char] = 0

    word_to_id, id_to_word = create_mapping(dictionary)
    return dictionary, word_to_id, id_to_word


def tag_mapping(sentences):
    """统计标签词频"""
    tags = [[char[-1] for char in s] for s in sentences]
    dico = create_dico(tags)
    tag_to_id, id_to_tag = create_mapping(dico)
    logger.debug(f"找到{len([i for i in dico.keys() if 'B' in i])}个唯一的实体标签")
    return dico, tag_to_id, id_to_tag


def get_seg_features(string):
    """序列采用jieba分词：特征以bies格式表示 B,BIE,BIIIE,BE-> 0, 123, 12223, 13"""
    seg_feature = []

    for word in jieba.cut(string):
        if len(word) == 1:
            seg_feature.append(0)
        else:
            tmp = [2] * len(word)
            tmp[0] = 1
            tmp[-1] = 3
            seg_feature.extend(tmp)
    return seg_feature


def prepare_dataset(sentences, char_to_id, tag_to_id, lower: bool=True, train: bool=True):
    none_index = tag_to_id["O"]

    def f(x):
        return x.lower() if lower else x

    data = []
    for s in sentences:
        string = [w[0] for w in s]
        chars = [char_to_id[f(w) if f(w) in char_to_id else "<UNK>"] for w in string]
        # 数据转换成bies形式
        segs = get_seg_features("".join(string))
        if train:
            tags = [tag_to_id[w[-1]] for w in s]
        else:
            tags = [none_index for _ in chars]
        data.append([string, chars, segs, tags])
    return data


def load_word2vec(emb_path, id_to_word, word_dim, old_weights):
    """读取预训练字向量"""
    new_weights = old_weights
    logger.debug(f"读取预训练向量地址来自{emb_path}")
    pre_trained = {}
    emb_invalid = 0
    for i, line in enumerate(codecs.open(emb_path, 'r', 'utf-8')):
        if i == 0:
            continue
        line = line.rstrip().split()
        if len(line) == word_dim + 1:
            pre_trained[line[0]] = np.array(
                [float(x) for x in line[1:]]
            ).astype(np.float32)
        else:
            emb_invalid += 1
    if emb_invalid > 0:
        logger.warning(f"{emb_invalid}个有问题的向量")
    c_found = 0
    c_lower = 0
    c_zeros = 0
    n_words = len(id_to_word)
    for i in range(n_words):
        word = id_to_word[i]
        if word in pre_trained:
            new_weights[i] = pre_trained[word]
            c_found += 1
        elif word.lower() in pre_trained:
            new_weights[i] = pre_trained[word.lower()]
            c_lower += 1
        elif re.sub('\d', '0', word.lower()) in pre_trained:
            new_weights[i] = pre_trained[
                re.sub('\d', '0', word.lower())
            ]
            c_zeros += 1
    logger.debug(f"读取{len(pre_trained)}个embedding")
    logger.debug(f"{c_found + c_lower + c_zeros}/{n_words} "
        f"({round(100. * (c_found + c_lower + c_zeros) / n_words, 4)}%)的word被初始化")
    logger.debug(f"直接发现: {c_found}")
    logger.debug(f"小写的: {c_lower}")
    logger.debug(f"小写的+0: {c_zeros}")
    return new_weights


def zero_digits(s):
    """数字替换成0"""
    return re.sub("\d", "0", s)


def iobes_iob(tags):
    """IOBES -> IOB"""
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.split('-')[0] == 'B':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'I':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'S':
            new_tags.append(tag.replace('S-', 'B-'))
        elif tag.split('-')[0] == 'E':
            new_tags.append(tag.replace('E-', 'I-'))
        elif tag.split('-')[0] == 'O':
            new_tags.append(tag)
        else:
            raise Exception('Invalid format!')
    return new_tags


def iob(tags):
    """bio检测"""
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:
            tags[i] = 'B' + tag[1:]
    return True
        
        
def iob_iobes(tags):
    """IOB -> IOBES"""
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
               tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags


def make_path(params):
    """创建文件夹"""
    if not os.path.isdir(params.result_path):
        os.makedirs(params.result_path)
    if not os.path.isdir(params.ckpt_path):
        os.makedirs(params.ckpt_path)
    if not os.path.isdir("log"):
        os.makedirs("log")

    map_file = "/".join(params.map_file.split("/")[:-1])
    if not os.path.isdir(map_file):
        os.makedirs(map_file)


def save_model(sess, model, path):
    checkpoint_path = os.path.join(path, "ner.ckpt")
    model.saver.save(sess, checkpoint_path)
    logger.info("模型存储成功~！")


def result_to_json(string, tags):
    print("--------------------")
    print(string)
    print(tags)
    print("--------------------")
    item = {"string": string, "entities": []}
    entity_name = ""
    entity_start = 0
    idx = 0
    for char, tag in zip(string, tags):
        if tag[0] == "S":
            item["entities"].append({"word": char, "start": idx, "end": idx + 1, "type": tag[2:]})
        elif tag[0] == "B":
            entity_name += char
            entity_start = idx
        elif tag[0] == "I":
            entity_name += char
        elif tag[0] == "E":
            entity_name += char
            item["entities"].append({"word": entity_name, "start": entity_start, "end": idx + 1, "type": tag[2:]})
            entity_name = ""
        else:
            entity_name = ""
            entity_start = idx
        idx += 1
    return item


def full_to_half(s):
    n = []
    for char in s:
        num = ord(char)
        if num == 0x3000:
            num = 32
        elif 0xFF01 <= num <= 0xFF5E:
            num -= 0xfee0
        char = chr(num)
        n.append(char)
    return ''.join(n)


def replace_html(s):
    s = s.replace('&quot;','"')
    s = s.replace('&amp;','&')
    s = s.replace('&lt;','<')
    s = s.replace('&gt;','>')
    s = s.replace('&nbsp;',' ')
    s = s.replace("&ldquo;", "“")
    s = s.replace("&rdquo;", "”")
    s = s.replace("&mdash;","")
    s = s.replace("\xa0", " ")
    return(s)


def input_from_line(line, char_to_id):
    line = full_to_half(line)
    line = replace_html(line)
    inputs = list()
    inputs.append([line])
    line.replace(" ", "$")
    inputs.append([[char_to_id[char] if char in char_to_id else char_to_id["<UNK>"]
                   for char in line]])
    inputs.append([get_seg_features(line)])
    inputs.append([[]])
    return inputs


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f,ensure_ascii=False, indent=4)


def print_config(config, annotation="config"):
    logger.debug("-"*10 + annotation + "-"*10)
    for k, v in config.items():
        logger.debug("{}:\t{}".format(k.ljust(15), v))
    logger.debug("-"*40)
