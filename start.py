# encoding=utf-8

import argparse
import math
import itertools
import os
import pickle
import random
import numpy as np
import tensorflow as tf

from collections import OrderedDict
from data_utils import *
from data_dispose import evaluate
from model import Model


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default="./data/example.train", type=str, help="train数据集地址")
    parser.add_argument("--dev_file", default="./data/example.dev", type=str, help="dev数据集地址")
    parser.add_argument("--test_file", default="./data/example.test", type=str, help="test数据集地址")

    parser.add_argument("--map_file", default="./ckpt/maps.pkl", type=str, help="pkl文件地址")
    parser.add_argument("--emb_file", default="./data/vec.txt", type=str, help="embedding文件地址")
    parser.add_argument("--config_file", default="./ckpt/config_file", type=str, help="embedding文件地址")

    parser.add_argument("--ckpt_path", default="./ckpt", type=str, help="embedding文件地址")
    parser.add_argument("--result_path", default="./result", type=str, help="dev or test 预测结果存储地址")

    parser.add_argument("--lower", default=True, type=bool, help="英文是否转换成小写字母")
    parser.add_argument("--zeros", default=True, type=bool, help="数字是否转换成0")
    parser.add_argument("--pre_emb", default=True, type=bool, help="是否使用预训练向量")

    parser.add_argument("--batch_size", default=20, type=int, help="batch size")
    parser.add_argument("--char_dim", default=100, type=int, help="字符集的embedding size")
    parser.add_argument("--lr", default=1e-3, type=float, help="初始学习率")
    parser.add_argument("--clip", default=5, type=int, help="梯度裁剪")
    parser.add_argument("--dropout", default=0.5, type=float, help="随机失活")
    parser.add_argument("--seg_dim", default=20, type=int, help="embedding size 进行分割，0 则不使用")
    parser.add_argument("--filter_dim", default=100, type=int, help="滤波器的宽度")
    parser.add_argument("--steps_check", default=100, type=int, help="检查点，每steps_check步检查一次")

    parser.add_argument("--optimizer", default="adam", type=str, help="优化器")

    parser.add_argument("--tag_schema", default="iobes", type=str, help="标注格式iob or iobes")
    parser.add_argument("--gpu_proportion", default=1, type=float, help="gpu的最大使用比例，0-1区间按百分比使用；1：全部使用")

    args = parser.parse_args()
    
    assert args.clip < 5.1, "渐变剪辑不应该太大"
    assert 0 < args.dropout < 1, "随机失活应该在(0, 1)区间"
    assert args.lr > 0, "学习率必须大于0"
    assert args.optimizer in ["adam", "sgd", "adagrad"], "优化器应属于这些其中的一个"
    return args


def config_model(args, char_to_id, tag_to_id):
    config = OrderedDict()
    # config["model_type"] = args.model_type
    config["num_chars"] = len(char_to_id)
    config["num_tags"] = len(tag_to_id)

    config["batch_size"] = args.batch_size
    config["char_dim"] = args.char_dim
    config["lr"] = args.lr
    config["clip"] = args.clip
    config["dropout_keep"] = 1.0 - args.dropout

    config["optimizer"] = args.optimizer

    config["emb_file"] = args.emb_file
    config["pre_emb"] = args.pre_emb
    config["zeros"] = args.zeros
    config["lower"] = args.lower
    config["seg_dim"] = args.seg_dim
    config["filter_dim"] = args.filter_dim
    config["tag_schema"] = args.tag_schema
    return config


class BatchManager(object):

    def __init__(self, data,  batch_size):
        self.batch_data = self.sort_and_pad(data, batch_size)
        self.len_data = len(self.batch_data)

    def sort_and_pad(self, data, batch_size):
        """将数据安装序列长度进行排序，升序"""
        # 向右取整
        num_batch = int(math.ceil(len(data) / batch_size))
        sorted_data = sorted(data, key=lambda x: len(x[0]))
        batch_data = list()
        for i in range(num_batch):
            batch_data.append(self.pad_data(sorted_data[int(i*batch_size) : int((i+1)*batch_size)]))
        return batch_data

    @staticmethod
    def pad_data(data):
        """保证每个批次数据长度统一"""
        strings = []
        chars = []
        segs = []
        targets = []
        max_length = max([len(sentence[0]) for sentence in data])
        for line in data:
            string, char, seg, target = line
            padding = [0] * (max_length - len(string))
            strings.append(string + padding)
            chars.append(char + padding)
            segs.append(seg + padding)
            targets.append(target + padding)
        return [strings, chars, segs, targets]

    def iter_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]


def create_model(sess, args, config, id_to_char, is_train=True):
    # 初始化tensor
    model = Model(config, is_train)
    # 获取检查点，是否有模型文件
    ckpt = tf.train.get_checkpoint_state(args.ckpt_path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        logger.debug(f"从{ckpt.model_checkpoint_path}读取模型")
        model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        logger.debug("创建一个新的模型和新的参数")
        sess.run(tf.compat.v1.global_variables_initializer())

        # 采用预训练词向量进行替换
        if config["pre_emb"]:
            emb_weights = sess.run(model.char_lookup.read_value())
            emb_weights = load_word2vec(config["emb_file"], id_to_char, config["char_dim"], emb_weights)
            sess.run(model.char_lookup.assign(emb_weights))
            logger.debug("读取预训练的embedding完成")
    return model


def del_model(args):
    if input("是否删除模型所有的文件(0/1):") == "1":
        os.system(f"rm -rf {args.ckpt_path} {args.result_path} log/ ")


def train(args):
    """训练"""
    del_model(args)
    # 读取数据集
    train_sentences = load_sentences(args.train_file, args.lower, args.zeros)
    dev_sentences = load_sentences(args.dev_file, args.lower, args.zeros)
    test_sentences = load_sentences(args.test_file, args.lower, args.zeros)

    # 数据处理BIEO
    update_tag_schema(train_sentences, args.tag_schema)
    update_tag_schema(dev_sentences, args.tag_schema)
    update_tag_schema(test_sentences, args.tag_schema)

    make_path(args)
    # 文件地址是否存在
    if not os.path.isfile(args.map_file):
        dico_chars_train, char_to_id, id_to_char = char_mapping(train_sentences)
        if args.pre_emb:
            # 词频字典
            dico_chars, char_to_id, id_to_char = augment_with_pretrained(
                dico_chars_train.copy(),
                args.emb_file,
                list(
                    itertools.chain.from_iterable(
                        [[w[0] for w in s] for s in test_sentences]
                    )
                )
            )
        # 为标记创建字典映射
        _t, tag_to_id, id_to_tag = tag_mapping(train_sentences)
        with open(args.map_file, "wb") as f:
            pickle.dump([char_to_id, id_to_char, tag_to_id, id_to_tag], f)
    else:
        with open(args.map_file, "rb") as f:
            char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)

    # 每条数据进行拆解成字、字索引、结巴分词（0,13,123,1223）、标签index
    train_data = prepare_dataset(train_sentences, char_to_id, tag_to_id, args.lower)
    dev_data = prepare_dataset(dev_sentences, char_to_id, tag_to_id, args.lower)
    test_data = prepare_dataset(test_sentences, char_to_id, tag_to_id, args.lower)

    logger.debug(f"{len(train_data)}, {len(dev_data)}, {len(test_data)} 句子分别来自train，dev，test")

    train_manager = BatchManager(train_data, args.batch_size)
    dev_manager = BatchManager(dev_data, 100)
    test_manager = BatchManager(test_data, 100)

    if os.path.isfile(args.config_file):
        config = load_json(args.config_file)
    else:
        config = config_model(args, char_to_id, tag_to_id)
        dump_json(config, args.config_file)

    print_config(config, " config 配置文件")

    tf_config = tf.compat.v1.ConfigProto()
    if args.gpu_proportion == 1:
        # 动态申请使用gpu内存，默认使用全部
        tf_config.gpu_options.allow_growth = True
    elif 0 < args.gpu_proportion < 1:
        # 当使用gpu时，设置使用最大比例:0-1区间
        tf_config.gpu_options.per_process_gpu_memory_fraction = args.gpu_proportion
    else:
        raise Exception(f"--gpu_proportion设置错误：'{args.gpu_proportion}',范围区间为(0, 1]")

    # 批次
    steps_per_epoch = train_manager.len_data
    sess = tf.compat.v1.Session(config=tf_config)
    with sess.as_default():
        model = create_model(sess, args, config, id_to_char, is_train=True)
        logger.info("开始进行训练")
        for i in range(100):
            for batch in train_manager.iter_batch(shuffle=True):
                loss = []
                step, batch_loss = model.run_step(sess, True, batch)
                loss.append(batch_loss)
                if step % args.steps_check == 0:
                    iteration = step // steps_per_epoch + 1
                    logger.debug(f"迭代: {iteration}次\t 步数: {step % steps_per_epoch}/{steps_per_epoch}\t loss: {np.mean(loss)}")
            best = evaluate(args, sess, model, "dev", dev_manager, id_to_tag)
            if best:
                save_model(sess, model, args.ckpt_path)
            evaluate(args, sess, model, "test", test_manager, id_to_tag)
    tf.compat.v1.summary.merge_all()
    board = tf.compat.v1.summary.FileWriter("./log", sess.graph)
    board.close()
    sess.close()


def test(args):
    config = load_json(args.config_file)

    tf_config = tf.compat.v1.ConfigProto()
    if args.gpu_proportion == 1:
        # 动态申请使用gpu内存，默认使用全部
        tf_config.gpu_options.allow_growth = True
    elif 0 < args.gpu_proportion < 1:
        # 当使用gpu时，设置使用最大比例:0-1区间
        tf_config.gpu_options.per_process_gpu_memory_fraction = args.gpu_proportion
    else:
        raise Exception(f"--gpu_proportion设置错误：'{args.gpu_proportion}',范围区间为(0, 1]")

    with open(args.map_file, "rb") as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)

    sess = tf.compat.v1.Session(config=tf_config)
    model = create_model(sess, args, config, id_to_char, is_train=False)
    while True:
        line = input("请输入测试句子:")
        result = model.evaluate_line(sess, input_from_line(line, char_to_id), id_to_tag)
        print(result)


train(get_parser())
# test(get_parser())