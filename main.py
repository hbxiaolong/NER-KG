# encoding = utf8



import os
import time
import codecs
import shutil
import pickle
import itertools
from collections import OrderedDict

import tensorflow as tf
import numpy as np
from model import Model
from loader import load_sentences, update_tag_scheme
from loader import char_mapping, tag_mapping
from loader import augment_with_pretrained, prepare_dataset
from utils import get_logger, make_path, clean, create_model, save_model
from utils import FError, print_config, save_config, load_config, test_ner
from data_utils import load_word2vec, create_input, input_from_line, BatchManager

flags = tf.app.flags
flags.DEFINE_boolean("clean",       True,      "clean train folder")
flags.DEFINE_string("status",       "test",    "train or test or decode")

# configurations for network
flags.DEFINE_boolean("dot_product", False,      "Whether use dot_product attention")
flags.DEFINE_boolean("dot_product_idcnn", False,      "Whether use dot_product attention for idcnn")
flags.DEFINE_boolean("highway_lstm", False,     "Whether use highway network for lstm")
flags.DEFINE_boolean("highway_idcnn", False,    "Whether use highway network for idcnn")

# configurations for the model
flags.DEFINE_integer("seg_dim",     20,         "Embedding size for segmentation, 0 if not used")
flags.DEFINE_integer("char_dim",    100,        "Embedding size for characters")
flags.DEFINE_integer("lstm_dim",    100,        "Num of hidden units in LSTM")
flags.DEFINE_string("tag_schema",   "iobes",      "tagging schema iobes or iob")

# configurations for training
flags.DEFINE_float("clip",          5,          "Gradient clip")
flags.DEFINE_float("dropout",       0.5,        "Dropout rate")
flags.DEFINE_float("batch_size",    20,         "batch size")
flags.DEFINE_float("lr",            0.001,      "Initial learning rate")
flags.DEFINE_string("optimizer",    "adam",     "Optimizer for training")
flags.DEFINE_boolean("pre_emb",     True,       "Whether use pre-trained embedding")
flags.DEFINE_boolean("zeros",       False,      "Whether replace digits with zero")
flags.DEFINE_boolean("lower",       False,      "Whether lower case")


flags.DEFINE_integer("max_epoch",   30,         "maximum training epochs")
flags.DEFINE_integer("steps_check", 30,         "steps per checkpoint")
flags.DEFINE_string("ckpt_path",    "ckpt",      "Path to save model")
flags.DEFINE_string("summary_path", "summary",      "Path to store summaries")
flags.DEFINE_string("log_file",     "train.log",    "File for log")
flags.DEFINE_string("log_path",     "log",    "File path for log")
flags.DEFINE_string("test_log_file",    "test.log",  "File for test log")
flags.DEFINE_string("map_file",     "maps.pkl",     "file for maps")
flags.DEFINE_string("vocab_file",   "vocab.json",   "File for vocab")
flags.DEFINE_string("config_file",  "config_file",  "File for config")
flags.DEFINE_string("script",       "conlleval",    "evaluation script")
flags.DEFINE_string("result_path",  "result",       "Path for results")
flags.DEFINE_string("emb_file",     "wiki_100.utf8", "Path for pre_trained embedding")
flags.DEFINE_string("train_file",   os.path.join("data", "train.char.bmes"),  "Path for train data")
flags.DEFINE_string("dev_file",     os.path.join("data", "dev.char.bmes"),    "Path for dev data")
flags.DEFINE_string("test_file",    os.path.join("data", "test.char.bmes"),   "Path for test data")
flags.DEFINE_string("model_type", "idcnn", "Model type, can be idcnn or bilstm or idcnn_bilstm")

FLAGS = tf.app.flags.FLAGS
assert FLAGS.clip < 5.1, "gradient clip should't be too much"   # assert用预测试如果目标不满足就弹出错误
assert 0 <= FLAGS.dropout < 1, "dropout rate between 0 and 1"
assert FLAGS.lr > 0, "learning rate must larger than zero"
assert FLAGS.optimizer in ["adam", "sgd", "adagrad"]
assert FLAGS.status in ["train", "test", "decode"]
assert FLAGS.model_type in ["bilstm", "idcnn", "idcnn_bilstm"], "the model type only bilstm or idcnn or idcnn_bilstm"
if FLAGS.dot_product and FLAGS.highway_lstm:
    raise FError("It is not designed to use attention and highway at the same time!")
if FLAGS.dot_product_idcnn and FLAGS.highway_idcnn:
    raise FError("It is not designed to use attention and highway at the same time!")

if FLAGS.model_type == 'idcnn':
    if FLAGS.highway_idcnn:
        FLAGS.ckpt_path += '/idcnn_highway'
        FLAGS.log_path += '/idcnn_highway'
    elif FLAGS.dot_product_idcnn:
        FLAGS.ckpt_path += '/idcnn_attention'
        FLAGS.log_path += '/idcnn_attention'
    else:
        FLAGS.ckpt_path += '/idcnn'
        FLAGS.log_path += '/idcnn'
elif FLAGS.model_type == 'bilstm':
    if FLAGS.dot_product:
        FLAGS.ckpt_path += '/bilstm_attention'
        FLAGS.log_path += '/bilstm_attention'
    elif FLAGS.highway_lstm:
        FLAGS.ckpt_path += '/bilstm_highway'
        FLAGS.log_path += '/bilstm_highway'
    else:
        FLAGS.ckpt_path += '/bilstm'
        FLAGS.log_path += '/bilstm'
elif FLAGS.model_type == 'idcnn_bilstm':
    if FLAGS.dot_product:
        FLAGS.ckpt_path += '/idcnn_bilstm_attention'
        FLAGS.log_path += '/idcnn_bilstm_attention'
    elif FLAGS.highway_lstm:
        FLAGS.ckpt_path += '/idcnn_bilstm_highway'
        FLAGS.log_path += '/idcnn_bilstm_highway'
    else:
        FLAGS.ckpt_path += '/idcnn_bilstm'
        FLAGS.log_path += '/idcnn_bilstm'

FLAGS.config_file = os.path.join(FLAGS.log_path, FLAGS.config_file)
# config for the model


def config_model(char_to_id, tag_to_id):
    config = OrderedDict()      # 创建字典、按照输入顺序排序
    config["model_type"] = FLAGS.model_type
    config["num_chars"] = len(char_to_id)
    config["char_dim"] = FLAGS.char_dim
    config["num_tags"] = len(tag_to_id)
    config["seg_dim"] = FLAGS.seg_dim
    config["lstm_dim"] = FLAGS.lstm_dim
    config["batch_size"] = FLAGS.batch_size

    config["highway_lstm"] = FLAGS.highway_lstm
    config["highway_idcnn"] = FLAGS.highway_idcnn       # 补充前面的实验，考虑idcnn层的改动，新增两个变量
    config["dot_product_idcnn"] = FLAGS.dot_product_idcnn
    config["dot_product"] = FLAGS.dot_product

    config["emb_file"] = FLAGS.emb_file
    config["clip"] = FLAGS.clip
    config["dropout_keep"] = 1.0 - FLAGS.dropout
    config["optimizer"] = FLAGS.optimizer
    config["lr"] = FLAGS.lr
    config["tag_schema"] = FLAGS.tag_schema
    config["pre_emb"] = FLAGS.pre_emb
    config["zeros"] = FLAGS.zeros
    config["lower"] = FLAGS.lower
    return config


def evaluate(sess, model, name, data, id_to_tag, logger):
    logger.info("evaluate:{}".format(name))     # info(self, msg, *args, **kwargs) 通过 msg和不定参数args来进行日志的格式化
    ner_results = model.evaluate(sess, data, id_to_tag)
    eval_lines = test_ner(ner_results, FLAGS.result_path)
    for line in eval_lines:
        logger.info(line)
    f1 = float(eval_lines[1].strip().split()[-1])

    if name == "dev":
        best_test_f1 = model.best_dev_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_dev_f1, f1).eval()
            logger.info("new best dev f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1
    elif name == "test":
        best_test_f1 = model.best_test_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_test_f1, f1).eval()
            logger.info("new best test f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1


def train():
    # load data sets
    # 第一个为文件，读取BIO文件，保存为数组，第一个元素为值，第二个元素为标签     目前程序读到这里9.17
    train_sentences = load_sentences(FLAGS.train_file, FLAGS.lower, FLAGS.zeros)
    # print(train_sentences)
    # [[['陈', 'B-Person'], ['X', 'I-Person'], ['X', 'I-Person'], ['9', '0'], ['7', '0'], ['5', '0'], ['4', '0'], ['8', '0'], ['0', '0'], ['3', 'B-Age'], ['6', 'I-Age'], ['岁', 'I-Age']],
    dev_sentences = load_sentences(FLAGS.dev_file, FLAGS.lower, FLAGS.zeros)
    test_sentences = load_sentences(FLAGS.test_file, FLAGS.lower, FLAGS.zeros)

    # Use selected tagging scheme (IOB / IOBES)
    # update_tag_scheme(train_sentences, FLAGS.tag_schema)
    # update_tag_scheme(test_sentences, FLAGS.tag_schema)
    # 用于格式调整与更新，

    # create maps if not exist
    if not os.path.isfile(FLAGS.map_file):
        # create dictionary for word
        # pre_emb 默认为True 是否使用预训练的词嵌入
        print("aaa")
        if FLAGS.pre_emb:
            # char_mapping 为构建字典的工作，返回值有三个，第一个为字典(记录词和词频)，第二个为char_to_id，第三个为id_to_char 不过后两个为返回字典排序（出现词频多少）后的id_char的关系
            dico_chars_train = char_mapping(train_sentences, FLAGS.lower)[0]
            # emb_file 100 维的词向量文件 wiki_100.utf8
            # 词典文件，如果test文件中的字符出现在词向量文件但未出现在字典中，则在字典中增加这个词
            dico_chars, char_to_id, id_to_char = augment_with_pretrained(
                dico_chars_train.copy(),
                FLAGS.emb_file,
                # itertools.chain.from_iterable 迭代器，无限迭代将test_sentences 迭代成一个list，所有数据
                list(itertools.chain.from_iterable(
                    [[w[0] for w in s] for s in test_sentences])
                )
            )
        else:
            _c, char_to_id, id_to_char = char_mapping(train_sentences, FLAGS.lower)

        # 创建标签字典 单词映射
        # Create a dictionary and a mapping for tags
        _t, tag_to_id, id_to_tag = tag_mapping(train_sentences)
        with open(FLAGS.map_file, "wb") as f:
            pickle.dump([char_to_id, id_to_char, tag_to_id, id_to_tag], f)
    else:
        print("bbb")
        with open(FLAGS.map_file, "rb") as f:
            char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)

    # print(char_to_id)
    # prepare data, get a collection of list containing index
    # 20191011 读到这里。前面为字典映射工程，映射包含训练集、测试集的字符。以及所有标记标签映射
    train_data = prepare_dataset(
        train_sentences, char_to_id, tag_to_id, FLAGS.lower
    )
    # print(train_data)
    dev_data = prepare_dataset(
        dev_sentences, char_to_id, tag_to_id, FLAGS.lower
    )
    test_data = prepare_dataset(
        test_sentences, char_to_id, tag_to_id, FLAGS.lower
    )
    print("%i / %i / %i sentences in train / dev / test." % (
        len(train_data), len(dev_data), len(test_data)))
    # print("%i / %i / %i sentences in train / dev / test." % (
    #     len(train_data), 0, len(test_data)))

    # 实现train_manager 固定长度，为每batch_size最大句子长度。不足则pading 2019-10-16 下午5.30分读至此处 batch_size 默认20 表示训练集为20条句子一个 batch
    train_manager = BatchManager(train_data, FLAGS.batch_size)
    dev_manager = BatchManager(dev_data, 100)
    test_manager = BatchManager(test_data, 100)

    # make path for store log and model if not exist
    make_path(FLAGS)
    if os.path.isfile(FLAGS.config_file):
        config = load_config(FLAGS.config_file)
    else:
        config = config_model(char_to_id, tag_to_id)
        save_config(config, FLAGS.config_file)
    make_path(FLAGS)

    # if not os.path.exists(FLAGS.log_path):
    #     os.makedirs(FLAGS.log_path)
    log_path = os.path.join(FLAGS.log_path, FLAGS.log_file)
    logger = get_logger(log_path)
    print_config(config, logger)

    # limit GPU memory
    tf_config = tf.ConfigProto()
    # 当使用GPU时候，Tensorflow运行自动慢慢达到最大GPU的内存
    tf_config.gpu_options.allow_growth = True
    steps_per_epoch = train_manager.len_data
    # print(steps_per_epoch)
    # print(dev_manager.len_data)
    # print(test_manager.len_data)
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger)
        logger.info("start training")
        loss = []
        start = time.time()
        for i in range(50):
            for batch in train_manager.iter_batch(shuffle=True):
                step, batch_loss = model.run_step(sess, True, batch)
                loss.append(batch_loss)
                if step % FLAGS.steps_check == 0:
                    iteration = step // steps_per_epoch + 1
                    logger.info("iteration:{} step:{}/{}, "
                                "NER loss:{:>9.6f}".format(
                        iteration, step%steps_per_epoch, steps_per_epoch, np.mean(loss)))
                    loss = []

            best = evaluate(sess, model, "dev", dev_manager, id_to_tag, logger)
            if best:
                save_model(sess, model, FLAGS.ckpt_path, logger)
            evaluate(sess, model, "test", test_manager, id_to_tag, logger)
        end = time.time()
        logger.info("Time cost:{:>9.6}  ".format(str((end - start)/60)))


def evaluate_line():
    config = load_config(FLAGS.config_file)
    logger = get_logger(FLAGS.log_file)
    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with open(FLAGS.map_file, "rb") as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger, False)
        while True:
                line = input("请输入测试句子:")
                result = model.evaluate_line(sess, input_from_line(line, char_to_id), id_to_tag, FLAGS.tag_schema)
                # print(result)
                print(result["string"])
                for entity in result["entities"]:
                    print(entity)


def evaluate_sentence():
    config = load_config(FLAGS.config_file)
    if not os.path.exists(FLAGS.log_path):
        os.makedirs(FLAGS.log_path)
    test_log_file = os.path.join(FLAGS.log_path, FLAGS.test_log_file)
    # if os.path.isfile(test_log_file):
    #     os.remove(test_log_file)
    test_logger = get_logger(test_log_file)
    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with open(FLAGS.map_file, "rb") as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
    test_sentences = load_sentences(FLAGS.test_file, FLAGS.lower, FLAGS.zeros)
    test_data = prepare_dataset(test_sentences, char_to_id, tag_to_id, FLAGS.lower)
    test_manager = BatchManager(test_data, 100)
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char, test_logger, False)
        test_logger.info("evaluate:{}".format('test'))  # info(self, msg, *args, **kwargs) 通过 msg和不定参数args来进行日志的格式化
        ner_results = model.evaluate(sess, test_manager, id_to_tag)
        eval_lines = test_ner(ner_results, FLAGS.result_path)
        for line in eval_lines:
            test_logger.info(line)


def main(_):
    if FLAGS.status == "train":
        if FLAGS.clean:
            clean(FLAGS)
        train()
    elif FLAGS.status == "test":
        evaluate_sentence()
    elif FLAGS.status == "decode":
        evaluate_line()
    else:
        print("error status! only train, test, decode allowed.")


if __name__ == "__main__":
    tf.app.run(main)



