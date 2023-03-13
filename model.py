# encoding = utf8
import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib.layers.python.layers import initializers

import tensorflow.contrib.rnn as rnn
# import rnncell as rnn
from utils import result_to_json, result_to_json_BIO
from data_utils import create_input, iobes_iob


class Model(object):
    def __init__(self, config, is_train = True):
        self.config = config
        self.is_train = is_train
        self.highway_lstm = config["highway_lstm"]
        self.highway_idcnn = config["highway_idcnn"]
        self.dot_product = config["dot_product"]
        self.dot_product_idcnn = config["dot_product_idcnn"]

        self.lr = config["lr"]
        self.char_dim = config["char_dim"]
        self.lstm_dim = config["lstm_dim"]
        self.seg_dim = config["seg_dim"]

        self.num_tags = config["num_tags"]
        self.num_chars = config["num_chars"]
        # seg 有四种表达方式  分词 0 表示单字， 1 表示首字， 2 表示中间字， 3 表示尾字
        self.num_segs = 4
        self.num_heads = 1
        self.global_step = tf.Variable(0, trainable=False)
        self.best_dev_f1 = tf.Variable(0.0, trainable=False)
        self.best_test_f1 = tf.Variable(0.0, trainable=False)
        # 参数初始化
        self.initializer = initializers.xavier_initializer()

        # add placeholders for the model

        # self.char_inputs = tf.placeholder(dtype=tf.int32,
        #                                   shape=[None, None],
        #                                   name="ChatInputs")
        self.char_inputs = tf.placeholder(dtype=tf.int32,
                                          shape=[None, None],
                                          name="CharInputs")

        self.seg_inputs = tf.placeholder(dtype=tf.int32,
                                         shape=[None, None],
                                         name="SegInputs")

        self.targets = tf.placeholder(dtype=tf.int32,
                                      shape=[None, None],
                                      name="Targets")
        # dropout keep prob
        self.dropout = tf.placeholder(dtype=tf.float32,
                                      name="Dropout")

        self.model_type = config['model_type']
        # parameters for idcnn
        self.layers = [
            {
                'dilation': 1
            },
            {
                'dilation': 1
            },
            {
                'dilation': 2
            },
        ]
        self.filter_width = 3
        self.num_filter = self.lstm_dim
        self.embedding_dim = self.char_dim + self.seg_dim
        self.repeat_times = 4
        self.cnn_output_width = 0

        self.embeddings = tf.placeholder(tf.float32, shape=[None, None, None], name="embeddings")
        # sign 表示符号函数， abs 取绝对值 ， 非0值被记录，保留输入
        # reduce_sum 表示求和， reduction_indices = 1 表示行求和
        used = tf.sign(tf.abs(self.char_inputs))

        length = tf.reduce_sum(used, reduction_indices=1)
        self.lengths = tf.cast(length, tf.int32)

        self.batch_size = tf.shape(self.char_inputs)[0]
        self.num_steps = tf.shape(self.char_inputs)[-1]

        # embeddings for chinese character and segmentation representation
        # 20191017 读至这里
        # print(self.char_inputs.shape)
        embedding = self.embedding_layer(self.char_inputs, self.seg_inputs, config)
        # embedding.shape(?, ?, 120)
        # print(embedding.shape)
        # apply dropout before feed to lstm layer
        # print(tf.shape(self.char_inputs))
        model_inputs = tf.nn.dropout(embedding, self.dropout)
        if self.model_type == 'bilstm':
            # bi-directional lstm layer
            lstm_outputs = self.biLSTM_layer(model_inputs, self.lstm_dim, self.lengths)

            # logits for tags
            self.logits = self.project_layer_bilstm(lstm_outputs)
        elif self.model_type == 'idcnn':
            idcnn_outputs = self.idcnn_layer(model_inputs)
            self.logits = self.project_layer_idcnn(idcnn_outputs)
        elif self.model_type == 'idcnn_bilstm':
            idcnn_outputs = self.idcnn_layer(model_inputs, isreshape=False)
            lstm_outputs = self.biLSTM_layer(idcnn_outputs, self.lstm_dim, self.lengths)
            self.logits = self.project_layer_bilstm(lstm_outputs)
        else:
            raise KeyError
        # loss of the model
        self.loss = self.loss_layer(self.logits, self.lengths)

        with tf.variable_scope("optimizer"):
            optimizer = self.config["optimizer"]
            if optimizer == "sgd":
                self.opt = tf.train.GradientDescentOptimizer(self.lr)
            elif optimizer == "adam":
                self.opt = tf.train.AdamOptimizer(self.lr)
            elif optimizer == "adgrad":
                self.opt = tf.train.AdagradOptimizer(self.lr)
            else:
                raise KeyError

            # apply grad clip to avoid gradient explosion
            grads_vars = self.opt.compute_gradients(self.loss)
            capped_grads_vars = [[tf.clip_by_value(g, -self.config["clip"], self.config["clip"]), v]
                                 for g, v in grads_vars]
            self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step)

        # saver of the model
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def embedding_layer(self, char_inputs, seg_inputs, config, name=None):
        """
        :param char_inputs: one-hot encoding of sentence
        :param seg_inputs: segmentation feature
        :param config: wither use segmentation feature
        :return: [1, num_steps, embedding size],
        """
        embedding = []
        with tf.variable_scope("char_embedding" if not name else name), tf.device('/cpu:0'):
            self.char_lookup = tf.get_variable(
                    name="char_embedding",
                    shape=[self.num_chars, self.char_dim],
                    initializer=self.initializer)
            # tf.nn.embedding_lookup 查找 self.char_lookup 中 char_inputs （索引）对应的值
            # https://www.cnblogs.com/gaofighting/p/9625868.html
            # embedding (?,?,100)
            embedding.append(tf.nn.embedding_lookup(self.char_lookup, char_inputs))
            if config["seg_dim"]:
                with tf.variable_scope("seg_embedding"), tf.device('/cpu:0'):
                    self.seg_lookup = tf.get_variable(
                        name="seg_embedding",
                        shape=[self.num_segs, self.seg_dim],
                        initializer=self.initializer)
                    embedding.append(tf.nn.embedding_lookup(self.seg_lookup, seg_inputs))
                    # embedding (?,?,100) embedding_lookup seg_lookup (?,?,20)
                    # 在维度上如何
            self.embeddings = tf.concat(embedding, axis=-1)
            # embed (?,?,120)
        return self.embeddings


    def normalize(self,inputs, epsilon = 1e-8,scope="ln",reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]
            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta = tf.Variable(tf.zeros(params_shape),dtype=tf.float32)
            gamma = tf.Variable(tf.ones(params_shape),dtype=tf.float32)
            normalized = (inputs - mean) / ((variance + epsilon) ** (0.5))
            outputs = gamma * normalized + beta
        return outputs

    def self_attention(self, keys, size, scope='multihead_attention', reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            Q = tf.nn.relu(tf.layers.dense(keys, size, kernel_initializer=self.initializer))
            K = tf.nn.relu(tf.layers.dense(keys, size, kernel_initializer=self.initializer))
            V = tf.nn.relu(tf.layers.dense(keys, size, kernel_initializer=self.initializer))
            Q_ = tf.concat(tf.split(Q, self.num_heads, axis=2), axis=0)
            K_ = tf.concat(tf.split(K, self.num_heads, axis=2), axis=0)
            V_ = tf.concat(tf.split(V, self.num_heads, axis=2), axis=0)
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
            key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))
            key_masks = tf.tile(key_masks, [self.num_heads, 1])
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(keys)[1], 1])
            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)
            outputs = tf.nn.softmax(outputs)
            query_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))
            query_masks = tf.tile(query_masks, [self.num_heads, 1])
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])
            outputs *= query_masks
            outputs = tf.nn.dropout(outputs, keep_prob=self.dropout)
            outputs = tf.matmul(outputs, V_)
            outputs = tf.concat(tf.split(outputs, self.num_heads, axis=0), axis=2)
            outputs += keys
            outputs = self.normalize(outputs)
        return outputs
    
    def highway(self, x, y, size):
        ss = tf.shape(x)
        x_ = tf.reshape(x, shape=[ss[0]*ss[1], size])
        # input2 = Dense(input2, size)
        y_ = tf.reshape(y, shape=[ss[0]*ss[1], size])

        W = tf.Variable(tf.random_normal([size, size], stddev=0.1))
        b = tf.Variable(tf.constant(-1.0, shape=[size]), name="bias")
        T = tf.sigmoid(tf.matmul(x_, W) + b, name="transform_gate")
        output = y_ * T + x_ * (1 - T)
        output = tf.reshape(output, shape=[ss[0], ss[1], size])
        return output


    def biLSTM_layer(self, lstm_inputs, lstm_dim, lengths, name=None):
        """
        :param lstm_inputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, 2*lstm_dim]
        """
        with tf.variable_scope("lstm1" if not name else name):
            lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=lstm_dim, initializer=self.initializer)
            lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, input_keep_prob=self.dropout)
            lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, input_keep_prob=self.dropout)
            value, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                                       cell_bw=lstm_bw_cell,
                                                       inputs=lstm_inputs,
                                                       dtype=tf.float32,
                                                       sequence_length=lengths)
            concat_tensor = tf.concat(value, axis=2)

        if self.dot_product:
            concat_tensor = self.self_attention(concat_tensor, 2 * lstm_dim)
        elif self.highway_lstm:
            embedding_word = self.embeddings[:, :, 0:self.char_dim]
            bi_embedding = tf.concat((embedding_word, embedding_word), axis=-1)
            concat_tensor = self.highway(bi_embedding, concat_tensor, 2 * lstm_dim)

        return concat_tensor

    def project_layer_bilstm(self, lstm_outputs, name=None):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size] 
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project" if not name else name):
            with tf.variable_scope("hidden"):
                W = tf.get_variable("W", shape=[self.lstm_dim*2, self.lstm_dim],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.lstm_dim], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(lstm_outputs, shape=[-1, self.lstm_dim*2])
                hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))

            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.lstm_dim, self.num_tags],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.num_tags], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

                pred = tf.nn.xw_plus_b(hidden, W, b)

            return tf.reshape(pred, [-1, self.num_steps, self.num_tags])

    def idcnn_layer(self, idcnn_inputs, name=None, isreshape = True):
        """
        :param idcnn_inputs: [batch_size, num_steps, emb_size]
        :param name:
        :return:[batch_size, num_steps, cnn_output_width]
        """
        highway_inputs = idcnn_inputs[:, :, 0:self.char_dim]
        concat_inputs = idcnn_inputs[:, :, 0:self.char_dim]
        idcnn_inputs = tf.expand_dims(idcnn_inputs, 1)
        reuse = False
        if not self.is_train:
            reuse = True
        with tf.variable_scope("idcnn" if not name else name):
            # shape = [1, 3, 120, 100]
            filter_weights = tf.get_variable(
                "idcnn_filter",
                shape=[1, self.filter_width, self.embedding_dim,
                       self.num_filter],
                initializer=self.initializer)
            """
            shape of input = [batch, in_height, in_width, in_channels]
            shape of filter = [filter_height, filter_width, in_channels, out_channels]
            """
            # 定义卷积层
            layerInput = tf.nn.conv2d(idcnn_inputs,
                                      filter_weights,
                                      strides=[1, 1, 1, 1],
                                      padding="SAME",
                                      name="init_layer")
            finalOutFromLayers = []
            totalWidthForLastDim = 0
            # 由于repeat time ，重复每一次的空洞卷积， 实际上卷积的层数为repeat_times * layers，
            # 在这里是4 * 3 = 12 层。 分别是 0 0 1 0 0 1 0 0 1 0 0 1 个空洞
            for _ in range(self.repeat_times):
                for i in range(len(self.layers)):
                    dilation = self.layers[i]['dilation']
                    isLast = True if i == (len(self.layers) - 1) else False
                    with tf.variable_scope("atrous-conv-layer-%d" % i,
                                           reuse=tf.AUTO_REUSE):
                        w = tf.get_variable(
                            "filterW",
                            shape=[1, self.filter_width, self.num_filter,
                                   self.num_filter],
                            initializer=tf.contrib.layers.xavier_initializer())
                        b = tf.get_variable("filterB", shape=[self.num_filter])
                        # tf.nn.atrous_conv2d 实现空洞卷积，即膨胀卷积
                        conv = tf.nn.atrous_conv2d(layerInput,
                                                   w,
                                                   rate=dilation,
                                                   padding="SAME")
                        conv = tf.nn.bias_add(conv, b)
                        conv = tf.nn.relu(conv)
                        if isLast:
                            finalOutFromLayers.append(conv)
                            totalWidthForLastDim += self.num_filter
                            highway_inputs = tf.concat((highway_inputs, concat_inputs), axis=-1)
                        # 这里似乎可以用highway，要求是四维的，要理解idcnn的输入输出格式
                        layerInput = conv

            finalOut = tf.concat(axis=3, values=finalOutFromLayers)
            keepProb = 1.0 if reuse else self.dropout
            finalOut = tf.nn.dropout(finalOut, keepProb)

            finalOut = tf.squeeze(finalOut, [1])
            if self.dot_product_idcnn:
                finalOut = self.self_attention(finalOut, totalWidthForLastDim)
            elif self.highway_idcnn:
                # 这里是维度错误，请核实finalOut, highway_inputs 维度问题
                highway_inputs = highway_inputs[:, :, 0:totalWidthForLastDim]
                finalOut = self.highway(highway_inputs, finalOut, totalWidthForLastDim)
            if isreshape:
                finalOut = tf.reshape(finalOut, [-1, totalWidthForLastDim])
            self.cnn_output_width = totalWidthForLastDim
            return finalOut

    # Project layer for idcnn by crownpku
    # Delete the hidden layer, and change bias initializer
    def project_layer_idcnn(self, idcnn_outputs, name=None):
        """
        :param lstm_outputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project" if not name else name):
            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.cnn_output_width, self.num_tags],
                                        dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", initializer=tf.constant(0.001, shape=[self.num_tags]))

                pred = tf.nn.xw_plus_b(idcnn_outputs, W, b)

            return tf.reshape(pred, [-1, self.num_steps, self.num_tags])

    def loss_layer(self, project_logits, lengths, name=None):
        """
        calculate crf loss
        :param project_logits: [1, num_steps, num_tags]
        :return: scalar loss
        """
        with tf.variable_scope("crf_loss" if not name else name):
            small = -1000.0
            # pad logits for crf loss
            start_logits = tf.concat(
                [small * tf.ones(shape=[self.batch_size, 1, self.num_tags]), tf.zeros(shape=[self.batch_size, 1, 1])], axis=-1)
            pad_logits = tf.cast(small * tf.ones([self.batch_size, self.num_steps, 1]), tf.float32)
            logits = tf.concat([project_logits, pad_logits], axis=-1)
            logits = tf.concat([start_logits, logits], axis=1)
            targets = tf.concat(
                [tf.cast(self.num_tags*tf.ones([self.batch_size, 1]), tf.int32), self.targets], axis=-1)

            self.trans = tf.get_variable(
                "transitions",
                shape=[self.num_tags + 1, self.num_tags + 1],
                initializer=self.initializer)
            log_likelihood, self.trans = crf_log_likelihood(
                inputs=logits,
                tag_indices=targets,
                transition_params=self.trans,
                sequence_lengths=lengths+1)
            return tf.reduce_mean(-log_likelihood)

    def create_feed_dict(self, is_train, batch):
        """
        :param is_train: Flag, True for train batch
        :param batch: list train/evaluate data 
        :return: structured data to feed
        """
        _, chars, segs, tags = batch
        feed_dict = {
            self.char_inputs: np.asarray(chars),
            self.seg_inputs: np.asarray(segs),
            self.dropout: 1.0,
        }
        if is_train:
            feed_dict[self.targets] = np.asarray(tags)
            feed_dict[self.dropout] = self.config["dropout_keep"]
        return feed_dict

    def run_step(self, sess, is_train, batch):
        """
        :param sess: session to run the batch
        :param is_train: a flag indicate if it is a train batch
        :param batch: a dict containing batch data
        :return: batch result, loss of the batch or logits
        """
        feed_dict = self.create_feed_dict(is_train, batch)
        if is_train:
            global_step, loss, _ = sess.run(
                [self.global_step, self.loss, self.train_op],
                feed_dict)
            return global_step, loss
        else:
            lengths, logits = sess.run([self.lengths, self.logits], feed_dict)
            return lengths, logits

    def decode(self, logits, lengths, matrix):
        """
        :param logits: [batch_size, num_steps, num_tags]float32, logits
        :param lengths: [batch_size]int32, real length of each sequence
        :param matrix: transaction matrix for inference
        :return:
        """
        # inference final labels usa viterbi Algorithm
        paths = []
        small = -1000.0
        start = np.asarray([[small]*self.num_tags +[0]])
        for score, length in zip(logits, lengths):
            score = score[:length]
            pad = small * np.ones([length, 1])
            logits = np.concatenate([score, pad], axis=1)
            logits = np.concatenate([start, logits], axis=0)
            path, _ = viterbi_decode(logits, matrix)

            paths.append(path[1:])
        return paths

    def evaluate(self, sess, data_manager, id_to_tag):
        """
        :param sess: session  to run the model 
        :param data: list of data
        :param id_to_tag: index to tag name
        :return: evaluate result
        """
        results = []
        trans = self.trans.eval()
        for batch in data_manager.iter_batch():
            strings = batch[0]
            tags = batch[-1]
            lengths, scores = self.run_step(sess, False, batch)
            batch_paths = self.decode(scores, lengths, trans)
            for i in range(len(strings)):
                result = []
                string = strings[i][:lengths[i]]
                gold = iobes_iob([id_to_tag[int(x)] for x in tags[i][:lengths[i]]])
                pred = iobes_iob([id_to_tag[int(x)] for x in batch_paths[i][:lengths[i]]])
                for char, gold, pred in zip(string, gold, pred):
                    result.append(" ".join([char, gold, pred]))
                results.append(result)
        return results

    def evaluate_line(self, sess, inputs, id_to_tag, tag_schema):
        trans = self.trans.eval()
        lengths, scores = self.run_step(sess, False, inputs)
        batch_paths = self.decode(scores, lengths, trans)
        tags = [id_to_tag[idx] for idx in batch_paths[0]]
        print(tags)
        if tag_schema.lower() == 'bioes':
            return result_to_json(inputs[0][0], tags)
        elif tag_schema.lower() == 'bio':
            return result_to_json_BIO(inputs[0][0], tags)

