# encoding=utf-8
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.crf import crf_log_likelihood, viterbi_decode
from data_utils import iobes_iob, result_to_json


class Model:

    def __init__(self, config, is_train=True):
        # 初始化配置文件
        self.initialize_config(config, is_train)
        self.create_tensor()

        embedding = self.embedding_layer(self.char_inputs, self.seg_inputs)

        model_inputs = tf.nn.dropout(embedding, rate=self.dropout)

        # ldcnn layer
        model_outputs = self.idcnn_layer(model_inputs)

        # logits for tags
        self.logits = self.project_layer_idcnn(model_outputs)

        self.loss = self.loss_layer(self.logits, self.lengths)

        tf.compat.v1.summary.scalar("IDCNN_LAYER", self.loss)

        # 优化器
        with tf.compat.v1.variable_scope("optimizer"):
            optimizer = self.config["optimizer"]
            if optimizer == "sgd":
                self.opt = tf.train.GradientDescentOptimizer(self.lr)
            elif optimizer == "adam":
                self.opt = tf.compat.v1.train.AdamOptimizer(self.lr)
            elif optimizer == "adgrad":
                self.opt = tf.train.AdagradOptimizer(self.lr)
            else:
                raise KeyError

            # apply grad clip to avoid gradient explosion
            grads_vars = self.opt.compute_gradients(self.loss)
            capped_grads_vars = [
                [tf.clip_by_value(g, -self.config["clip"], self.config["clip"]), v] for g, v in grads_vars
            ]
            self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step)

        self.saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=5)

    def initialize_config(self, config, is_train):
        """初始化配置文件"""
        self.config = config
        self.is_train = is_train
        self.lr = self.config["lr"]
        self.num_chars = self.config["num_chars"]
        self.char_dim = self.config["char_dim"]
        # 过滤器的宽度
        self.filter_dim = self.config["filter_dim"]
        # 暂时未知什么意思
        self.seg_dim = self.config["seg_dim"]
        self.num_tags = config["num_tags"]
        self.num_segs = 4

        # 过滤器的宽度\滤波器的宽度
        self.filter_width = 3
        self.embedding_dim = self.char_dim + self.seg_dim
        self.repeat_times = 4
        self.cnn_output_width = 0

        # IDCNN 的连续三层膨胀卷积间隔
        self.layers = [
            {'dilation': 1},
            {'dilation': 1},
            {'dilation': 2},
        ]

    def create_tensor(self):
        """创建张量流"""
        self.global_step = tf.Variable(0, trainable=False)
        self.best_dev_f1 = tf.Variable(0.0, trainable=False)
        self.best_test_f1 = tf.Variable(0.0, trainable=False)

        self.initializer = initializers.xavier_initializer()

        self.char_inputs = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, None], name="ChatInputs")
        self.seg_inputs = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, None], name="SegInputs")
        self.targets = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, None], name="Targets")

        self.dropout = tf.compat.v1.placeholder(dtype=tf.float32, name="Dropout")

        # tf.sign   -1 if x < 0 elif x == 0 0 else: 1
        used = tf.sign(tf.abs(self.char_inputs))
        # 求和，通过axis来进行按行还是列，reduction_indices（axis旧的版本）
        length = tf.reduce_sum(used, reduction_indices=1)
        self.lengths = tf.cast(length, tf.int32)
        self.batch_size = tf.shape(self.char_inputs)[0]
        self.num_steps = tf.shape(self.char_inputs)[-1]

    def embedding_layer(self, char_inputs, seg_inputs):
        embedding = []
        with tf.compat.v1.variable_scope("char_embedding"), tf.device("/cpu:0"):
            # 字符集的向量
            self.char_lookup = tf.compat.v1.get_variable(
                name="char_embedding",
                shape=[self.num_chars, self.char_dim],
                initializer=self.initializer
            )
            embedding.append(tf.nn.embedding_lookup(self.char_lookup, char_inputs))
            if self.config["seg_dim"]:
                with tf.compat.v1.variable_scope("seg_embedding"), tf.device('/cpu:0'):
                    self.seg_lookup = tf.compat.v1.get_variable(
                        name="seg_embedding",
                        shape=[self.num_segs, self.seg_dim],
                        initializer=self.initializer)
                    embedding.append(tf.nn.embedding_lookup(self.seg_lookup, seg_inputs))
            embed = tf.concat(embedding, axis=-1)
        tf.compat.v1.summary.scalar("EMBEDDING_LAYER",embed)
        return embed

    def idcnn_layer(self, model_inputs):
        # 增加一个维度在shape的索引为1处
        model_inputs = tf.expand_dims(model_inputs, 1)
        with tf.compat.v1.variable_scope("idcnn"):
            shape = [1, self.filter_width, self.embedding_dim, self.filter_dim]
            filter_weights = tf.compat.v1.get_variable("idcnn_filter", shape=shape, initializer=self.initializer)
            # model_inputs.shape = [batch, 1, sentence_size, embedding_size]
            layerInput = tf.nn.conv2d(model_inputs, filter_weights, strides=[1, 1, 1, 1], padding="SAME", name="init_layer")
            finalOutFromLayers = []
            totalWidthForLastDim = 0
            for j in range(self.repeat_times):
                for i in range(len(self.layers)):
                    dilation = self.layers[i]['dilation']
                    isLast = True if i == (len(self.layers) - 1) else False
                    # 如果变量不存在，则创建它们
                    with tf.compat.v1.variable_scope("atrous-conv-layer-%d" % i, reuse=tf.compat.v1.AUTO_REUSE):
                        w = tf.compat.v1.get_variable(
                            "filterW", 
                            shape=[1, self.filter_width, self.filter_dim, self.filter_dim],
                            initializer=tf.contrib.layers.xavier_initializer()
                        )
                        b = tf.compat.v1.get_variable("filterB", shape=[self.filter_dim])
                        conv = tf.nn.atrous_conv2d(layerInput, w, rate=dilation, padding="SAME")
                        conv = tf.nn.bias_add(conv, b)
                        conv = tf.nn.relu(conv)
                        if isLast:
                            finalOutFromLayers.append(conv)
                            totalWidthForLastDim += self.filter_dim
                        layerInput = conv
            # 做了self.layers次3层的膨胀卷积，（叠加的，layerInput=conv）
            finalOut = tf.concat(axis=3, values=finalOutFromLayers)
            # 训练就给0.5的dropout
            keepProb = 0.5 if self.is_train else 1.0
            finalOut = tf.nn.dropout(finalOut, rate=1-keepProb)

            # 删除上面添加的1维度
            finalOut = tf.squeeze(finalOut, [1])
            finalOut = tf.reshape(finalOut, [-1, totalWidthForLastDim])
            self.cnn_output_width = totalWidthForLastDim
            tf.compat.v1.summary.scalar("IDCNN_LAYER", finalOut)
            return finalOut

    def project_layer_idcnn(self, idcnn_outputs, name=None):
        """全连接预测结果"""
        # idcnn_outputs.shape = [batch_size * 句子长度 , self.repeat_times * embedding]
        with tf.compat.v1.variable_scope("project"):
            with tf.compat.v1.variable_scope("logits"):
                W = tf.compat.v1.get_variable("W", shape=[self.cnn_output_width, self.num_tags],
                                    dtype=tf.float32, initializer=self.initializer)
                b = tf.compat.v1.get_variable("b",  initializer=tf.constant(0.001, shape=[self.num_tags]))
                pred = tf.compat.v1.nn.xw_plus_b(idcnn_outputs, W, b)
            layer = tf.reshape(pred, [-1, self.num_steps, self.num_tags])
            tf.compat.v1.summary.scalar("PRE_IDCNN_LAYER", layer)
            return layer

    def loss_layer(self, project_logits, lengths):
        with tf.compat.v1.variable_scope("crf_loss"):
            small = -1000.0
            # pad logits for crf loss
            start_logits = tf.concat(
                [small * tf.ones(shape=[self.batch_size, 1, self.num_tags]), tf.zeros(shape=[self.batch_size, 1, 1])], axis=-1)
            pad_logits = tf.cast(small * tf.ones([self.batch_size, self.num_steps, 1]), tf.float32)
            logits = tf.concat([project_logits, pad_logits], axis=-1)
            logits = tf.concat([start_logits, logits], axis=1)
            targets = tf.concat(
                [tf.cast(self.num_tags*tf.ones([self.batch_size, 1]), tf.int32), self.targets], axis=-1)

            self.trans = tf.compat.v1.get_variable(
                "transitions",
                shape=[self.num_tags + 1, self.num_tags + 1],
                initializer=self.initializer)
            #  Maximum Likelihood Estimation ：  https://www.cnblogs.com/lliuye/p/9139032.html
            #  crf_log_likelihood  viterbi_decode   http://www.cnblogs.com/lovychen/p/8490397.html
            log_likelihood, self.trans = crf_log_likelihood(
                inputs=logits,
                tag_indices=targets,
                sequence_lengths=lengths+1,
                transition_params=self.trans
            )
        return tf.reduce_mean(-log_likelihood)

    def run_step(self, sess, is_train, batch):
        _, chars, segs, tags = batch
        feed_dict = {
            self.char_inputs: np.asarray(chars),
            self.seg_inputs: np.asarray(segs),
            self.dropout: 1.0,
        }

        if is_train:
            feed_dict[self.targets] = np.asarray(tags)
            feed_dict[self.dropout] = self.config["dropout_keep"]

            global_step, loss, _ = sess.run(
                [self.global_step, self.loss, self.train_op],
                feed_dict
            )
            return global_step, loss
        else:
            lengths, logits = sess.run([self.lengths, self.logits], feed_dict)
            return lengths, logits

    def decode(self, logits, lengths, matrix):
        paths = []
        small = -1000.0
        start = np.asarray([[small] * self.num_tags + [0]])
        for score, length in zip(logits, lengths):
            score = score[:length]
            pad = small * np.ones([length, 1])
            logits = np.concatenate([score,pad], axis=1)
            logits = np.concatenate([start, logits], axis=0)
            path, _ = viterbi_decode(logits, matrix)
            paths.append(path[1:])
        return paths

    def evaluate(self, sess, data_manager, id_to_tag):
        """评价"""
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

    def evaluate_line(self, sess, inputs, id_to_tag):
        trans = self.trans.eval(session=sess)
        lengths, scores = self.run_step(sess, False, inputs)
        batch_paths = self.decode(scores, lengths, trans)
        tags = [id_to_tag[idx] for idx in batch_paths[0]]
        return result_to_json(inputs[0][0], tags)

"""
tensorboard \
    --logdir=log \
    --host=localhost \
    --port=30001
"""
