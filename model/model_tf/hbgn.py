import argparse
import os
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from typing import Tuple
from util.helper import ensure_dir, logger
from model_tf.load_gnn import GNNLoader


class HBGN(object):
    def __init__(self, args:argparse.Namespace, data_generator:GNNLoader,
    pretrain_embedding:np.array, message_embedding:np.array) -> None:
        super().__init__()
        """ Parse arguments for HBGN """
        self._parse_args(args, data_generator, pretrain_embedding, message_embedding)

        """ Create placeholder for training inputs """
        self._build_inputs()
        if args.select_gpu == 0:
            a, b = 0, 1
        else:
            a, b = 2, 3

        """ Create variable for training weights """
        with tf.device("/device:GPU:%d" % a):
            self._build_weights()
            self._create_entity_graph_embed()
            self._build_intra_block_embedding()
        with tf.device("/device:GPU:%d" % b):
            self._build_inter_block_embedding()
            self._build_classification_model()
            self._build_classification_loss()

        """ Count the number of model parameters """
        self._statistics_params()

    def setup_sess(self) -> tf.Session:
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        sess = tf.Session(config=tf_config)
        sess.run(tf.global_variables_initializer())

        return sess

    def _parse_args(self, args:argparse.Namespace, data_generator:GNNLoader, 
    pretrain_embedding:np.array, pre_message_embedding:np.array) -> None:
        """"""
        # pretrain word2vec embedding
        self.pretrain_embedding = pretrain_embedding
        self.pre_message_embedding = pre_message_embedding

        # use message embedding or not
        self.append_message = True if args.use_message < 2 else False

        # statistics for dataset 
        self.n_entity = data_generator.n_entity
        self.n_block = data_generator.n_block
        self.n_log_token = data_generator.n_log_token
        self.max_entity_block = data_generator.max_entity_block
        self.max_token_log = data_generator.max_token_log
        self.one_hot = data_generator.one_hot

        # setting for intra-block gnn
        self.entity_dim = args.type_dim * 2
        self.intra_weight_size = eval(args.intra_layer_size)
        self.n_intra_layer = len(self.intra_weight_size)
        
        self.inter_dim = self.entity_dim + sum(self.intra_weight_size)

        # setting for inter-block gnn
        self.log_dim = args.log_dim
        self.inter_weight_size = eval(args.inter_layer_size)
        self.n_inter_layer = len(self.inter_weight_size) 

        # aggregation approach
        self.agg_type = args.agg_type

        # setting for loss function (optimization)
        self.regs = eval(args.regs)
        self.opt_type = args.opt_type
        self.lr = args.lr
        self.fc_size = args.fc_size

        # init adjacency matrix for gnn (setup n_fold to avoid 'run out of
        # memory' during model training)
        self.A_intra = data_generator.A_intra
        self.A_inter = data_generator.A_inter

        # block ids 
        self.block_ids = data_generator.blocks

        self.n_fold = 1
        self.n_fold_block = 1

        self.model_type = '%s_%s_%s' % (args.model_type, args.adj_type, self.agg_type)
        self.layer_size = eval(args.intra_layer_size) + eval(args.inter_layer_size)
        self.layer_type = '-'.join([str(l) for l in self.layer_size])
        self.reg_type = '-'.join([str(r) for r in eval(args.regs)])
        
        # setting for code classification
        self.classification_num = args.classification_num

    def _build_inputs(self) -> None:
        """"""
        # dropout: message dropout (adopted on GNN aggregation)
        self.intra_mess_dropout = tf.placeholder(tf.float32, shape=[None], name='intra_mess_dropout')
        self.inter_mess_dropout = tf.placeholder(tf.float32, shape=[None], name='inter_mess_dropout')
        
        # input block/label for log level prediction
        self.b_classification = tf.placeholder(dtype=tf.int64, name='b_classification',
            shape=[None])
        self.y_classification = tf.placeholder(dtype=tf.float32, name='y_classification',
            shape=[None, self.classification_num])
        
        # log message
        self.message_token = tf.placeholder(tf.int64, name='message_token', shape=[None, self.max_token_log])

        logger.info("Finish building inputs for HBGN")

    def _build_weights(self) -> None:
        """"""
        all_weight = dict()
        initializer = tf.contrib.layers.variance_scaling_initializer(seed=2022)
    
        # weights for entity embeddings
        entity_trainable = True
        if self.pretrain_embedding is None:
            all_weight['entity_embedding'] = tf.Variable(
            initial_value=initializer([self.n_entity, self.entity_dim]),
            dtype=tf.float32,
            trainable=entity_trainable,
            name='entity_embedding')
            logger.info("Init entity embeddings with He")
        else:
            all_weight['entity_embedding'] = tf.Variable(
            initial_value=self.pretrain_embedding,
            trainable=entity_trainable,
            name='entity_embedding')
            logger.info("Init entity embeddings with pre-trained word2vec embeddings")
        
        # weights for message embeddings
        if self.pre_message_embedding is None:
            all_weight['message_embedding'] = tf.Variable(
                initial_value=initializer([self.n_log_token, self.log_dim]),
                dtype=tf.float32,
                trainable=entity_trainable,
                name='message_embedding')
            logger.info("Init message embeddings with He")
        else:
            all_weight['message_embedding'] = tf.Variable(
                initial_value=self.pre_message_embedding,
                trainable=entity_trainable,
                name='message_embedding')
            logger.info('Init message embeddings with pre-trained word2vec embeddings')

        # we add a `ghost` entity whose embedding is [0,..,0] to allow using 
        # tf.nn.embedding_lookup for obtaining code block representations
        paddings = tf.constant([[0, 1], [0, 0]])
        all_weight['entity_embedding'] = tf.pad(
            all_weight['entity_embedding'], paddings, "CONSTANT", 
            name='entity_embedding_pad')
        all_weight['message_embedding'] = tf.pad(
            all_weight['message_embedding'], paddings, "CONSTANT",
            name='message_embedding_pad')

        # weights for intra-block gnn
        intra_weight_size_list = [self.entity_dim] + self.intra_weight_size
        for k in range(self.n_intra_layer):
            if self.agg_type in ['graphsage']:
                all_weight['w_intra_%d' % k] = tf.Variable(
                    initial_value=initializer([intra_weight_size_list[k] * 2, intra_weight_size_list[k+1]]),
                    name='w_intra_%d' % k)
                all_weight['b_intra_%d' % k] = tf.Variable(
                    initial_value=initializer([1, intra_weight_size_list[k+1]]),
                    name='b_intra_%d' % k)    
            else:
                logger.error('aggregator type for GNN is unknown')
                exit(-1)
        
        # weights for inter-block gnn
        self.middle_dim = self.inter_dim
        inter_weight_size_list = [self.middle_dim] + self.inter_weight_size
        for k in range(self.n_inter_layer):
            if self.agg_type in ['graphsage']:
                all_weight['w_inter_%d' % k] = tf.Variable(
                    initial_value=initializer([inter_weight_size_list[k] * 2, inter_weight_size_list[k+1]]),
                    name='w_inter_%d' % k)
                all_weight['b_inter_%d' % k] = tf.Variable(
                    initial_value=initializer([1, inter_weight_size_list[k+1]]),
                    name='b_inter_%d' % k)
            else:
                logger.error('aggregator type for GNN is unknown')
                exit(-1)


        # weights for log level prediction
        self.outer_dim = sum(inter_weight_size_list)

        # fully-connect layer size
        all_weight['fc_w_message'] = tf.Variable(
            initial_value=initializer([self.log_dim, self.fc_size]), name='fc_w_message')
        all_weight['fc_b_message'] = tf.Variable(
            initial_value=initializer([1, self.fc_size]), name='fc_b_message')
        all_weight['fc_w_classification'] = tf.Variable(
            initial_value=initializer([self.outer_dim, self.fc_size]), name='fc_w_classification')
        all_weight['fc_b_classification'] = tf.Variable(
            initial_value=initializer([1, self.fc_size]), name='fc_b_classification')
        
        if self.append_message:
            fc_out_size = 2 * self.fc_size
        else:
            fc_out_size = self.outer_dim

        all_weight['fc_w_out'] = tf.Variable(
            initial_value=initializer([fc_out_size, self.classification_num]), name='fc_w_out')
        all_weight['fc_b_out'] = tf.Variable(
            initial_value=initializer([1, self.classification_num]), name='fc_b_out')


        self.weights = all_weight
        
        logger.info("Finish building weights for HBGN")

    def _create_entity_graph_embed(self) -> tf.Tensor:
        """"""
        # generate a set of adjacency sub-matrix.
        A_fold_hat = self._split_A_hat(self.A_intra, self.n_entity, self.n_fold)

        # previous embedding (before update): shape = [n_entity + 1, inter_dim]
        pre_embedding = self.weights['entity_embedding']
        g_embeddings = [pre_embedding]

        for k in range(self.n_intra_layer):
            # propagation & neighbor_embedding shape = [n_entity + 1, inter_dim]
            if self.n_fold == 1:
                neighbor_embedding = tf.sparse_tensor_dense_matmul(
                    A_fold_hat[0], pre_embedding, name='intra_neighbor_%d' % k)
            else:
                temp_embed = []
                for i in range(self.n_fold):
                    temp_embed.append(tf.sparse_tensor_dense_matmul(
                        A_fold_hat[i], pre_embedding))
                neighbor_embedding = tf.concat(temp_embed, axis=0, name='intra_neighbor_%d' % k)

            # aggregation: (W(eh || eNh) + b)
            pre_embedding = tf.concat([pre_embedding, neighbor_embedding], 1)
            pre_embedding = tf.matmul(pre_embedding, self.weights['w_intra_%d' % k]) + self.weights['b_intra_%d' % k]
            pre_embedding = tf.math.l2_normalize(pre_embedding, axis=1, name='intra_norm_%d' % k)
            pre_embedding = tf.nn.leaky_relu(pre_embedding, name='intra_agg_%d' %k)

            # dropout 
            pre_embedding = tf.nn.dropout(
                pre_embedding, rate=self.intra_mess_dropout[k], name='intra_dropout_%d' % k, seed=2022)

            # normalize the distribution of entity embeddings
            # norm_embeddings = tf.math.l2_normalize(pre_embedding, axis=1, name='intra_norm_%d' % k)

            # concatenate information from different layers
            g_embeddings += [pre_embedding]
        
        g_embeddings = tf.concat(g_embeddings, 1)
    
        self.g_embeddings = g_embeddings

    def _generate_intra_block_embedding(self, b) -> tf.Tensor:
        """"""
        # lookup entity embeddings for block statements
        # b_e shape: [batch_size, max_entity_block, inter_dim]
        b_e = tf.nn.embedding_lookup(self.g_embeddings, b)

        # average pooling entity embeddings in code blocks
        b_p = tf.layers.average_pooling1d(
            inputs=b_e,
            pool_size=self.max_entity_block,
            strides=1
        )

        b_p = tf.reshape(b_p, [-1, self.inter_dim])

        return b_p
    
    def _build_intra_block_embedding(self) -> None:
        """To avoid OOM problem, build in times.
        """
        B_fold = self._split_B(self.block_ids, self.n_fold_block)
        block_embeddings = []

        for b in B_fold:
            block_embeddings.append(self._generate_intra_block_embedding(b))

        intra_embeddings = tf.concat(block_embeddings, axis=0)
        intra_embeddings = tf.math.l2_normalize(intra_embeddings, axis=1)

        self.intra_embeddings = intra_embeddings

    def _build_inter_block_embedding(self):
        """"""
        # generate a set of adjacency sub-matrix
        A_fold_hat = self._split_A_hat(self.A_inter, self.n_block, self.n_fold)
        intra_embeddings = self.intra_embeddings
        inter_embeddings = [intra_embeddings]

        for k in range(self.n_inter_layer):
            if self.n_fold == 1:
                neighbor_embedding = tf.sparse_tensor_dense_matmul(A_fold_hat[0], intra_embeddings, name='inter_neighbor_%d' % k)
            else:
                temp_embed = []
                for i in range(self.n_fold):
                    temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[i], intra_embeddings))
                neighbor_embedding = tf.concat(temp_embed, axis=0, name='inter_neighbor_%d' % k)
            
            # aggregation
            intra_embeddings = tf.concat([intra_embeddings, neighbor_embedding], axis=1)
            intra_embeddings = tf.matmul(intra_embeddings, self.weights['w_inter_%d' % k]) + self.weights['b_inter_%d' % k]

            norm_embeddings = tf.math.l2_normalize(intra_embeddings, axis=1, name='inter_norm_%d' % k)

            norm_embeddings = tf.nn.leaky_relu(norm_embeddings, name='inter_agg_%d' % k)
            # dropout
            intra_embeddings = tf.nn.dropout(norm_embeddings, rate=self.inter_mess_dropout[k], name='inter_dropout_%d' % k,seed=2022)

            # normalize the distribution of block embeddings
            # norm_embeddings = tf.math.l2_normalize(intra_embeddings, axis=1, name='inter_norm_%d' % k)

            inter_embeddings += [intra_embeddings]
        
        inter_embeddings = tf.concat(inter_embeddings, axis=1)

        self.inter_embeddings = inter_embeddings

    def _build_message_embedding(self):
        """"""
        m_e = tf.nn.embedding_lookup(self.weights['message_embedding'], self.message_token)
        m_e = tf.nn.l2_normalize(m_e, axis=1)

        # average pooling
        m_p = tf.layers.average_pooling1d(
            inputs=m_e,
            pool_size=self.max_token_log, strides=1,
            name='m_p')
        m_p = tf.reshape(m_p, [-1, self.log_dim])

        self.message_embedding = m_p

    def _split_A_hat(self, A:sp.coo_matrix, total_num:int, fold_num:int) -> list:
        """"""
        A_fold_hat = []
        fold_len = total_num // fold_num

        A = A.tocsr()
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == fold_num - 1:
                end = total_num + 1
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(A[start:end]))

        return A_fold_hat
    
    def _split_B(self, B:np.array, n_fold_block:int) -> list:
        """Split B into small parts.
        """
        B_fold = []
        b_fold_len = self.n_block // n_fold_block

        for i_fold in range(n_fold_block):
            start = i_fold * b_fold_len
            if i_fold == n_fold_block - 1:
                end = self.n_block + 1
            else:
                end = (i_fold + 1) * b_fold_len
            B_fold.append(B[start:end])
        
        return B_fold

    def _convert_sp_mat_to_sp_tensor(self, X:sp.csr_matrix) -> tf.SparseTensor:
        """"""
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()

        if len(coo.data) > 0:
            return tf.SparseTensor(indices, coo.data, coo.shape)
        else:
            return tf.SparseTensor(indices=np.empty((0,2), dtype=np.int64), values=coo.data, dense_shape=coo.shape)

    def _build_classification_model_without_message(self) -> None:

        log_blocks_e = tf.nn.embedding_lookup(self.inter_embeddings, self.b_classification)

        # fully connected layer: log_blocks_fc shape: [batch_size, fc_dim]
        self.f_classification_fc = tf.nn.leaky_relu(tf.matmul(log_blocks_e, self.weights['fc_w_out']) + self.weights['fc_b_out'], name='out_layer')

        # log level prediction
        self.classification_prediction = self.f_classification_fc

    def _build_classification_model_with_message(self) -> None:
        """ Combine log block information with log message.
        """
        log_blocks_e = tf.nn.embedding_lookup(self.inter_embeddings, self.b_classification)
        f_classification_fc = tf.nn.leaky_relu(tf.matmul(log_blocks_e, self.weights['fc_w_classification']) + self.weights['fc_b_classification'], name='fc_c_layer')

        self._build_message_embedding()
        f_message_fc = tf.nn.leaky_relu(tf.matmul(self.message_embedding, self.weights['fc_w_message']) + self.weights['fc_b_message'], name='fc_m_layer')
        _out = tf.concat([f_classification_fc, f_message_fc], axis=1)

        self.f_classification_fc = tf.nn.leaky_relu(tf.matmul(_out, self.weights['fc_w_out']) + self.weights['fc_b_out'], name='out_layer')

        self.classification_prediction = self.f_classification_fc
    
    def _build_classification_model(self) -> None:
        """"""
        if self.append_message:
            self._build_classification_model_with_message()
        else:
            self._build_classification_model_without_message()

    def _build_classification_loss_ordinal(self) -> None:
        """"""
        # log level prediction applies cross entropy as the loss function
        classification_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.f_classification_fc,
            labels=self.y_classification,
            name='classification_cross_entropy')
        self.classification_loss = tf.reduce_mean(classification_cross_entropy, name='classification_loss')

         # rep optimization
        if self.opt_type in ['Adam', 'adam']:
            self.classification_opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.classification_loss)
        elif self.opt_type in ['SGD', 'sgd']:
            self.classification_opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.classification_loss)
        elif self.opt_type in ['AdaDelta']:
            self.classification_opt = tf.train.AdadeltaOptimizer(learning_rate=self.lr).minimize(self.classification_loss)
        else:
            logger.error('Optimization is unknown')    
            exit(-1)

        logger.info('Finish building loss for log level prediction.')
    
    def _build_classification_loss_one_hot(self) -> None:
        """"""
        # log level prediction applies cross entropy as the loss function
        classification_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            logits=self.f_classification_fc,
            labels=self.y_classification,
            name='classification_cross_entropy')
        self.classification_loss = tf.reduce_mean(classification_cross_entropy, name='classification_loss')

         # rep optimization
        if self.opt_type in ['Adam', 'adam']:
            self.classification_opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.classification_loss)
        elif self.opt_type in ['SGD', 'sgd']:
            self.classification_opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.classification_loss)
        elif self.opt_type in ['AdaDelta']:
            self.classification_opt = tf.train.AdadeltaOptimizer(learning_rate=self.lr).minimize(self.classification_loss)
        else:
            logger.error('Optimization is unknown')    
            exit(-1)

        logger.info('Finish building loss for log level prediction.')
    
    def _build_classification_loss(self) -> None:
        """"""
        if self.one_hot:
            self._build_classification_loss_one_hot()
        else:
            self._build_classification_loss_ordinal()

    def _statistics_params(self) -> None:
        """"""
        total_parameters = 0
        for var in self.weights:
            shape = self.weights[var].get_shape()
            var_para = 1
            for dim in shape:
                var_para *= dim.value
            logger.debug('Variable name: %s Shape: %d' % (var, var_para))
            total_parameters += var_para

        logger.debug('%s has %d parameters' % (self.model_type, total_parameters))

    def train_classification(self, sess:tf.Session, feed_dict:dict) -> Tuple[float, float]:
        """"""
        run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
        return sess.run([self.classification_opt, self.classification_loss], feed_dict, options=run_options)

    def eval_classification(self, sess:tf.Session, feed_dict:dict) -> list:
        """"""
        return sess.run(self.classification_prediction, feed_dict)

    def get_variables_to_restore(self) -> list:
        """"""
        variables_to_restore = []
        variables = tf.global_variables()
        for v in variables:
            if v.name.split(':')[0] != 'entity_embedding':
                variables_to_restore.append(v)

        return variables_to_restore

    def store_model(self, sess:tf.Session, pretrain_path:str, epoch:int) -> None:
        """"""
        model_save_dir = '%s/%s/%s/%s/%s/' % \
            (pretrain_path, self.model_type, self.lr, self.layer_type, self.reg_type)
        model_save_path = model_save_dir + 'model.weights'
        ensure_dir(model_save_path)

        variables_to_restore = self.get_variables_to_restore()

        model_saver = tf.train.Saver(variables_to_restore, max_to_keep=1)
        model_saver.save(sess, model_save_path, global_step=epoch)

        logger.info("Model save in %s" % model_save_path)

    def load_model(self, sess:tf.Session, pretrain_path:str) -> None:
        checkpoint_path = '%s/%s/%s/%s/%s/checkpoint' % \
            (pretrain_path, self.model_type, self.lr, self.layer_type, self.reg_type)
        
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(checkpoint_path))
        if ckpt and ckpt.all_model_checkpoint_paths:
            logger.info("Load model from %s" % os.path.dirname(checkpoint_path))
            variables_to_restore = self.get_variables_to_restore()
            saver = tf.train.Saver(variables_to_restore)
            saver.restore(sess, ckpt.all_model_checkpoint_paths[0])
        else:
            logger.error("fail to load model in %s" % checkpoint_path)
            exit(-1)
