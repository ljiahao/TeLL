import argparse
from typing import Tuple
import scipy.sparse as sp
import numpy as np
import random as rd
from sklearn.model_selection import train_test_split

from util.setting import logger
from util.load_data import Data

class GNNLoader(Data):
    def __init__(self, args:argparse.Namespace) -> None:
        super().__init__(args)
        self.adj_type = args.adj_type

        # generate sparse adjacency matrices for system entity inter_data
        logger.info("Converting interactions (ast & cfg) into sparse adjacency matrix")
        intra_adj_list = self._get_relational_adj_list(self.intra_data, self.n_entity, True)
        inter_adj_list = self._get_relational_adj_list(self.inter_data, self.n_block, False)

        # generate normalized (sparse adjacency) metrices
        logger.info("Generating normalized sparse adjacency matrix")
        self.intra_norm_list = self._get_relational_norm_list(intra_adj_list)
        self.inter_norm_list = self._get_relational_norm_list(inter_adj_list)

        # load the norm matrix (used for information propagation)
        self.A_intra = sum(self.intra_norm_list)
        self.A_inter = sum(self.inter_norm_list)

        # mess_dropout
        self.intra_mess_dropout = eval(args.intra_mess_dropout)
        self.inter_mess_dropout = eval(args.inter_mess_dropout)

        # split logged blocks into training/validation/testing sets for log level prediction
        logger.info('Generating log level prediction training, validation, and testing sets')
        self.hbgn_val_size = args.hbgn_val_size
        self.hbgn_test_size = args.hbgn_test_size
        self.one_hot = args.one_hot

        self.hbgn_train_data, self.hbgn_val_data, self.hbgn_test_data = self._sample_hbgn_split()

        # statistics
        self.n_hbgn_train, self.n_hbgn_val, self.n_hbgn_test = len(self.hbgn_train_data[0]), len(self.hbgn_val_data[0]), len(self.hbgn_test_data[0])
        logger.debug('HBGN [n_train, n_val, n_test] = [%d, %d, %d]' % (self.n_hbgn_train, self.n_hbgn_val, self.n_hbgn_test))

        # batch iter
        self.hbgn_data_iter = self.n_hbgn_train // self.batch_size_tell
        if self.n_hbgn_train % self.batch_size_tell:
            self.hbgn_data_iter += 1
        
        # block array
        self.blocks = self._build_sorted_block_array()

    def _get_relational_adj_list(self, inter_data, matrix_size, ghost) -> Tuple[list, list]:
        def _np_mat2sp_adj(np_mat:np.array, matrix_size, row_pre=0, col_pre=0) -> Tuple[sp.coo_matrix, sp.coo_matrix]:
            # to-node interaction: A: A->B
            a_rows = np_mat[:, 0] + row_pre # all As
            a_cols = np_mat[:, 1] + col_pre # all Bs
            # must use float 1. (int is not allowed)
            a_vals = [1.] * len(a_rows)

            # from-node interaction: A: B->A
            b_rows = a_cols
            b_cols = a_rows
            b_vals = [1.] * len(b_rows)

            # self.n_entity + 1: 
            # we add a `ghost` entity to support parallel AST node embedding 
            # retrival for program statements
            a_adj = sp.coo_matrix((a_vals, (a_rows, a_cols)), shape=(matrix_size, matrix_size))
            b_adj = sp.coo_matrix((b_vals, (b_rows, b_cols)), shape=(matrix_size, matrix_size))

            return a_adj, b_adj

        adj_mat_list = []

        if ghost:
            matrix_size += 1

        r, r_inv = _np_mat2sp_adj(inter_data, matrix_size)
        adj_mat_list.append(r)
        # Todo: whether r_inv (inverse directions) helps infer code representations
        adj_mat_list.append(r_inv)

        return adj_mat_list
    
    def _get_relational_norm_list(self, adj_list:str) -> list:
        # Init for 1/Nt
        def _si_norm(adj):
            rowsum = np.array(adj.sum(axis=1))
            # np.power(rowsum, -1).flatten() may trigger divide by zero
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj = d_mat_inv.dot(adj)
            return norm_adj

        # Init for 1/(Nt*Nh)^(1/2)
        def _bi_norm(adj):
            rowsum = np.array(adj.sum(axis=1))
            # np.power(rowsum, -1).flatten() may trigger divide by zero
            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
            # bi_norm = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
            bi_norm = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
            return bi_norm

        if self.adj_type == 'bi':
            norm_list = [_bi_norm(adj) for adj in adj_list]
        else:
            norm_list = [_si_norm(adj) for adj in adj_list]

        return norm_list

    def _sample_hbgn_split(self):
        """"""
        hbgn_train_data, hbgn_test_val_data, hbgn_train_data_label, hbgn_test_val_data_label = train_test_split(self.logged_blocks, self.logged_blocks_level, test_size=self.hbgn_val_size+self.hbgn_test_size, random_state=2022, stratify=self.logged_blocks_level)

        hbgn_test_data, hbgn_val_data, hbgn_test_data_label, hbgn_val_data_label = train_test_split(hbgn_test_val_data, hbgn_test_val_data_label, test_size=self.hbgn_test_size/(self.hbgn_test_size+self.hbgn_val_size), random_state=2022, stratify=hbgn_test_val_data_label)
  
        hbgn_train_data = [int(x) for x in hbgn_train_data]
        hbgn_train_data_label = [eval(x) for x in hbgn_train_data_label]
        hbgn_val_data = [int(x) for x in hbgn_val_data]
        hbgn_val_data_label = [eval(x) for x in hbgn_val_data_label]
        hbgn_test_data = [int(x) for x in hbgn_test_data]
        hbgn_test_data_label = [eval(x) for x in hbgn_test_data_label]

        

        return [hbgn_train_data, hbgn_train_data_label], [hbgn_val_data, hbgn_val_data_label], [hbgn_test_data, hbgn_test_data_label]


    def _build_sorted_block_array(self) -> np.array:
        """Build block array for generating block embeddings.
        """
        block_entities = []
        for idx in range(self.n_block):
            block_entities.append(self.block_dict[idx])
        
        if len(block_entities) != self.n_block:
            logger.error('Cannot construct block entities matrix')
            exit(-1)
        
        return np.array(block_entities)

    def _convert_csr_to_sparse_tensor_inputs(self, X:sp.csr_matrix) -> Tuple[list, list, list]:
        """"""
        coo = X.tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
        
        return indices, coo.data, coo.shape
    
    def _convert_log_level_encoding_schema(self, logged_block_labels:list) -> list:
        """Convert ordinal encoding to one-hot encoding"""
        one_hot_labels = list()
        
        for label in logged_block_labels:
            label_sum = sum(label)
            if label_sum == 1:
                one_hot_labels.append([1,0,0,0,0])
            elif label_sum == 2:
                one_hot_labels.append([0,1,0,0,0])
            elif label_sum == 3:
                one_hot_labels.append([0,0,1,0,0])
            elif label_sum == 4:
                one_hot_labels.append([0,0,0,1,0])
            elif label_sum == 5:
                one_hot_labels.append([0,0,0,0,1])
            else:
                logger.error('Unknown logged block label')
                exit(-1)
        
        return one_hot_labels

    def generate_hbgn_train_batch(self, i_batch:int) -> dict:
        """"""
        batch_data = {}

        start = i_batch * self.batch_size_tell
        if i_batch == self.hbgn_data_iter:
            end = self.n_hbgn_train
        else:
            end = (i_batch + 1) * self.batch_size_tell

        log_blocks = self.hbgn_train_data[0][start:end]
        log_block_labels = self.hbgn_train_data[1][start:end]
        
        if self.one_hot:
            log_block_labels = self._convert_log_level_encoding_schema(log_block_labels)

        batch_data['b_classification'] = log_blocks
        batch_data['y_classification'] = log_block_labels
        batch_data['message_token'] = [self.block_messages[k] for k in log_blocks]

        return batch_data

    def generate_hbgn_val_batch(self, i_batch:int, last_batch:bool) -> dict:
        """"""
        batch_data = {}
        start = i_batch * self.batch_size_tell
        if last_batch:
            end = self.n_hbgn_val
        else:
            end = (i_batch + 1) * self.batch_size_tell

        log_blocks = self.hbgn_val_data[0][start:end]
        log_block_labels = self.hbgn_val_data[1][start:end]

        if self.one_hot:
            log_block_labels = self._convert_log_level_encoding_schema(log_block_labels)

        batch_data['b_classification'] = log_blocks
        batch_data['y_classification'] = log_block_labels
        batch_data['message_token'] = [self.block_messages[k] for k in log_blocks]

        return batch_data

    def generate_hbgn_test_batch(self, i_batch:int, last_batch:bool) -> dict:
        """"""
        batch_data = {}
        start = i_batch * self.batch_size_tell
        if last_batch:
            end = self.n_hbgn_test
        else:
            end = (i_batch + 1) * self.batch_size_tell

        log_blocks = self.hbgn_test_data[0][start:end]
        log_block_labels = self.hbgn_test_data[1][start:end]

        if self.one_hot:
            log_block_labels = self._convert_log_level_encoding_schema(log_block_labels)

        batch_data['b_classification'] = log_blocks
        batch_data['y_classification'] = log_block_labels
        batch_data['message_token'] = [self.block_messages[k] for k in log_blocks]

        return batch_data

    def generate_hbgn_train_feed_dict(self, model, batch_data):
        feed_dict = {
            # Classification
            model.b_classification: batch_data['b_classification'],
            model.y_classification: batch_data['y_classification'],
            model.message_token: batch_data['message_token'],
            # hardcode dropping probability as 0
            model.intra_mess_dropout: [0.2,0.2,0.2,0.2,0.2,0.2],
            model.inter_mess_dropout: [0.2,0.2,0.2,0.2,0.2,0.2],
        }
        
        return feed_dict

    def generate_hbgn_val_feed_dict(self, model, batch_data):
        feed_dict = {
            model.b_classification: batch_data['b_classification'],
            model.message_token: batch_data['message_token'],
            # hardcode dropping probability as 0
            model.intra_mess_dropout: [0,0,0,0,0,0],
            model.inter_mess_dropout: [0,0,0,0,0,0],
        }
        return feed_dict
    
    def generate_hbgn_test_feed_dict(self, model, batch_data):
        feed_dict = {
            model.b_classification: batch_data['b_classification'],
            model.message_token: batch_data['message_token'],
            # hardcode dropping probability as 0
            model.intra_mess_dropout: [0,0,0,0,0,0],
            model.inter_mess_dropout: [0,0,0,0,0,0],
        }
        return feed_dict
    
    def shuffle_train_data(self):
        """Shuffle train data"""
        merge_data = list(zip(self.hbgn_train_data[0], self.hbgn_train_data[1]))
        rd.shuffle(merge_data)
        a, b = zip(*merge_data)
        self.hbgn_train_data = [a, b]