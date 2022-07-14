import argparse
import numpy as np
from gensim.models import word2vec
from util.helper import ensure_dir
from util.setting import logger


class Type2vec:
    def __init__(self, args:argparse.Namespace) -> None:
        self.type_dim = args.type_dim
        self.window = args.word2vec_window
        self.min_count = args.word2vec_count
        self.worker = args.word2vec_worker
        self.model = None
        self.embedding = None

    def init_embedding(self, e2t_list:list, typetoken_seq:list):
        """"""
        logger.info('Initing type/token embeddings with word2vec')
        model = word2vec.Word2Vec(sentences=typetoken_seq, 
            vector_size=self.type_dim, window=self.window, 
            min_count=self.min_count, workers=self.worker, seed=2022)

        init_embedding = np.zeros(shape=(len(e2t_list), 2*self.type_dim), dtype=np.float32)
        words = list(model.wv.index_to_key)
        
        # Embedding: entity = type || token
        for idx, typetoken in enumerate(e2t_list):
            if typetoken[1] != -1:
                init_embedding[idx] = np.append(model.wv[typetoken[0]], model.wv[typetoken[1]])
            else:
                init_embedding[idx] = np.append(model.wv[typetoken[0]], np.zeros(self.type_dim))

        self.embedding = init_embedding
        self.model = model

    def store_embedding(self, pretrain_path:str):
        """"""
        embedding_save_dir = '%s/word2vec/%s_%s_%s/' % \
            (pretrain_path, self.type_dim, self.window, self.min_count)
        ensure_dir(embedding_save_dir)

        embedding_path = embedding_save_dir + 'word2vec.embedding'

        with open (embedding_path, 'wb') as f:
            np.save(f, self.embedding)

        logger.info("save word2vec embeddings in %s" % embedding_path)

    def load_embedding(self, pretrain_path:str):
        """"""
        embedding_path = '%s/word2vec/%s_%s_%s/word2vec.embedding' % \
            (pretrain_path, self.type_dim, self.window, self.min_count)

        with open(embedding_path, 'rb') as f:
            self.embedding = np.load(embedding_path)

        logger.info("load word2vec embeddings in %s" % embedding_path)

    def print_embedding(self):
        logger.debug(self.embedding)

class Log2vec:
    def __init__(self, args: argparse.Namespace) -> None:
        self.log_dim = args.log_dim
        self.window = args.word2vec_window
        self.min_count = args.word2vec_count
        self.worker = args.word2vec_worker
        self.model = None
        self.embedding = None

    def init_embedding(self, block_tokens: list, message_seq: list):
        """"""
        logger.info('Initing logged message embeddings with word2vec')
        model = word2vec.Word2Vec(sentences=message_seq, 
            vector_size=self.log_dim, window=self.window, 
            min_count=self.min_count, workers=self.worker, seed=2022)

        init_embedding = np.zeros(shape=(len(block_tokens), self.log_dim), dtype=np.float32)
        words = list(model.wv.index_to_key)

        for idx, messagetoken in enumerate(block_tokens):
            if messagetoken in words:
                tmp_embedding = model.wv[messagetoken]
            else:
                tmp_embedding = np.zeros((1,self.log_dim))
            init_embedding[idx] = tmp_embedding
        
        self.embedding = init_embedding
        self.model = model
    
    def store_embedding(self, pretrain_path:str):
        """"""
        embedding_save_dir = '%s/word2vec/%s_%s_%s/' % \
            (pretrain_path, self.log_dim, self.window, self.min_count)
        ensure_dir(embedding_save_dir)

        embedding_path = embedding_save_dir + 'word2vec.log.embedding'

        with open (embedding_path, 'wb') as f:
            np.save(f, self.embedding)

        logger.info("save word2vec embeddings in %s" % embedding_path)

    def load_embedding(self, pretrain_path:str):
        """"""
        embedding_path = '%s/word2vec/%s_%s_%s/word2vec.log.embedding' % \
            (pretrain_path, self.log_dim, self.window, self.min_count)

        with open(embedding_path, 'rb') as f:
            self.embedding = np.load(embedding_path)

        logger.info("load word2vec embeddings in %s" % embedding_path)

    def print_embedding(self):
        logger.debug(self.embedding)