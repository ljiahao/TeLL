import os
import argparse
import random
import numpy as np
from typing import Tuple

from numpy.core.shape_base import block
from util.setting import logger
from util.helper import ensure_dir, exist_dir

class Data(object):
    def __init__(self, args:argparse.Namespace) -> None:
        super().__init__()
        # batch_size for tell
        self.batch_size_tell = args.batch_size_tell

        # encoding and embedding file paths
        self.in_path, self.out_path = self.init_path(args.dataset)

        # init file_path to load cpg
        self.entity_file = os.path.join(self.in_path, 'entity2id.txt')
        exist_dir(self.entity_file)
        self.ast_file = os.path.join(self.in_path, 'ast2id.txt')
        exist_dir(self.ast_file)
        self.cfg_file = os.path.join(self.in_path, 'connected_edge.txt')
        exist_dir(self.cfg_file)
        self.block2id_file = os.path.join(self.in_path, 'connected_block_id.txt')
        exist_dir(self.block2id_file) 
        self.block2entity_file = os.path.join(self.in_path, 'connected_block_entity.txt')
        exist_dir(self.block2entity_file)
        self.loglevel2id_file = os.path.join(self.in_path, 'connected_logged_block_level.txt')
        exist_dir(self.loglevel2id_file)
        self.typetoken2id_file = os.path.join(self.in_path, 'typetoken2id.txt')
        exist_dir(self.typetoken2id_file)
        self.typetoken_seq_file = os.path.join(self.in_path, 'typetoken_seq.txt')
        exist_dir(self.typetoken_seq_file)
        self.entity2typetoken_file = os.path.join(self.in_path, 'entity2typetoken.txt')
        exist_dir(self.entity2typetoken_file)
        self.logtoken2id_file = os.path.join(self.in_path, 'logtoken2id.txt')
        exist_dir(self.logtoken2id_file)
        self.logtokenseq_file = os.path.join(self.in_path, 'connected_log_message_sequence.txt')
        exist_dir(self.logtokenseq_file)

        # extract entity to AST type/token information
        self.e2t_list = self._extract_entity2typetoken()
        self.typetoken_seq = self._extract_typetoken_seq()

        # extract log tokens from logged blocks
        self.logtoken_seq = self._extract_logtoken_seq()

        # extract log level for logged blocks
        self.logged_blocks, self.logged_blocks_level = self._extract_block_level()

        # collect statistic info about the dataset
        self.n_typetoken, self.n_entity, self.n_block, self.n_ast, self.n_cfg, self.n_log_token = self._load_fast_stat()

        # extract entities of code blocks
        logger.info('Extracting blocks')
        self.max_entity_block = args.max_block_entity
        self.max_token_log = args.max_token_log
        self.block_dict = self._extract_block()
        self.block_messages = self._extract_block_message()
        self.log_tokens = self._extract_log_tokens()

        # extract interactions from fast
        logger.info('Extracting interactions')
        self.intra_data = self._extract_intra()
        self.inter_data = self._extract_inter()

        self._print_fast_info()
        self._print_logged_block_info()

    def _extract_entity2typetoken(self) -> list:
        """"""
        e2t_list = []

        with open(self.entity2typetoken_file, 'r') as f:
            next(f)
            for line in f.readlines():
                e2t = line.strip().split(',')
                type = e2t[1]
                token = e2t[2]
                e2t_list.append([int(type), int(token)])
        
        return e2t_list
        
    def _extract_typetoken_seq(self) -> list:
        """"""
        typetoken_seq = []

        with open(self.typetoken_seq_file, 'r') as f:
            next(f)
            for line in f.readlines():
                seq = list(map(int, line.strip().split(',')))
                typetoken_seq.append(seq)

        return typetoken_seq

    def _extract_logtoken_seq(self) -> list:
        """"""
        logtoken_seq = []

        with open(self.logtokenseq_file, 'r') as f:
            next(f)
            for line in f.readlines():
                seq = list(map(int, line.strip().split(',')))
                logtoken_seq.append(seq[1:])
        
        return logtoken_seq
    
    def _extract_log_tokens(self) -> list:
        """"""
        log_tokens = []
        
        with open(self.logtoken2id_file, 'r') as f:
            next(f)
            for line in f.readlines():
                line = line.strip().split(',')
                log_tokens.append(int(line[0]))
        
        return log_tokens

    def _extract_block_message(self) -> dict:
        """"""
        block_messages = dict()
        
        with open(self.logtokenseq_file, 'r') as f:
            next(f)
            for line in f.readlines():
                message_tokens = [int(i) for i in line.strip().split(',')]
                if len(list(set(message_tokens))) < self.max_token_log + 1:
                    block_messages[message_tokens[0]] = [e for e in list(set(message_tokens[1:]))]
                else:
                    block_messages[message_tokens[0]] = [e for e in list(set(message_tokens))[1:self.max_token_log+1]]
            for block in block_messages:
                if len(block_messages[block]) < self.max_token_log:
                    zero_padding = [self.n_log_token] * (self.max_token_log - len(block_messages[block]))
                    block_messages[block].extend(zero_padding)
        
        return block_messages

    def init_path(self, dataset:str) -> Tuple[str, str]:
        """ """
        # in_path defines where to load code encodings
        in_path = os.path.abspath(os.path.join('data', dataset))
        exist_dir(in_path)

        # output_path defines where to save code embeddings (representations)
        out_path = os.path.abspath(os.path.join('pretrain', dataset))
        ensure_dir(out_path)

        return in_path, out_path

    def _load_fast_stat(self) -> Tuple[int, int, int, int, int]:
        """ """
        with open(self.typetoken2id_file, 'r') as f:
            n_typetoken = int(f.readline().strip())
        with open(self.entity_file, 'r') as f:
            n_entity = int(f.readline().strip())
        with open(self.block2id_file, 'r') as f:
            n_block = int(f.readline().strip())
        with open(self.ast_file, 'r') as f:
            n_ast = int(f.readline().strip())
        with open(self.cfg_file, 'r') as f:
            n_cfg = int(f.readline().strip())
        with open(self.logtoken2id_file, 'r') as f:
            n_log_token = int(f.readline().strip()) 

        return n_typetoken, n_entity, n_block, n_ast, n_cfg, n_log_token

    def _print_fast_info(self):
        """ """
        logger.debug('FAST statistics')
        logger.debug('[n_typetoken, n_entity, n_block, n_log_token] = [%d, %d, %d, %d]'
        % (self.n_typetoken, self.n_entity, self.n_block, self.n_log_token))
        logger.debug('[n_ast, n_cfg] = [%d, %d]' % (self.n_ast, self.n_cfg))
        logger.debug('[max n_entity of a block] = [%d]' % self.max_entity_block)
        logger.debug('[max n_token of a logging statement] = [%d]' % self.max_token_log)
    
    def _print_logged_block_info(self):
        """"""
        logger.debug('Logged block statistics')
        logger.debug('[n_trace, n_debug, n_info, n_warn, n_error, n_total] = [%d, %d, %d, %d, %d, %d]' %(self.n_trace, self.n_debug, self.n_info, self.n_warn, self.n_error, self.n_logged_block))

    def _extract_intra(self) -> list:
        """Extract intra block edges (AST).
        """
        intra_mat = list()

        with open(self.ast_file, 'r') as f:
            next(f)
            for line in f.readlines():
                triple = line.strip().split(',')
                h_id = int(triple[0])
                t_id = int(triple[1])
                intra_mat.append([h_id, t_id])
        
        intra_data = np.array(intra_mat)

        return intra_data
    
    def _extract_inter(self) -> list:
        """Extract inter block edges (CFG).
        """
        inter_mat = list()

        with open(self.cfg_file, 'r') as f:
            next(f)
            for line in f.readlines():
                triple = line.strip().split(',')
                h_id = int(triple[0])
                t_id = int(triple[1])
                inter_mat.append([h_id, t_id])
        
        inter_data = np.array(inter_mat)

        return inter_data
    
    def _extract_block(self) -> dict:
        block_dict = dict()

        with open(self.block2entity_file, 'r') as f:
            next(f)
            for line in f.readlines():
                entities = [int(i) for i in line.strip().split(',')]
                if len(list(set(entities))) < self.max_entity_block + 1:
                    block_dict[entities[0]] = [e for e in list(set(entities[1:]))]
                else:
                    block_dict[entities[0]] = [e for e in list(set(entities))[1:self.max_entity_block+1]]
            for block in block_dict:
                if len(block_dict[block]) < self.max_entity_block:
                    # self.n_entity is a `ghost` entity
                    zero_padding = [self.n_entity] * (self.max_entity_block - len(block_dict[block]))
                    block_dict[block].extend(zero_padding)
        
        return block_dict


    def _sampling_block(self, blocks: list, block_levels: list) -> Tuple[list, list]:
        """Repeat small logged blocks.

        Note: if blocks for some levels is less than 5,
        just repeat some block to increase it to 5.
        """
        res_blocks, res_blocks_level = [], []
        if len(blocks) == 0:
            logger.error('Block level number is 0')
            exit(-1)
        for _ in range(5):
            res_blocks.append(random.sample(blocks, 1)[0])
            res_blocks_level.append(block_levels[0])

        return res_blocks, res_blocks_level

    def _extract_block_level(self) -> Tuple[list, list]:
        """Extract levels of logged block.
        """
        visited = list()
        self.n_trace, self.n_debug, self.n_info, self.n_warn, self.n_error = 0, 0, 0, 0, 0
        trace_blocks, trace_blocks_level = [], []
        debug_blocks, debug_blocks_level = [], []
        info_blocks, info_blocks_level = [], []
        warn_blocks, warn_blocks_level = [], []
        error_blocks, error_blocks_level = [], []
        logged_blocks, logged_blocks_level = [], []

        with open(self.loglevel2id_file, 'r') as f:
            self.n_logged_block = int(f.readline().strip())
            for line in f.readlines():
                b2l = line.strip().split(',', 1)
                block_id = b2l[0]
                block_level = b2l[1]
                if block_level == '[1,0,0,0,0]':
                    self.n_trace += 1
                    trace_blocks.append(block_id)
                    trace_blocks_level.append(block_level)
                elif block_level == '[1,1,0,0,0]':
                    self.n_debug += 1
                    debug_blocks.append(block_id)
                    debug_blocks_level.append(block_level)
                elif block_level == '[1,1,1,0,0]':
                    self.n_info += 1
                    info_blocks.append(block_id)
                    info_blocks_level.append(block_level)
                elif block_level == '[1,1,1,1,0]':
                    self.n_warn += 1
                    warn_blocks.append(block_id)
                    warn_blocks_level.append(block_level)
                elif block_level == '[1,1,1,1,1]':
                    self.n_error += 1
                    error_blocks.append(block_id)
                    error_blocks_level.append(block_level)
                else:
                    logger.error(f'Cannot identify log level')
                    exit(-1)

                if block_id not in visited:
                    visited.append(block_id)
                else:
                    pass
                    # logger.warning(f'Appear repeated logged block')
                    # exit(-1)
        if self.n_trace < 5:
            trace_blocks, trace_blocks_level = self._sampling_block(trace_blocks, trace_blocks_level)
            self.n_trace = 5
        if self.n_debug < 5:
            debug_blocks, debug_blocks_level = self._sampling_block(debug_blocks, debug_blocks_level)
            self.n_debug = 5
        if self.n_info < 5:
            info_blocks, info_blocks_level = self._sampling_block(info_blocks, info_blocks_level)
            self.n_info = 5
        if self.n_warn < 5:
            warn_blocks, warn_blocks_level = self._sampling_block(warn_blocks, warn_blocks_level)
            self.n_warn = 5
        if self.n_error < 5:
            error_blocks, error_blocks_level = self._sampling_block(error_blocks, error_blocks_level)
            self.n_error = 5
        logged_blocks = trace_blocks + debug_blocks + info_blocks + warn_blocks + error_blocks
        logged_blocks_level = trace_blocks_level + debug_blocks_level + info_blocks_level + warn_blocks_level + error_blocks_level

        if len(logged_blocks) != len(logged_blocks_level):
            logger.error('Block and label not match')
            exit(-1)

        return logged_blocks, logged_blocks_level