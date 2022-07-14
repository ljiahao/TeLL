"""Provide some basic help functions.
"""
import os
import pickle

from networkx import DiGraph

from fast.ast_constructor import gen_ast_cpg
from fast.cfg_constructor import cfg_build
from fast.log_block import LoggedBlock
from sast.fun_unit import FunUnit

from util.visualize import visualize_ast_cfg
from util.setting import logger


def traverse_files(dir_path: str = None) -> list:
    all_files = list()
    walk_tree = os.walk(dir_path)
    for root, _, files in walk_tree:
        for file in files:
            all_files.append(os.path.join(root, file))
    
    return all_files

def check_extension(file_name: str, extension: str) -> bool:
    _extension = os.path.splitext(file_name)[-1][1:]
    if _extension == extension:
        return True
    return False

def traverse_src_files(dir_path: str, extension: str) -> list:
    """Obtain all source files we want to parse.

    attributes:
        dir_path -- the directory path we want to parse.
        extension -- the file extension we want to parse (e.g., 'java')
    
    returns:
        files -- list including files we want to parse.
    """
    files = list()
    all_files = traverse_files(dir_path)
    for file in all_files:
        if check_extension(file, extension):
            files.append(file)
    
    return files

def check_key_repeat(entities: list) -> bool:
    _exist_key = list()
    for entity in entities:
        node_key = entity[0]
        if node_key not in _exist_key:
            _exist_key.append(node_key)
        else:
            return True
    
    return False

def visualize_helper(func: FunUnit, file_name: str) -> bool:
    func_root = func.sast.root
    func_cpg = gen_ast_cpg(func.sast)
    _, _ = cfg_build(func_cpg, func_root)
    visualize_ast_cfg(func_cpg, file_name)

    return True

def load_inter_results(dir_path: str) -> list:
    """Load inter results of function list and function dict
    """
    func_list_path = os.path.join(dir_path, 'func_list.pkl')
    func_dict_path = os.path.join(dir_path, 'func_dict.pkl')
    logged_block_path = os.path.join(dir_path, 'logged_block.pkl')

    if not os.path.exists(func_list_path) or not os.path.exists(func_dict_path):
        logger.error('Cannot find stored inter results.')
        exit(-1)

    fl = open(func_list_path, 'rb')
    func_list = pickle.load(fl)
    fl.close()

    fd = open(func_dict_path, 'rb')
    func_dict = pickle.load(fd)
    fd.close()

    lb = open(logged_block_path, 'rb')
    logged_block = pickle.load(lb)
    lb.close()

    return [func_list, func_dict, logged_block] 

def store_inter_results(dir_path: str, func_list: list, func_dict: dict, logged_block: list) -> bool:
    """Store inter results of function list and function dict, logged block
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    func_list_path = os.path.join(dir_path, 'func_list.pkl')
    func_dict_path = os.path.join(dir_path, 'func_dict.pkl')
    logged_block_path = os.path.join(dir_path, 'logged_block.pkl')

    try:
        fl = open(func_list_path, 'wb')
        pickle.dump(func_list, fl)
        fl.close()

        fd = open(func_dict_path, 'wb')
        pickle.dump(func_dict, fd)
        fd.close()

        lb = open(logged_block_path, 'wb')
        pickle.dump(logged_block, lb)
        lb.close()

        return True
    except:
        return False

def compare_loglevel(b1: LoggedBlock, b2: LoggedBlock) -> bool:
    """if level of b1 is higher than b2, return true
    """
    level_num = {'trace':1, 'debug':2, 'info':3, 'warn':4, 'error':5, 'fatal':6}
    l1 = level_num[b1.level]
    l2 = level_num[b2.level]

    return l1 > l2

def clean_logged_block(logged_blocks: list) -> list:
    """Find the highest level logging statement for each logged block.
    """
    res_dict = dict()
    
    for block in logged_blocks:
        key = block.block_id
        if key not in res_dict:
            res_dict[key] = block
        else:
            if compare_loglevel(block, res_dict[key]):
                res_dict[key] = block
    
    return list(res_dict.values())

def check_match_block(cpg: DiGraph) -> None:
    edges = list(cpg.edges)

    for edge in edges:
        start, end = edge
        if cpg[start][end]['edge_type'] == '010':
            if not cpg.nodes[start]['cpg_node'].match_block:
                logger.error('Start not be block id')
                logger.error(cpg.nodes[start]['cpg_node'].node_type)
                exit(-1)
            if not cpg.nodes[end]['cpg_node'].match_block:
                logger.error('End not be block id')
                logger.error(cpg.nodes[end]['cpg_node'].node_type)
                exit(-1)

def extract_dataset_name(file_name: str) -> str:
    """Extract dataset name from file name"""
    _datasets = ['cassandra', 'elasticsearch', 'flink', 'hbase', 'jmeter', 'kafka', 'karaf', 'wicket', 'zookeeper']
    _file = file_name.split('/')[-1].split('_', 1)[0]
    if _file not in _datasets:
        logger.error(f'Unknown dataset name {_file}')
        exit(-1)
    
    return _file
