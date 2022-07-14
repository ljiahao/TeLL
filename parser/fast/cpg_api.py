"""This file provides some simple apis of code property graph.
Note: For Java Language, the input must be class file.
"""
from fast.ast_constructor import *
from fast.cfg_constructor import *
from fast.cg_constructor import *
from fast.ddg_constructor import *
from fast.block_splitter import *
from fast.log_capturer import *
from sast.src_parser import *
from util.common import *
from util.helper import *
from util.visualize import *
from util.setting import logger

def ast4singleclass(file_path: str = None) -> list:
    """Generate ast representations for functions in one class.
    """
    if file_path == None:
        logger.error('AST4singleclass lacks file path parameter.')
        exit(-1)
    file_funcs = java_parser(file_path)
    func_list = list()

    for func in file_funcs:
        if not func.has_type('ERROR'):
            ast_cpg = gen_ast_cpg(func.sast)
            func_list.append(ast_cpg)

    return func_list

def cpg4singleclass(file_path: str = None) -> list:
    """Generate cpg representations for functions in one class.
    """
    if file_path == None:
        logger.error('CPG4singleclass lacks fle path parameter.')
        exit(-1)
    
    file_funcs = java_parser(file_path)
    func_list = list()
    for func in file_funcs:
        logger.info(func.file_name, func.func_name)
        if not func.has_type('ERROR'):
            ast_cpg = gen_ast_cpg(func.sast)
            root = func.sast.root
            cfg_build(ast_cpg, root)
            func_list.append(ast_cpg)
    logger.info(len(file_funcs))
    return func_list

def cpg_constructor(func: FunUnit) -> DiGraph:
    """Construct cpg.
    """
    func_root = func.sast.root
    ast_cpg = gen_ast_cpg(func.sast)
    cfg_build(ast_cpg, func_root)

    return ast_cpg

def extract_funcs(dir_path: str = None) -> list:
    """Extract all functions from multiple classes (directory).
    """
    if dir_path == None:
        logger.error('Cpg4multiclass lacks directory path.')
        exit(-1)
    files = traverse_src_files(dir_path, 'java')
    logger.error(len(files))
    func_list = list()
    logger.info('Extract functions...')

    for file in files:
        file_func = java_parser(file)
        logger.info(f'Parsing file: {file}')
        if len(file_func) == 0:
            logger.warn(f'Cannot extract functions from {file}')
        for func in file_func:
            if not func.has_type('ERROR'):
                func_list.append(func)
            else:
                logger.error(f'File: {file} \t function: {func.func_name} has ERROR Type.')
                exit(-1)

    return func_list

def cpg4multifiles(func_list: list) -> dict:
    """Generate cpg for multi files.
    """
    logger.info('Start generating CPG Dict...')
    cpg_dict = cg_dict_constructor(func_list)

    return cpg_dict

def fast_builder(dir_path: str) -> list:
    """
    """
    if dir_path == None:
        logger.error('FAST directory wrong.')
        exit(-1)
    
    files = traverse_src_files(dir_path, 'java')
    logger.info(f'Total File: {len(files)}')
    
    func_list = list()
    logged_blocks = list()
    cg_dict = dict()

    for file in files:
        file_funcs, serial_code = java_parser(file)
        logger.info(f'Parsing file: {file}')
        if len(file_funcs) == 0:
            logger.warning(f'Cannot extract functions from {file}')
            exit(-1)
        dataset_name = extract_dataset_name(file)
        for func in file_funcs:
            if func.has_type('ERROR'):
                logger.error(f'Exist ERROR in {func.file_name} {func.func_name}')
                exit(-1)
            func_list.append(func)
            ast_cpg = gen_ast_cpg(func.sast)
            root = func.sast.root
            entrynode, fringe = cfg_build(ast_cpg, root)
            # logger.warning(f'Func Name: {func.func_name}')
            identify_block(ast_cpg, root)
            check_match_block(ast_cpg)
            logged_blocks += find_logged_blocks(ast_cpg, root, serial_code, dataset_name)
            callees = list_all_callees(ast_cpg, root, dataset_name)
            func_cgnode = CGNode(func.file_name, func.import_header, func.func_name, func.parameter_type, func.parameter_name, entrynode, fringe, ast_cpg, callees)

            key = dataset_name + '-' + func.func_name + '-' + str(len(func.parameter_type))
            if key not in cg_dict.keys():
                cg_dict[key] = [func_cgnode]
            else:
                cg_dict[key].append(func_cgnode)

    return [func_list, cg_dict, logged_blocks]

