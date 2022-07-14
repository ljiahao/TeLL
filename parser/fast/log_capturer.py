from networkx import DiGraph
from util.data_structure import Queue
from util.setting import logger

from fast.log_block import LoggedBlock

def is_logging_statement(cpg: DiGraph, node: str, serial_code: bytes, dataset_name: str) -> list:
    """Check whether current statement is logging statement. 

    If current statement is logging statement, return start and end index, else, return False.
    """
    if not cpg.nodes[node]['cpg_node'].node_type == 'method_invocation':
        logger.error('Node is not method invocation.')
        exit(-1)
    
    level_prix = ['log', 'logger', 'logging', 'getlogger', 'getlog']
    levels = ['trace', 'debug', 'info', 'warn', 'error', 'fatal']

    token_ids = list()
    tokens = list()

    queue = Queue()
    visited = list()

    queue.push(node)

    while not queue.is_empty():
        current_node = queue.pop()
        current_node_token = cpg.nodes[current_node]['cpg_node'].node_token
        tokens.append(current_node_token)
        token_ids.append(current_node)
        visited.append(current_node)
        current_node_successors = list(cpg.successors(current_node))
        for _successor in current_node_successors:
            edge_type = cpg[current_node][_successor]['edge_type']
            if _successor not in visited and edge_type == '100':
                queue.push(_successor)
    
    tokens = [x.lower() for x in tokens]

    _prix = list(set(level_prix) & set(tokens))
    _level = list(set(levels) & set(tokens))

    if _prix and _level:
        log_level = extract_level(_level)
        log_id = extract_blk_id(cpg, node)
        if log_id is None:
            logger.error(f'Cannot find block id for {node}')
            exit(-1)
        start_idx = cpg.nodes[node]['cpg_node'].start_idx
        end_idx = cpg.nodes[node]['cpg_node'].end_idx
        log_content = extract_content(serial_code, start_idx, end_idx).replace('\n', ' ')
        logged_block = LoggedBlock(log_id, log_level, log_content, start_idx, end_idx, dataset_name)

        token_ids.remove(node)
        cpg.remove_nodes_from(token_ids)

        return [True, logged_block]
    
    return [False, None]

def extract_level(levels: list) -> str:
    if len(levels) == 1:
        return levels[0]
    else:
        if 'fatal' in levels:
            return 'fatal'
        elif 'error' in levels:
            return 'error'
        elif 'warn' in levels:
            return 'warn'
        elif 'info' in levels:
            return 'info'
        elif 'debug' in levels:
            return 'debug'
        elif 'trace' in levels:
            return 'trace'

def extract_blk_id(cpg: DiGraph, node: str) -> str:
    """Back tracking to find block id.
    """
    queue = Queue()
    visited = list()
    queue.push(node)
    
    blk_id = None
    while not queue.is_empty():
        current_node = queue.pop()
        if cpg.nodes[current_node]['cpg_node'].match_block:
            blk_id = current_node
            break
        visited.append(current_node)
        
        node_parents = list(cpg.predecessors(current_node))
        for parent in node_parents:
            if parent not in visited and cpg[parent][current_node]['edge_type'] == '100':
                queue.push(parent)
    
    return blk_id
        
def extract_content(serial_code: bytes, start_idx: int, end_idx: int) -> str:
    """Extract content of logging statement.
    """
    content = serial_code[start_idx:end_idx].decode('utf8')

    return content

def find_logged_blocks(cpg: DiGraph, node: str, serial_code: bytes, dataset_name: str) -> list:
    """Go through the whole cpg to find logged blocks.
    """
    logged_blocks = list()
    queue = Queue()
    visited = list()

    queue.push(node)

    while not queue.is_empty():
        current_node = queue.pop()
        current_node_type = cpg.nodes[current_node]['cpg_node'].node_type
        if current_node_type == 'method_invocation':
            is_log, logged_block = is_logging_statement(cpg, current_node, serial_code, dataset_name)
            if is_log:
                logged_blocks.append(logged_block)
        visited.append(current_node)
        current_node_successors = list(cpg.successors(current_node))
        for _successor in current_node_successors:
            if _successor not in visited:
                queue.push(_successor)
    
    replace_log_guard(cpg)

    return logged_blocks

def replace_log_guard(cpg: DiGraph) -> None:
    """Replace log guard (e.g., isDebugEnabled, isTraceEnabled) to empty.
    """
    log_guards = ['istraceenabled', 'isdebugenabled', 'isinfoenabled', 'iswarnenabled', 'iserrorenabled', 'isfatalenabled']

    all_nodes = list(cpg.nodes)

    for node in all_nodes:
        node_token = cpg.nodes[node]['cpg_node'].node_token.lower()
        if node_token in log_guards:
            cpg.nodes[node]['cpg_node'].node_token = ''