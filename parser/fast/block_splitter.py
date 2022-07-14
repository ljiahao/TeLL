from networkx import DiGraph
from fast.cpg_node import CPGNode
from sast.ast_node import ASTNode
from util.data_structure import Queue
from util.setting import logger
import uuid

block_start = ['method_declaration', 'if_statement', 'for_statement', 'enhanced_for_statement', 'while_statement', 'do_statement', 'try_statement', 'try_with_resources_statement', 'catch_clause', 'finally_clause', 'synchronized_statement', 'switch_block_statement_group', 'ret_type', 'block', 'switch_expression']

def check_parent_num(cpg: DiGraph, node: str) -> int:
    """Check parent number of current node (control flow parents).
    """
    cfg_parents = 0
    parents = list(cpg.predecessors(node))
    for parent in parents:
        edge_type = cpg[parent][node]['edge_type']
        if edge_type in ['010', '110']:
            cfg_parents += 1
    
    return cfg_parents

def has_same_ast_parent(cpg: DiGraph, n1: str, n2: str) -> list:
    """Determine whether two nodes has the same ast parent.
    """
    n1_parents = list(cpg.predecessors(n1))
    n2_parents = list(cpg.predecessors(n2))

    n1_ast_parent = None
    n2_ast_parent = None

    for n1_parent in n1_parents:
        if cpg[n1_parent][n1]['edge_type'] in ['100', '110']:
            n1_ast_parent = n1_parent
            break
    
    for n2_parent in n2_parents:
        if cpg[n2_parent][n2]['edge_type'] in ['100', '110']:
            n2_ast_parent = n2_parent
            break
    
    is_same =  (n1_ast_parent == n2_ast_parent)

    return [is_same, n1_ast_parent]

def update_cfg_edge(cpg: DiGraph, node: str) -> None:
    """Modify control flow edge type according to block node.
    """
    # two situation: block statement or non-block statement
    node_parents = list(cpg.predecessors(node))
    if cpg.nodes[node]['cpg_node'].match_block:
        for parent in node_parents:
            if cpg[parent][node]['edge_type'] == '100':
                cpg.remove_edge(parent, node)
            elif cpg[parent][node]['edge_type'] == '110':
                parent_type = cpg.nodes[parent]['cpg_node'].node_type
                if parent_type == 'block':
                    cpg[parent][node]['edge_type'] = '100'
                    cpg.nodes[node]['cpg_node'].match_block = False
                else:
                    cpg[parent][node]['edge_type'] = '010'
    else:
        for parent in node_parents:
            if cpg[parent][node]['edge_type'] == '110':
                cpg[parent][node]['edge_type'] = '100'
            elif cpg[parent][node]['edge_type'] in ['010']:
                is_same, n1_parent = has_same_ast_parent(cpg, parent, node)
                if is_same:
                    cpg.remove_edge(parent, node)
                elif n1_parent:
                    # logger.warning(cpg.nodes[node]['cpg_node'].node_type)
                    # logger.error(n1_parent)
                    combine_expression(cpg, node, n1_parent)
            
def parent_is_branch(cpg: DiGraph, node: str) -> bool:
    """Determine whether block is a basic block.
    """
    start_flag = ['if_statement']
    is_start = False

    node_parents = list(cpg.predecessors(node))
    for parent in node_parents:
        parent_type = cpg.nodes[parent]['cpg_node'].node_type
        if parent_type in start_flag:
            is_start = True
            break
    
    return is_start

def combine_expression(cpg: DiGraph, node: str, n1_parent: str) -> None:
    """Combine special expression or statement.
    """
    node_parents = list(cpg.predecessors(node))
    for parent in node_parents:
        cpg.remove_edge(parent, node)
    
    cpg.add_edge(n1_parent, node, edge_type='100')
    

def create_sum_block(cpg: DiGraph, node: str) -> None:
    """For basic blocks formed by expression or local variable ...
    Create a block to sum them.
    """
    seed_str = cpg.nodes[node]['cpg_node'].node_key
    block_key = uuid.uuid3(uuid.NAMESPACE_DNS, seed_str)
    block_key = str(block_key).replace('-', '')
    # logger.error(f'add {block_key}')

    # determine whether has existed in cpg, if yes, exit
    if cpg.has_node(block_key):
        logger.error('Failed to generate sum block for special statements.')
        exit(-1)
    block_node = CPGNode(ASTNode(block_key, 'block', '', -1, -1))
    block_node.set_block_node()
    block_node.set_statement_node()

    cpg.add_node(block_key, cpg_node=block_node)
    cpg.nodes[node]['cpg_node'].match_block = False
    
    node_parents = list(cpg.predecessors(node))

    for parent in node_parents:
        edge_type = cpg[parent][node]['edge_type']
        if edge_type in ['010', '110']:
            cpg.add_edge(parent, block_key, edge_type='010')
            cpg.remove_edge(parent, node)
        elif edge_type == '100':
            cpg.remove_edge(parent, node)
    
    cpg.add_edge(block_key, node, edge_type='100')

def identify_block(cpg: DiGraph, node: str) -> None:
    """Traverse current code property graph, identify and split basic blocks.
    """
    queue = Queue()
    visited = list()

    queue.push(node)

    while not queue.is_empty():
        current_node = queue.pop()
        current_node_type = cpg.nodes[current_node]['cpg_node'].node_type
        cfg_parent_num = check_parent_num(cpg, current_node)
        if current_node_type in block_start:
            cpg.nodes[current_node]['cpg_node'].set_block_node()
        elif cfg_parent_num > 1:
            create_sum_block(cpg, current_node)
        elif parent_is_branch(cpg, current_node):
            create_sum_block(cpg, current_node)

        update_cfg_edge(cpg, current_node)
        visited.append(current_node)
        current_node_successors = list(cpg.successors(current_node))
        for _successor in current_node_successors:
            edge_type = cpg[current_node][_successor]['edge_type']
            if cpg.nodes[_successor]['cpg_node'].match_statement and _successor not in visited and edge_type in ['110', '010']:
                queue.push(_successor)
    
    unify_cpg(cpg)

def unify_cpg(cpg: DiGraph) -> None:
    """Adjust control flow edge.
    """
    all_edges = list(cpg.edges)
    
    for edge in all_edges:
        start, end = edge
        edge_type = cpg[start][end]['edge_type']
        if edge_type == '100':
            continue
        elif edge_type == '010':
            unify_edge(cpg, start, end)
        else:
            logger.error('Exist wrong edge in FAST.')
            exit(-1)

def unify_edge(cpg: DiGraph, start: str, end: str) -> None:
    """Check nodes of edge and find block parent to unify edge.

    Note: block node should be start or end 's parent. And it is unique.
    """
    def find_ast_parent(cpg: DiGraph, node: str):
        queue = Queue()
        visited = list()
        queue.push(node)

        while not queue.is_empty():
            current_node = queue.pop()
            current_node_parents = list()
            tmp_node_parents = list(cpg.predecessors(current_node))
            for parent in tmp_node_parents:
                if cpg[parent][current_node]['edge_type'] == '100':
                    current_node_parents.append(parent)
            if len(current_node_parents) != 1:
                logger.error(f'Node {current_node} has {len(current_node_parents)} ast parents.')
                exit(-1)
            parent = current_node_parents[0]
            if cpg[parent][current_node]['edge_type'] != '100':
                logger.error('Find ast parent not be 100 edge')
                exit(-1)
            if not cpg.nodes[parent]['cpg_node'].match_block:
                if parent not in visited:
                    queue.push(parent)
                else:
                    logger.error('Find ast parent appear repeated node')
                    exit(-1)
            else:
                return parent

        return None


    start_is_block = cpg.nodes[start]['cpg_node'].match_block
    end_is_block = cpg.nodes[end]['cpg_node'].match_block

    if not start_is_block and not end_is_block:
        start_ast_parent = find_ast_parent(cpg, start)
        end_ast_parent = find_ast_parent(cpg, end)

        if start_ast_parent is None or end_ast_parent is None:
            logger.error('Cannot find ast parent')
            exit(-1)
        cpg.remove_edge(start, end)
        cpg.add_edge(start_ast_parent, end_ast_parent, edge_type='010')
    elif not start_is_block:
        start_ast_parent = find_ast_parent(cpg, start)
        if start_ast_parent is None:
            logger.error('Cannot find ast parent')
            exit(-1)
        cpg.remove_edge(start, end)
        cpg.add_edge(start_ast_parent, end, edge_type='010')
    elif not end_is_block:
        end_ast_parent = find_ast_parent(cpg, end)
        if end_ast_parent is None:
            logger.error('Cannot find ast parent')
            exit(-1)
        cpg.remove_edge(start, end)
        cpg.add_edge(start, end_ast_parent, edge_type='010')