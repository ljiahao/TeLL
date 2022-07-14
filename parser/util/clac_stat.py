from util.setting import logger

def calc_different_level_logging_statements(logged_blocks: list) -> None:
    """Calculate the number of different level statements.
    """
    level_dict = dict()

    for block in logged_blocks:
        level = block.level
        if level not in level_dict:
            level_dict[level] = 1
        else:
            level_dict[level] += 1

    trace = level_dict['trace']
    debug = level_dict['debug']
    info =  level_dict['info']
    warn = level_dict['warn']
    error = level_dict['error']
    if 'fatal' in level_dict:
        fatal = level_dict['fatal']
    else:
        fatal = 0

    logger.info(f'Total logging stataments: {len(logged_blocks)}')
    logger.info(f'trace: {trace}\tdebug: {debug}\tinfo: {info}\twarn: {warn}\terror: {error}\tfatal: {fatal}')

def calc_different_level_logging_blocks(logged_blocks: list) -> None:
    """Calculate the number of logged block with different levels.
    """
    def select_level(levels: list) -> str:
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
        else:
            logger.info('Level is empty')
            exit(-1)

    block_dict = dict()


    for block in logged_blocks:
        if block.block_id not in block_dict:
            block_dict[block.block_id] = [block.level]
        else:
            block_dict[block.block_id].append(block.level)
    
    logger.info(f'Total logged blocks: {len(block_dict)}')

    trace, debug, info, warn, error, fatal = 0, 0, 0, 0, 0, 0
    mixed = 0

    for _, value in block_dict.items():
        levels = list(set(value))
        if len(levels) > 1:
            mixed += 1
        else:
            level = select_level(levels)
            if level == 'trace':
                trace += 1
            elif level == 'debug':
                debug += 1
            elif level == 'info':
                info += 1
            elif level == 'warn':
                warn += 1
            elif level == 'error':
                error += 1
            elif level == 'fatal':
                fatal += 1
        
    logger.info(f'trace: {trace}\tdebug: {debug}\tinfo: {info}\twarn: {warn}\terror: {error}')