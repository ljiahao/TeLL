from fast.block_splitter import *
from fast.cpg_api import *
from fast.log_capturer import *
from sast.src_parser import *
from util.common import *
from util.helper import *
from util.setting import *
from util.visualize import *
from util.clac_stat import *

def main():
    args = init_setting()

    if args.load_iresult:
        func_list, func_dict, logged_blocks = load_inter_results(args.iresult_path)
    else:
        func_list, func_dict, logged_blocks = fast_builder(args.src_path)

    logged_blocks = clean_logged_block(logged_blocks)
    if args.store_iresult:
        store_inter_results(args.iresult_path, func_list, func_dict, logged_blocks)
    
    if args.no_stat:
        calc_different_level_logging_blocks(logged_blocks)

if __name__ == "__main__":
    main()
