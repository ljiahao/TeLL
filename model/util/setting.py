import argparse
import logging
from colorlog import ColoredFormatter

logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser(prog="driver",
                                    description="learning representation for code blocks")
    # setting for envs
    parser.add_argument('--logging', type=int, default=10,
                        help='Log level [10-50] (default: 10 - Debug)')
    parser.add_argument('--log_file', type=str, default='training_log',
                        help='Log file name (default traing_log)')
    parser.add_argument('--gpu_id', type=str, default='-1',
                        help='GPU device id (default: -1 - CPU)')

    # setting for dataset
    parser.add_argument('--dataset', type=str, default='zookeeper',
                        help='Dir to store code encodings (default: zookeeper)')

     # setting for model
    parser.add_argument('--pretrain', type=int, default=0,
                        help='0: No pretrain, 1: Pretrain with word2vec embeddings, 2:Pretrain with stored models. (default: 0)')
    parser.add_argument('--model_type', type=str, default='hbgn',
                        help='type of learning model from {hbgn} (default: hbgn)')
    parser.add_argument('--adj_type', type=str, default='si',
                        help='type of adjacency matrix from {bi, si}. (default: si)')
                        
    # setting for hbgn model training    
    parser.add_argument('--hbgn_val_size', type=float, default=0.2,
                        help='Size of validation dataset for log level prediction')
    parser.add_argument('--hbgn_test_size', type=float, default=0.2,
                        help='Size of test dataset for log level prediction')
    parser.add_argument('--epoch', type=int, default=100,
                        help='Number of training epoch')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--regs', nargs='?', default='[1e-4,1e-4]',
                        help='Regularization for entity embeddings.')
    parser.add_argument('--opt_type', type=str, default='Adam',
                        help='type of training optimizer from {Adam, SGD, AdaDelta}')
    parser.add_argument('--intra_mess_dropout', nargs='?', default='[0.1,0.1]',
                        help='drop probability for intra-block gnn')
    parser.add_argument('--inter_mess_dropout', nargs='?', default='[0.1, 0.1, 0.1]',
                        help='drop probability for inter-block gnn')
    parser.add_argument('--early_stop', default=True, action='store_false',
                        help='early stop as the performance on validation sets starts to degrade')

    # setting for cwp (inter-block information aggregating)
    parser.add_argument('--inter_block', default=False, action='store_true',
                        help='whether aggregate inter-block information for code blocks (default: False)')

    parser.add_argument('--batch_size_tell', type=int, default=16,
                        help='batch size for log level predicting')
                        
    # setting for Word2vec
    parser.add_argument('--type_dim', type=int, default=32,
                        help='embedding size for AST type/token')
    parser.add_argument('--word2vec_window', type=int, default=10,
                        help='context window size for sequences')
    parser.add_argument('--word2vec_count', type=int, default=1,
                        help='frequency are dropped before training occurs')
    parser.add_argument('--word2vec_worker', type=int, default=8,
                        help='number of workers to train word2vec')
    parser.add_argument('--word2vec_save', default=False, action='store_true',
                        help='whether save word2vec embeddings')

    # setting for GNN
    parser.add_argument('--intra_layer_size', nargs='?', default='[64,32]',
                        help='embedding size for intra-block distilling gnn (changed with mess_dropout)')
    parser.add_argument('--inter_layer_size', nargs='?', default='[64,32,16]',
                        help='embedding size for inter-block aggregating gnn (changed with mess_dropout)')
    parser.add_argument('--agg_type', nargs='?', default='graphsage',
                        help='type of gnn aggregation from {gcn, graphsage}.')

    parser.add_argument('--save_model', default=False, action='store_true',
                        help='whether save HBGN model parameters.')
    
    parser.add_argument('--test', default=False, action='store_true',
                        help='whether test hbgn model or not.')
    parser.add_argument('--validate', default=False, action='store_true',
                        help='whether validate hbgn model or not.')
    
    # log level prediction
    parser.add_argument('--classification_num', type=int, default=5,
                        help='the number of classes of log levels')
    
    parser.add_argument('--max_block_entity', type=int, default=100,
                        help='the max number of entities of code blocks')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='threshold to split different log level')
    
    # log message
    parser.add_argument('--log_dim', type=int, default=64,
                        help='the dimension of log message')
    parser.add_argument('--use_message', type=int, default=2,
                        help='whether use log message or not.')
    parser.add_argument('--max_token_log', type=int, default=25,
                        help='the max number of log tokens in one logging statements')
    
    # fully-connected layer
    parser.add_argument('--fc_size', type=int, default=64,
                        help='the dimension of fully_connected layer')
    parser.add_argument('--one_hot', default=False, action='store_true',
                        help='whether use one_hot encoding schema.')
    parser.add_argument('--select_gpu', type=int, default=0,
                        help='select gpu')

    
    args = parser.parse_args()

    return args
    
def init_logging(level:int, file:str, log_path:str = 'log') -> None:
    formatter = ColoredFormatter(
        "%(white)s%(asctime)10s | %(log_color)s%(levelname)6s | %(log_color)s%(message)6s",
        reset=True,
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'yellow',
            'WARNING':  'green',
            'ERROR':    'red',
            'CRITICAL': 'red,bg_white',
        },
    )

    file_formatter = ColoredFormatter(
        "%(white)s%(asctime)10s | %(log_color)s%(levelname)6s | %(log_color)s%(message)6s",
        reset=True,
        no_color=True,
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    output_file_handler = logging.FileHandler('{0}/{1}.log' .format(log_path, file), mode='w')
    output_file_handler.setFormatter(file_formatter)
    logger.addHandler(handler)
    logger.addHandler(output_file_handler)
    logger.setLevel(level)

def init_setting() -> argparse.Namespace:
    args = parse_args()
    init_logging(args.logging, args.log_file)
    return args
