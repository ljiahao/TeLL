# TeLL (Tell Log Levels)

## Introduction
TeLL is an end-to-end tool to suggest suitable log levels for logging statements in source code by exploiting multi-level code block information (i.e., intra-block and inter-block information) with hierarchical graph neural network.

More specifically, TeLL presents a joint representation *Flow of Abstract Syntax Tree (FAST)* to integrate multi-level block information of code blocks.
Then, we build a *Hierarchical Block Graph Network (HBGN)* upon FAST to model different level information for log level suggestions.

## Environment Requirement
The code has been tested running under Python 3.6.5. The OS is 20.04.2 LTS Ubuntu Linux 64-bit distribution.


## Usage

### Flow of Abstract Syntax Tree (FAST)
FAST is a new joint graph representation integrating intra-block and inter-block information. The nodes in the graph represent basic code blocks (e.g., for block) and edges represent control dependencies (i.e., icfg).

We currently support parsing java source code into Abstract Syntax Tree (AST), Inter-procedure control flow graph (ICFG), and then integrating them into the joint representation FAST.

### How to Use
Make sure you have successfully install the following dependency libraries.
- tree-sitter == 0.19.0
- treelib == 1.6.1
- networkx == 2.6.3
- colorlog == 6.6.0
- nltk == 3.7
- graphviz = 0.20

Enter the directory of `parser`

**Show usage or help**
```bash
python driver.py -h
```
```bash
usage: driver [-h] [--logging LOGGING] [--src_path SRC_PATH]
              [--iresult_path IRESULT_PATH] [--store_iresult] [--load_iresult]
              [--no_stat]

fast constructor for java programs

optional arguments:
  -h, --help            show this help message and exit
  --logging LOGGING     log level [10-50] (default: 10 - Debug)
  --src_path SRC_PATH   directory path of source codes
  --iresult_path IRESULT_PATH
                        directory path of inter results (function list & dict)
  --store_iresult       store inter results or not
  --load_iresult        load inter result or not
  --no_stat             Calculate stat info or not
```

**Transform source code to fast**
```bash
python driver.py --src_path zookeeper
```
```bash
2022-01-29 18:54:35,612 |   INFO | Total File: 604
2022-01-29 18:54:35,640 |   INFO | Parsing file: zookeeper_CreateCommand.java
2022-01-29 18:54:35,759 |   INFO | Parsing file: zookeeper_Util.java
2022-01-29 18:54:35,824 |   INFO | Parsing file: zookeeper_StandaloneTest.java
2022-01-29 18:54:35,961 |   INFO | Parsing file:zookeeper_ZooKeeperTestClient.java
2022-01-29 18:54:36,555 |   INFO | Parsing file: zookeeper_TruncateTest.java
2022-01-29 18:54:36,771 |   INFO | Parsing file: zookeeper_FileTxnLog.java
2022-01-29 18:54:36,857 |   INFO | Parsing file: zookeeper_RemoveWatchesCommand.java
...
2022-01-29 18:55:37,159 |   INFO | Total logged blocks: 1496
2022-01-29 18:55:37,160 |   INFO | trace: 33	debug: 225	info: 602	warn: 372	error: 264
```

### Hierarchical Block Graph Network (HBGN)
HBGN is a two-level graph neural network. The first level gnn traverses the AST topology of code blocks to distill intra-block information. The second level gnn explores the ICFG structure to aggregate neighbor messages for refining the representation of ego code block.

### How to Use
Make sure you have successfully install all the following dependency libraries.
- tensorflow == 1.14.0
- scipy == 1.7.3
- sklearn == 0.20.0
- numpy == 1.21.4
- gensim == 4.1.2

Enter the directory of `model`

**Show usage or help**
```bash
python tf_driver.py -h
```
```bash
usage: driver [-h] [--logging LOGGING] [--log_file LOG_FILE] [--gpu_id GPU_ID]
              [--dataset DATASET] [--pretrain PRETRAIN]
              [--model_type MODEL_TYPE] [--adj_type ADJ_TYPE]
              [--hbgn_val_size HBGN_VAL_SIZE]
              [--hbgn_test_size HBGN_TEST_SIZE] [--epoch EPOCH] [--lr LR]
              [--regs [REGS]] [--opt_type OPT_TYPE]
              [--intra_mess_dropout [INTRA_MESS_DROPOUT]]
              [--inter_mess_dropout [INTER_MESS_DROPOUT]]
              [--inter_block] [--batch_size_tell BATCH_SIZE_TELL]
              [--type_dim TYPE_DIM] [--word2vec_window WORD2VEC_WINDOW]
              [--word2vec_count WORD2VEC_COUNT]
              [--word2vec_worker WORD2VEC_WORKER] [--word2vec_save]
              [--intra_layer_size [INTRA_LAYER_SIZE]]
              [--inter_layer_size [INTER_LAYER_SIZE]] [--agg_type [AGG_TYPE]]
              [--save_model] [--test] [--validate]
              [--classification_num CLASSIFICATION_NUM]
              [--max_block_entity MAX_BLOCK_ENTITY] [--threshold THRESHOLD]
              [--log_dim LOG_DIM] [--use_message USE_MESSAGE]
              [--max_token_log MAX_TOKEN_LOG] [--fc_size FC_SIZE] [--one_hot]
              [--select_gpu SELECT_GPU]

learning representation for code blocks

optional arguments:
  -h, --help            show this help message and exit
  --logging LOGGING     Log level [10-50] (default: 10 - Debug)
  --log_file LOG_FILE   Log file name (default traing_log)
  --gpu_id GPU_ID       GPU device id (default: -1 - CPU)
  --dataset DATASET     Dir to store code encodings (default: zookeeper)
  --pretrain PRETRAIN   0: No pretrain, 1: Pretrain with word2vec embeddings,
                        2:Pretrain with stored models. (default: 0)
  --model_type MODEL_TYPE
                        type of learning model from {hbgn} (default: hbgn)
  --adj_type ADJ_TYPE   type of adjacency matrix from {bi, si}. (default: si)
  --hbgn_val_size HBGN_VAL_SIZE
                        Size of validation dataset for log level prediction
  --hbgn_test_size HBGN_TEST_SIZE
                        Size of test dataset for log level prediction
  --epoch EPOCH         Number of training epoch
  --lr LR               learning rate
  --regs [REGS]         Regularization for entity embeddings.
  --opt_type OPT_TYPE   type of training optimizer from {Adam, SGD, AdaDelta}
  --intra_mess_dropout [INTRA_MESS_DROPOUT]
                        drop probability for intra-block gnn
  --inter_mess_dropout [INTER_MESS_DROPOUT]
                        drop probability for inter-block gnn
...
```
**Train log level suggestion model**
```bash
python tf_driver.py --dataset zookeeper --use_message 0 --validate --test
```
```bash
2022-01-28 20:12:51,914 |   INFO | Loading data from zookeeper
2022-01-28 20:12:52,084 |   INFO | Extracting blocks
2022-01-28 20:12:52,295 |   INFO | Extracting interactions
2022-01-28 20:12:52,369 |  DEBUG | FAST statistics
2022-01-28 20:12:52,370 |  DEBUG | [n_typetoken, n_entity, n_block, n_log_token] = [14775, 13797, 11894, 841]
2022-01-28 20:12:52,370 |  DEBUG | [n_ast, n_cfg] = [25334, 16235]
2022-01-28 20:12:52,370 |  DEBUG | [max n_entity of a block] = [100]
2022-01-28 20:12:52,370 |  DEBUG | [max n_token of a logging statement] = [25]
2022-01-28 20:12:52,371 |  DEBUG | Logged block statistics
2022-01-28 20:12:52,371 |  DEBUG | [n_trace, n_debug, n_info, n_warn, n_error, n_total] = [33, 225, 602, 372, 264, 1496]
2022-01-28 20:12:52,371 |   INFO | Converting interactions (ast & cfg) into sparse adjacency matrix
2022-01-28 20:12:52,378 |   INFO | Generating normalized sparse adjacency matrix
2022-01-28 20:12:52,391 |   INFO | Generating log level prediction training, validation, and testing sets
2022-01-28 20:12:52,406 |  DEBUG | HBGN [n_train, n_val, n_test] = [897, 300, 299]
2022-01-28 20:12:52,496 |   INFO | Initing type/token embeddings with word2vec
2022-01-28 20:12:54,072 |   INFO | Initing logged message embeddings with word2vec
2022-01-28 20:12:54,176 |   INFO | Initing HBGN model
2022-01-28 20:12:54,180 |   INFO | Finish building inputs for HBGN
2022-01-28 20:12:54,583 |   INFO | Init entity embeddings with pre-trained word2vec embeddings
2022-01-28 20:12:54,586 |   INFO | Init message embeddings with pre-trained word2vec embeddings
2022-01-28 20:12:54,665 |   INFO | Finish building weights for HBGN
2022-01-28 20:12:56,238 |   INFO | Finish building loss for log level prediction.
2022-01-28 20:12:56,238 |  DEBUG | Variable name: entity_embedding Shape: 883072
2022-01-28 20:12:56,239 |  DEBUG | Variable name: message_embedding Shape: 53888
2022-01-28 20:12:56,239 |  DEBUG | Variable name: w_intra_0 Shape: 8192
2022-01-28 20:12:56,239 |  DEBUG | Variable name: b_intra_0 Shape: 64
2022-01-28 20:12:56,239 |  DEBUG | Variable name: w_intra_1 Shape: 4096
2022-01-28 20:12:56,240 |  DEBUG | Variable name: b_intra_1 Shape: 32
2022-01-28 20:12:56,240 |  DEBUG | Variable name: w_inter_0 Shape: 20480
2022-01-28 20:12:56,240 |  DEBUG | Variable name: b_inter_0 Shape: 64
2022-01-28 20:12:56,240 |  DEBUG | Variable name: w_inter_1 Shape: 4096
2022-01-28 20:12:56,241 |  DEBUG | Variable name: b_inter_1 Shape: 32
2022-01-28 20:12:56,241 |  DEBUG | Variable name: w_inter_2 Shape: 1024
2022-01-28 20:12:56,241 |  DEBUG | Variable name: b_inter_2 Shape: 16
2022-01-28 20:12:56,241 |  DEBUG | Variable name: fc_w_message Shape: 4096
2022-01-28 20:12:56,242 |  DEBUG | Variable name: fc_b_message Shape: 64
2022-01-28 20:12:56,242 |  DEBUG | Variable name: fc_w_classification Shape: 17408
2022-01-28 20:12:56,242 |  DEBUG | Variable name: fc_b_classification Shape: 64
2022-01-28 20:12:56,242 |  DEBUG | Variable name: fc_w_out Shape: 640
2022-01-28 20:12:56,243 |  DEBUG | Variable name: fc_b_out Shape: 5
2022-01-28 20:12:56,243 |  DEBUG | hbgn_si_graphsage has 997333 parameters
2022-01-28 20:12:56,243 |   INFO | Setup tensorflow session
2022-01-28 20:12:57,289 |   INFO | Training 100 epochs
2022-01-28 20:13:06,910 |  DEBUG | Epoch 1 [9.6s]: train[lr=0.10000]=[(classification: 24.32744)]
2022-01-28 20:13:06,911 |  DEBUG | Validating Results
2022-01-28 20:13:07,685 |   INFO | [trace, trace_t, acc] = [0, 6, 0.000000]
2022-01-28 20:13:07,686 |   INFO | [debug, debug_t, acc] = [8, 45, 0.177778]
2022-01-28 20:13:07,686 |   INFO | [info, info_t, acc] = [111, 121, 0.917355]
2022-01-28 20:13:07,687 |   INFO | [warn, warn_t, acc] = [21, 75, 0.280000]
2022-01-28 20:13:07,687 |   INFO | [error, error_t, acc] = [2, 53, 0.037736]
2022-01-28 20:13:07,688 |   INFO | [correct_t, t, acc] = [142, 300, 0.473333]
2022-01-28 20:13:07,691 |   INFO | [auc = 0.875287]
2022-01-28 20:13:07,696 |   INFO | [macro_f1 = 0.688318]
2022-01-28 20:13:07,697 |   INFO | [aod = 0.799722]
2022-01-28 20:13:07,697 |  DEBUG | Testing Results
2022-01-28 20:13:08,284 |   INFO | [trace, trace_t, acc] = [0, 7, 0.000000]
2022-01-28 20:13:08,284 |   INFO | [debug, debug_t, acc] = [15, 45, 0.333333]
2022-01-28 20:13:08,285 |   INFO | [info, info_t, acc] = [113, 120, 0.941667]
2022-01-28 20:13:08,285 |   INFO | [warn, warn_t, acc] = [20, 74, 0.270270]
2022-01-28 20:13:08,285 |   INFO | [error, error_t, acc] = [2, 53, 0.037736]
2022-01-28 20:13:08,285 |   INFO | [correct_t, t, acc] = [150, 299, 0.501672]
2022-01-28 20:13:08,287 |   INFO | [auc = 0.883774]
2022-01-28 20:13:08,292 |   INFO | [macro_f1 = 0.687810]
2022-01-28 20:13:08,293 |   INFO | [aod = 0.812430]
2022-01-28 20:13:12,633 |  DEBUG | Epoch 2 [4.3s]: train[lr=0.10000]=[(classification: 10.86318)]
2022-01-28 20:13:12,634 |  DEBUG | Validating Results
2022-01-28 20:13:13,273 |   INFO | [trace, trace_t, acc] = [0, 6, 0.000000]
2022-01-28 20:13:13,274 |   INFO | [debug, debug_t, acc] = [34, 45, 0.755556]
2022-01-28 20:13:13,274 |   INFO | [info, info_t, acc] = [95, 121, 0.785124]
2022-01-28 20:13:13,275 |   INFO | [warn, warn_t, acc] = [47, 75, 0.626667]
2022-01-28 20:13:13,275 |   INFO | [error, error_t, acc] = [14, 53, 0.264151]
2022-01-28 20:13:13,276 |   INFO | [correct_t, t, acc] = [190, 300, 0.633333]
2022-01-28 20:13:13,280 |   INFO | [auc = 0.890453]
2022-01-28 20:13:13,285 |   INFO | [macro_f1 = 0.793159]
2022-01-28 20:13:13,286 |   INFO | [aod = 0.814722]
2022-01-28 20:13:13,286 |  DEBUG | Testing Results
2022-01-28 20:13:13,876 |   INFO | [trace, trace_t, acc] = [1, 7, 0.142857]
2022-01-28 20:13:13,877 |   INFO | [debug, debug_t, acc] = [31, 45, 0.688889]
2022-01-28 20:13:13,878 |   INFO | [info, info_t, acc] = [90, 120, 0.750000]
2022-01-28 20:13:13,878 |   INFO | [warn, warn_t, acc] = [46, 74, 0.621622]
2022-01-28 20:13:13,878 |   INFO | [error, error_t, acc] = [18, 53, 0.339623]
2022-01-28 20:13:13,878 |   INFO | [correct_t, t, acc] = [186, 299, 0.622074]
2022-01-28 20:13:13,881 |   INFO | [auc = 0.896185]
2022-01-28 20:13:13,885 |   INFO | [macro_f1 = 0.817159]
2022-01-28 20:13:13,886 |   INFO | [aod = 0.817447]
...
2022-01-28 20:16:47,051 |  DEBUG | Testing Results
2022-01-28 20:16:47,647 |   INFO | [trace, trace_t, acc] = [5, 7, 0.714286]
2022-01-28 20:16:47,648 |   INFO | [debug, debug_t, acc] = [32, 45, 0.711111]
2022-01-28 20:16:47,648 |   INFO | [info, info_t, acc] = [103, 120, 0.858333]
2022-01-28 20:16:47,649 |   INFO | [warn, warn_t, acc] = [51, 74, 0.689189]
2022-01-28 20:16:47,650 |   INFO | [error, error_t, acc] = [33, 53, 0.622642]
2022-01-28 20:16:47,650 |   INFO | [correct_t, t, acc] = [224, 299, 0.749164]
2022-01-28 20:16:47,652 |   INFO | [auc = 0.931878]
2022-01-28 20:16:47,656 |   INFO | [macro_f1 = 0.894223]
2022-01-28 20:16:47,657 |   INFO | [aod = 0.888796]
```

## Citation
If you find this work useful for your research, please consider citing our paper:

```
@inproceedings{liu2022tell,
  title={TeLL: Log Level Suggestions via Modeling Multi-level Code Block Information},
  author={Liu, Jiahao and Zeng, Jun and Wang, Xiang and Ji, Kaihang and Liang, Zhenkai},
  booktitle={Proceedings of the 31st ACM SIGSOFT International Symposium on Software Testing and Analysis},
  pages = {27-38},
  year={2022}
}
```