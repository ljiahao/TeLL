import os
import warnings
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
warnings.simplefilter('ignore', category=FutureWarning)
warnings.simplefilter('ignore', category=RuntimeWarning)
from time import time
import numpy as np
import tensorflow as tf
import random as rd
from model_tf.hbgn import HBGN
from model_tf.load_gnn import GNNLoader
from model_tf.eval import validation_ordinal, test_ordinal
from util.setting import init_setting, logger
from util.Word2vec import Log2vec, Type2vec

def main():
    """ Get argument settings """
    seed = 2022
    np.random.seed(seed)
    tf.set_random_seed(seed)
    rd.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.reset_default_graph()

    """ Initialize args and dataset """
    args = init_setting()
    logger.info("Loading data from %s" % args.dataset)
    data_generator = GNNLoader(args)

    """ Define GPU/CPU device to train model """
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    """ Use pre-trained word2vec embeddings to initialize AST nodes (entities) """
    type2vec = Type2vec(args)
    if args.pretrain == 0 or args.pretrain == 2:
        type2vec.init_embedding(data_generator.e2t_list, data_generator.typetoken_seq)
        """ Save pre-trained word2vec embeddings"""
        if args.word2vec_save:
            type2vec.store_embedding(data_generator.out_path)
    elif args.pretrain == 1:
        type2vec.load_embedding(data_generator.out_path)
    
    """ Use log message 0 & 1 means use log message, 2 means do not use"""
    log2vec = Log2vec(args)
    if args.use_message == 0:
        log2vec.init_embedding(data_generator.log_tokens, data_generator.logtoken_seq)
        """ Save pre-trained log word2vec embeddings"""
        if args.word2vec_save:
            log2vec.store_embedding(data_generator.out_path
            )
    elif args.use_message == 1:
        log2vec.load_embedding(data_generator.out_path)
    
    """ Select learning models """
    if args.model_type == 'hbgn':
        logger.info("Initing HBGN model")
        model = HBGN(args, data_generator, pretrain_embedding=type2vec.embedding, message_embedding=log2vec.embedding)
    else:
        logger.error("The ML model is unknown")
        exit(-1)

    """ Setup tensorflow session """
    logger.info("Setup tensorflow session")
    sess = model.setup_sess()
    
    """ Reload model parameters for fine tune """
    if args.pretrain == 2:
        model.load_model(sess, data_generator.out_path)

    """ Training phase """
    logger.info("Training %d epochs" % args.epoch)
    best_res = [0.] * 9 
    start = time()
    for epoch in range(args.epoch):
        data_generator.shuffle_train_data()
        if epoch % 20 == 0 and epoch != 0:
            model.lr = model.lr * 0.8
        classification_loss = 0.
        t_train = time()
        
        """ log level prediction """
        for i_batch in range(data_generator.hbgn_data_iter):
            batch_data = data_generator.generate_hbgn_train_batch(i_batch)
            feed_dict = data_generator.generate_hbgn_train_feed_dict(model, batch_data)
            _, classification_loss_batch = model.train_classification(sess, feed_dict)
            classification_loss += classification_loss_batch
        perf_train_ite = 'Epoch %d [%.1fs]: train[lr=%.5f]=[(classification: %.5f)]' % (epoch + 1, time() - t_train, model.lr, classification_loss)

        logger.debug(perf_train_ite)
        logger.debug('Validating Results')
        if epoch % 1 == 0 and args.validate:
            validation_ordinal(sess, model, data_generator,args.threshold, best_res)

        logger.debug('Testing Results')
        if epoch % 1 == 0 and args.test:
            test_ordinal(sess, model, data_generator, args.threshold)

    logger.warning(f'Training time: {time() - start}')
    """ Save the model parameters """
    if args.save_model:
        model.store_model(sess, data_generator.out_path, epoch)

if __name__ == '__main__':
    main()
