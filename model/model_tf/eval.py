import numpy as np
import tensorflow as tf
from sklearn import metrics
from util.setting import logger
from model_tf.load_gnn import GNNLoader
from model_tf.hbgn import HBGN

def filter_rel(line, threshold) -> list:
    res = [1., 0., 0., 0., 0.]
    for x in range(1, len(res)):
        if line[x] >= threshold:
            res[x] = 1.
        else:
            break
    
    return res

def query_level(level: float) -> str:
    if level == 1.:
        return 'trace'
    elif level == 2.:
        return 'debug'
    elif level == 3.:
        return 'info'
    elif level == 4.:
        return 'warn'
    elif level == 5.:
        return 'error'
    else:
        logger.error('Unknown level')
        exit(-1)

def accuracy(classification_pred, classification_label) -> list:
    log_correct = {'trace': 0, 'debug': 0, 'info': 0, 'warn': 0, 'error': 0, 'total': 0}
    log_total = {'trace': 0, 'debug': 0, 'info': 0, 'warn': 0, 'error': 0, 'total': 0}

    length = len(classification_pred)
    for idx in range(length):
        pred_sum = sum(classification_pred[idx])
        label_sum = sum(classification_label[idx])
        level = query_level(label_sum)
        if pred_sum == label_sum:
            log_correct[level] += 1
            log_correct['total'] += 1
        log_total[level] += 1
        log_total['total'] += 1

    trace_acc = log_correct['trace'] / log_total['trace']
    debug_acc = log_correct['debug'] / log_total['debug']
    info_acc = log_correct['info'] / log_total['info']
    warn_acc = log_correct['warn'] / log_total['warn']
    error_acc = log_correct['error'] / log_total['error']
    correct_acc = log_correct['total'] / log_total['total']

    logger.info('[trace, trace_t, acc] = [%d, %d, %f]' %(log_correct['trace'], log_total['trace'], trace_acc))
    logger.info('[debug, debug_t, acc] = [%d, %d, %f]' %(log_correct['debug'], log_total['debug'], debug_acc))
    logger.info('[info, info_t, acc] = [%d, %d, %f]' %(log_correct['info'], log_total['info'], info_acc))
    logger.info('[warn, warn_t, acc] = [%d, %d, %f]' %(log_correct['warn'], log_total['warn'], warn_acc))
    logger.info('[error, error_t, acc] = [%d, %d, %f]' %(log_correct['error'], log_total['error'], error_acc))
    logger.info('[correct_t, t, acc] = [%d, %d, %f]' %(log_correct['total'], log_total['total'], correct_acc))

    return [correct_acc, trace_acc, debug_acc, info_acc, warn_acc, error_acc]

def auc(classification_pred, classification_label) -> float:
    """Calculate AUC """
    auc = metrics.roc_auc_score(classification_label, classification_pred, average='micro', multi_class='ovo')

    logger.info('[auc = %f]' % auc)

    return auc

def macro_f1(classification_pred, classification_label) -> float:
    """Calculate Macro F1"""
    f1 = metrics.f1_score(classification_label, classification_pred, average='macro')

    logger.info('[macro_f1 = %f]' %f1)

    return f1

def aod(classification_pred, classification_label) -> float:
    """Calculate AOD (average ordinal distance)"""
    max_distance = {'trace':4., 'debug':3., 'info':2., 'warn':3., 'error':4.}

    distance_sum = 0.
    length = len(classification_pred)
    
    for idx in range(length):
        pred_sum = sum(classification_pred[idx])
        label_sum = sum(classification_label[idx])
        level = query_level(label_sum)
        _distance = abs(label_sum - pred_sum)
        distance_sum = distance_sum + (1 - _distance / max_distance[level])
    
    aod = distance_sum / length

    logger.info('[aod = %f]' % aod)
    
    return aod

def validation_ordinal(sess:tf.Session, model:HBGN, data_generator:GNNLoader, threshold: list, best_res: list) -> None:
    """"""
    n_batch_hbgn_val = data_generator.n_hbgn_val // data_generator.batch_size_tell
    if n_batch_hbgn_val == 0:
        n_batch_hbgn_val = 1
    elif data_generator.n_hbgn_val % data_generator.batch_size_tell:
        n_batch_hbgn_val += 1
    
    classification_rel = []
    classification_label = []

    for i_batch in range(n_batch_hbgn_val):
        batch_data = data_generator.generate_hbgn_val_batch(i_batch, (i_batch == n_batch_hbgn_val - 1))
        feed_dict = data_generator.generate_hbgn_val_feed_dict(model, batch_data)
        
        classification_rel.extend(model.eval_classification(sess, feed_dict))
        classification_label.extend(batch_data['y_classification'])
        classification_pred = np.apply_along_axis(filter_rel, 1, classification_rel, threshold).tolist()
    
    acc_res = accuracy(classification_pred, classification_label)
    auc_res = auc(classification_pred, classification_label)
    macro_res = macro_f1(classification_pred, classification_label)
    aod_res = aod(classification_pred, classification_label)


def test_ordinal(sess:tf.Session, model:HBGN, data_generator:GNNLoader, threshold: float) -> None:
    """"""
    n_batch_hbgn_test = data_generator.n_hbgn_test // data_generator.batch_size_tell
    if n_batch_hbgn_test == 0:
        n_batch_hbgn_test = 1
    elif data_generator.n_hbgn_test % data_generator.batch_size_tell:
        n_batch_hbgn_test += 1
    
    classification_rel = []
    classification_label = []

    for i_batch in range(n_batch_hbgn_test):
        batch_data = data_generator.generate_hbgn_test_batch(i_batch, (i_batch == n_batch_hbgn_test - 1))
        feed_dict = data_generator.generate_hbgn_test_feed_dict(model, batch_data)
        
        classification_rel.extend(model.eval_classification(sess, feed_dict))
        classification_label.extend(batch_data['y_classification'])

    classification_pred = np.apply_along_axis(filter_rel, 1, classification_rel, threshold=threshold).tolist()
    accuracy(classification_pred, classification_label)
    auc(classification_pred, classification_label)
    macro_f1(classification_pred, classification_label)
    aod(classification_pred, classification_label)