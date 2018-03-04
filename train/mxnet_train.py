#!/usr/bin/env python
# -*- coding: utf-8 -*-
# created 2017/05/05 @Northrend
#
# Training script
# for mxnet image classification
#

from __future__ import print_function
import sys
import os
import time
import math
import re
import logging
from logging import config
import docopt
import mxnet as mx
from importlib import import_module

# init global logger
log_format = '%(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
# logger = logging.getLogger(__name__)
logger = logging.getLogger()  # sync with mxnet internal logger
# logging.config.dictConfig({'version': 1.0, 'disable_existing_loggers': False})
# shandler = logging.StreamHandler()  # log to stdout
fhandler = None     # log to file


def _init_():
    '''
    Training script for image-classification task on mxnet
    Update: 2018/02/03
    Author: @Northrend
    Contributor:

    Changelog:
    2018/02/03  v2.8        support random resize scale
    2018/01/29  v2.7        fix resume training job bugs
    2017/12/27  v2.6        support se-inception-v4
    2017/12/04  v2.5        support change shorter edge size
    2017/11/21  v2.4        support input nomalization
    2017/09/22  v2.3        support resume training job
    2017/09/04  v2.2        support modify cardinality for resnext
    2017/08/15  v2.1        support modify dropout argument
    2017/08/09  v2.0        apply logging module
    2017/08/08  v1.5        support scratch-training and more networks
    2017/07/24  v1.4        support customized input shape
    2017/07/13	v1.3	    fix lr descend bug
    2017/05/19  v1.2        support multi-metrics during training
                            support threshold and save-json options
                            modify script name from mxnet_train_finetune to mxnet_train
    2017/05/10  v1.1        separate _init_ as a function
    2017/05/08  v1.0        finetune task tested
                            resume training mode unsupported

    Usage:
        mxnet_train.py      <log> [-f|--finetune] [-r|--resume] [-t|--test-io] [-s|--save-json]
                            [--log-lv=str --log-mode=str --data-train=str --data-dev=str]
                            [--model-prefix=str --num-epochs=int --threshold=flt --gpus=lst]
                            [--kv-store=str --network=str --num-layers=int --pretrained-model=str]
                            [--load-epoch=int --num-classes=int  --num-samples=int --img-width=int]
                            [--resize=int --resize-scale=lst]
                            [--batch-size=int --optimizer=str --lr=flt --lr-factor=flt --momentum=flt]
                            [--weight-decay=flt --lr-step-epochs=lst --disp-batches=int --disp-lr]
                            [--top-k=int --metrics=lst --dropout=flt --num-groups=int --mean=lst --std=lst]
        mxnet_train.py      -v | --version
        mxnet_train.py      -h | --help

    Arguments:
        <log>                       file to save log

    Options:
        -h --help                   show this help screen
        -v --version                show current version
        -f --finetune               set to start a finetune training job
        -t --test-io                test reading speed mode, without training
        -r --resume                 set to resume training job
        -s --save-json              whether save ./finetuned-symbol.json or not
        ------------------------------------------------------------------------------------------------------------
        --log-lv=str                logging level, one of INFO DEBUG WARNING ERROR CRITICAL [default: INFO]
        --log-mode=str              log file mode [default: w]
        --data-train=str            training data path [default: data/cifar10_train.rec]
        --data-dev=str              developping data path [default: data/cifar10_val.rec]
        --model-prefix=str          prefix of model to save
        --num-epochs=int            number of epochs [default: 8]
        --threshold=flt             threshold of training result warning, no warning by default
        --gpus=lst                  list of gpus to run, e.g. 0 or 0,2,5 (mxnet will use cpu by default)
        --kv-store=str              key-value store type [default: device]
        --network=str               net architecture [default: resnet]
        --num-layers=int            number of layers in the neural network, required by some networks such as resnet [default: 0]
        --pretrained-model=str      pre-trained model path [default: model/resnet-50]
        --load-epoch=int            load the model on an epoch using the model-load-prefix [default: 0]
        --num-classes=int           number of classes [default: 10]
        --num-samples=int           number of samples in training data [default: 50000]
        --num-groups=int            value of cardinality for resnext [default: 32]
        --img-width=int             input image size, keep width=height [default: 28]
        --resize=int                set to resize shorter edge if needed [default: -1]
        --resize-scale=lst          set to randomly resize shorter edge in this scale range [default: 1,1]
        --batch-size=int            the batch size on each gpu [default: 128]
        --dropout=flt               set dropout probability if needed [default: 0]
        --optimizer=str             optimizer type [default: sgd]
        --lr=flt                    initial learning rate [default: 0.01]
        --lr-factor=flt             the ratio to reduce lr on each step [default: 0.1]
        --lr-step-epochs=lst        the epochs to reduce the lr, e.g. 30,60 [default: 20,40,60,80]
        --momentum=flt              momentum for sgd [default: 0.9]
        --weight-decay=flt          weight decay for sgd [default: 0.0005]
        --disp-batches=int          show progress for every n batches [default: 20]
        --disp-lr                   whether to show and log learning-rates after each batch
        --top-k=int                 report the top-k accuracy, 0 means no report [default: 0]
        --mean=lst                  list of rgb mean value [default: 123.68,116.779,103.939]
        --std=lst                   list of rgb std value [default: 58.395,57.12,57.375]
        --metrics=lst               metric of logging, set list of metrics to log several metrics [default: accuracy]
    '''
    # config logger
    # logging.basicConfig(filename=args['<log>'],level=eval('logging.{}'.format(args['--log-lv'])), format=log_format)
    logger.setLevel(eval('logging.{}'.format(args['--log-lv'])))
    fhandler = logging.FileHandler(args['<log>'], mode=str(args['--log-mode']))
    # logger.propagate = False    # avoid repeating logs
    logger.addHandler(fhandler)
    # logger.addHandler(shandler)

    # print arguments
    logger.info('-' * 80 + '\nArguments submitted:')
    for key in sorted(args.keys()):
        logger.info('{:<20}= {}'.format(key.replace('--', ''), args[key]))
    logger.info('-' * 80)

    # reset logger format
    # shandler.setFormatter(logging.Formatter(log_format))
    fhandler.setFormatter(logging.Formatter(log_format))


def _get_lr_scheduler(args, kv, num_gpus):
    if '--lr-factor' not in args or float(args['--lr-factor']) >= 1:
        return (args['--lr'], None)
    epoch_size = int(
        math.ceil(int(args['--num-samples']) / int(args['--batch-size'])))
    if 'dist' in args['--kv-store']:
        epoch_size /= kv.num_workers
    epoch_size /= num_gpus
    begin_epoch = int(args['--load-epoch']) if args['--load-epoch'] else 0
    step_epochs = [int(l) for l in args['--lr-step-epochs'].split(',')]
    lr = float(args['--lr'])
    for s in step_epochs:
        if begin_epoch >= s:
            lr *= float(args['--lr-factor'])
    if lr != float(args['--lr']):
        logging.info('Adjust learning rate to %e for epoch %d' %
                     (lr, begin_epoch))
    steps = [epoch_size * (x - begin_epoch)
             for x in step_epochs if x - begin_epoch > 0]
    return (lr, mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=float(args['--lr-factor'])))


def _load_model(args, rank=0):
    if '--load-epoch' not in args or args['--load-epoch'] is None:
        return (None, None, None)
    assert args['--model-prefix'] is not None, logger.error(
        'Missing model prefix!')
    model_prefix = args['--model-prefix']
    if rank > 0 and os.path.exists("%s-%d-symbol.json" % (model_prefix, rank)):
        model_prefix += "-%d" % (rank)
    sym, arg_params, aux_params = mx.model.load_checkpoint(
        model_prefix, args['--load-epoch'])
    logging.info('Loaded model %s_%04d.params',
                 model_prefix, args['--load-epoch'])
    return (sym, arg_params, aux_params)


def _save_model(model_prefix, rank=0):
    if model_prefix is None:
        return None
    dst_dir = os.path.dirname(model_prefix)
    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)
    return mx.callback.do_checkpoint(model_prefix if rank == 0 else "%s-%d" % (
        model_prefix, rank))


def _get_iterators(data_train, data_dev, batch_size, data_shape=(3, 224, 224), resize=-1):
    '''
    define the function which returns the data iterators
    '''
    [mean_r, mean_g, mean_b] = [float(x) for x in args['--mean'].split(',')]
    [std_r, std_g, std_b] = [float(x) for x in args['--std'].split(',')]
    [max_random_scale, min_random_scale] = [float(x) for x in args['--resize-scale'].split(',')]
    logger.info('Input normalization params: mean_rgb {}, std_rgb {}'.format([mean_r, mean_g, mean_b],[std_r, std_g, std_b]))
    train = mx.io.ImageRecordIter(
        path_imgrec=data_train,
	# preprocess_threads=32,
        data_name='data',
        label_name='softmax_label',
        batch_size=batch_size,
        data_shape=data_shape,
        resize=resize,
        max_random_scale=max_random_scale,
        min_random_scale=min_random_scale,
        shuffle=True,
        rand_crop=True,
        rand_mirror=True,
        mean_r=mean_r,
        mean_g=mean_g,
        mean_b=mean_b,
        std_r=std_r,
        std_g=std_g,
        std_b=std_b
        )
    val = mx.io.ImageRecordIter(
        path_imgrec=data_dev,
	# preprocess_threads=32,
        data_name='data',
        label_name='softmax_label',
        batch_size=batch_size,
        data_shape=data_shape,
        # resize=resize,
        shuffle=False,
        rand_crop=False,
        rand_mirror=False,
        mean_r=mean_r,
        mean_g=mean_g,
        mean_b=mean_b,
        std_r=std_r,
        std_g=std_g,
        std_b=std_b
        )
    logger.info("Data iters created successfully")
    return (train, val)


def _get_eval_metrics(lst_metrics):
    '''
    return multiple evaluation metrics
    '''
    all_metrics = ['accuracy', 'ce', 'f1',
                   'mae', 'mse', 'rmse', 'top_k_accuracy']
    lst_child_metrics = []
    eval_metrics = mx.metric.CompositeEvalMetric()
    for metric in lst_metrics:
        assert metric in all_metrics, logger.error(
            'Invalid evaluation metric!')
        if metric == 'accuracy':
            lst_child_metrics.append(mx.metric.Accuracy())
        elif metric == 'ce':
            lst_child_metrics.append(mx.metric.CrossEntropy())
        elif metric == 'f1':
            lst_child_metrics.append(mx.metric.F1())
        elif metric == 'mae':
            lst_child_metrics.append(mx.metric.MAE())
        elif metric == 'mse':
            lst_child_metrics.append(mx.metric.MSE())
        elif metric == 'rmse':
            lst_child_metrics.append(mx.metric.RMSE())
        elif metric == 'top_k_accuracy':
            lst_child_metrics.append(
                mx.metric.TopKAccuracy(top_k=int(args['--top-k'])))
    for child_metric in lst_child_metrics:
        eval_metrics.add(child_metric)
    return eval_metrics


def _get_batch_end_callback(batch_size, display_batch=40):
    '''
    get callback function after each batch
    '''
    def _get_cbs_metrics(cb_eval_metrics):
        logger.debug(type(cb_eval_metrics))
        metric, value = cb_eval_metrics.get()
        str_metric = str()
        for index, met in enumerate(metric):
            str_metric += '\t{}={}'.format(met, value[index])
        return str_metric

    def _display_metrics(BatchEndParam):
        '''
        call back eval metrics info
        '''
        if BatchEndParam.nbatch % display_batch == 0:
            logger.info("Epoch[{}] Batch [{}]\t{}".format(
                BatchEndParam.epoch, BatchEndParam.nbatch,  _get_cbs_metrics(BatchEndParam.eval_metric)))

    def _display_lr(BatchEndParam):
        '''
        call back learning rate info
        '''
        if BatchEndParam.nbatch != 0 and BatchEndParam.nbatch % display_batch == 0:
            logger.info("Epoch[{}] Batch [{}]\tlearning-rate={}".format(
                BatchEndParam.epoch, BatchEndParam.nbatch, BatchEndParam.locals["self"]._optimizer._get_lr(0)))

    cbs = list()
    if args['--disp-lr']:
        cbs.append(_display_lr)
    cbs.append(mx.callback.Speedometer(batch_size, display_batch))
    return cbs


def _get_fine_tune_model(symbol, arg_params, num_classes, layer_name='flatten0'):
    '''
    define the function which replaces the the last fully-connected layer for a given network
    symbol: the pre-trained network symbol
    arg_params: the argument parameters of the pre-trained model
    num_classes: the number of classes for the fine-tune datasets
    layer_name: the layer name before the last fully-connected layer
    '''
    all_layers = symbol.get_internals()
    net = all_layers[layer_name + '_output']
    net = mx.symbol.FullyConnected(
        data=net, num_hidden=num_classes, name='fc-' + str(num_classes))
    net = mx.symbol.SoftmaxOutput(data=net, name='softmax')
    new_args = dict({k: arg_params[k] for k in arg_params if 'fc1' not in k})
    if args['--save-json']:
        net.save('./finetuned-symbol.json')
        logger.info('Saved to ./finetuned-symbol.json')
    return (net, new_args)


def _whatever_the_fucking_fit_is(symbol, arg_params, aux_params, train, val, batch_size, args):
    '''
    training function
    '''
    # official argument kv-store
    kv = mx.kvstore.create(args['--kv-store'])

    # save model
    checkpoint = _save_model(args['--model-prefix'], kv.rank)

    # set devices
    devices = mx.cpu() if args['--gpus'] is None else [mx.gpu(int(i))
                                                       for i in args['--gpus'].split(',')]

    # update learning rate
    num_gpus = len(args['--gpus'].split(',')
                   ) if args['--gpus'] is not None else 1
    lr, lr_scheduler = _get_lr_scheduler(args, kv, num_gpus)

    # optimizer params of default optimizer-'sgd'
    lr_scheduler = lr_scheduler
    optimizer_params = {
        'learning_rate': lr,
        'momentum': float(args['--momentum']),
        'wd': float(args['--weight-decay']),
        'lr_scheduler': lr_scheduler}

    mod = mx.mod.Module(symbol=symbol, context=devices)
    mod.bind(data_shapes=train.provide_data, label_shapes=train.provide_label)
    # initialize weights
    if args['--network'] == 'alexnet':
        mod.init_params(initializer=mx.init.Normal())
    else:
        mod.init_params(initializer=mx.init.Xavier(
            rnd_type='gaussian', factor_type="in", magnitude=2))
    if args['--finetune']:
        # replace all parameters except for the last fully-connected layer with pre-trained model
        mod.set_params(arg_params, aux_params, allow_missing=True)
    elif args['--resume']:
        # set weights
        mod.set_params(arg_params, aux_params, allow_missing=False, allow_extra=False)

    # set metrics during training
    # if len(args['--metrics'].split(',')) == 1:
    #     metrics = args['--metrics']
    # elif len(args['--metrics'].split(',')) > 1:
    #     metrics = _get_eval_metrics(args['--metrics'].split(','))
    metrics = args['--metrics'] if len(args['--metrics'].split(
        ',')) == 1 else _get_eval_metrics(args['--metrics'].split(','))

    batch_end_callbacks = _get_batch_end_callback(
        batch_size, int(args['--disp-batches']))
    # batch_end_callbacks = mx.callback.Speedometer(batch_size, int(args['--disp-batches']))

    mod.fit(train, val,
            eval_metric=metrics,
            begin_epoch=int(args['--load-epoch']) if args['--resume'] else 0,
            num_epoch=int(args['--num-epochs']),
            kvstore=kv,
            optimizer=args['--optimizer'],
            optimizer_params=optimizer_params,
            batch_end_callback=batch_end_callbacks,
            epoch_end_callback=checkpoint,
            allow_missing=True)
    return mod.score(val, metrics)


def main():
    logger.debug('main function entry checkpoint')
    logger.info('MXNet version: {}'.format(mx.__version__))
    # prepare arguments
    num_classes = int(args['--num-classes'])
    batch_per_gpu = int(args['--batch-size'])
    num_gpus = len(args['--gpus'].split(',')
                   ) if args['--gpus'] is not None else 1
    batch_size = batch_per_gpu * num_gpus
    (train, val) = _get_iterators(args['--data-train'], args['--data-dev'],
                                  batch_size, data_shape=(3, int(args['--img-width']), int(args['--img-width'])), resize=int(args['--resize']))

    # io testing mode
    if args['--test-io']:
        tic = time.time()
        for i, batch in enumerate(train):
            for j in batch.data:
                j.wait_to_read()
            if (i + 1) % int(args['--disp-batches']) == 0:
                logging.info('Batch [%d]\tSpeed: %.2f samples/sec' % (i + 1, float(
                    args['--disp-batches']) * int(args['--batch-size']) / (time.time() - tic)))
                tic = time.time()
        return

    if args['--finetune']:
        # load pre-trained model
        sym, arg_params, aux_params = mx.model.load_checkpoint(
            args['--pretrained-model'], int(args['--load-epoch']))
        logger.info('pre-trained model loaded successfully, start finetune job...')
        # adapt original network to finetune network
        (new_sym, new_args) = _get_fine_tune_model(sym, arg_params, num_classes)
    elif args['--resume']:
        # load model
        new_sym, new_args, aux_params = mx.model.load_checkpoint(
            args['--pretrained-model'], int(args['--load-epoch']))
        logger.info('model loaded successfully, resume training job...')
    else:
        # initialize network symbols only
        network = import_module('symbols.' + args['--network'])
        new_sym = network.get_symbol(int(args['--num-classes']), int(
            args['--num-layers']), '3,{0},{0}'.format(args['--img-width']), **args)
        new_args = None
        aux_params = None

    mod_score = _whatever_the_fucking_fit_is(
        new_sym, new_args, aux_params, train, val, batch_size, args)

    if args['--threshold']:
        if mod_score >= float(args['--threshold']):
            logger.warn(
                "Training acc seems to be quite low, maybe you have messed up :D")
    return 0


if __name__ == '__main__':
    version = re.compile('.*\d+/\d+\s+(v[\d.]+)').findall(_init_.__doc__)[0]
    args = docopt.docopt(
        _init_.__doc__, version='mxnet training script {}'.format(version))
    _init_()
    logger.info('Start training job...')
    main()
    logger.info('...Done')
