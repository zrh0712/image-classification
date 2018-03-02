# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
import mxnet as mx
import pprint


def modify_net(sym, arg_params):
    '''
    write your network-modify codes here
    '''
    input_shape = (1, 3, 352, 352)   # input image size, (batch, channel, height, width)

    # modify symbol
    s4_u3_b3 = sym.get_internals()['stage4_unit3_bn3_output']
    s4_u3_a3 = mx.sym.Activation(data=s4_u3_b3, act_type='relu', name='stage4_unit3_relu')
    s5_u1_c1 = mx.sym.Convolution(data=s4_u3_a3, num_filter=4096, kernel=(3, 3), stride=(
        2, 2), pad=(1, 1), no_bias=True, workspace=512, name='stage5_unit1_conv1')
    s5_u1_b1 = mx.sym.BatchNorm(data=s5_u1_c1, fix_gamma=False, eps=2e-5, momentum=0.9, name='stage5_unit1_bn1')
    s5_u1_a1 = mx.sym.Activation(data=s5_u1_b1, act_type='relu', name='stage5_unit1_relu1')
    pool = mx.symbol.Pooling(data=s5_u1_a1, global_pool=True, kernel=(6, 6), pool_type='avg', name='pool1')
    flatten = mx.symbol.Flatten(data=pool, name='flatten0')
    fc = mx.symbol.FullyConnected(data=flatten, num_hidden=1000, name='fc1')
    softmax = mx.symbol.SoftmaxOutput(data=fc, name='softmax')

    # remove fc1 params
    arg_new = dict({k: arg_params[k] for k in arg_params if 'fc1' not in k})

    # ---- print newly added layers output dim ----
    print('==> Changed layers output dimension:')
    print('stage4_unit3_bn3:', s4_u3_b3.infer_shape(data=input_shape)[1])
    print('stage4_unit3_relu3:', s4_u3_a3.infer_shape(data=input_shape)[1])
    print('stage5_unit1_conv1:', s5_u1_c1.infer_shape(data=input_shape)[1])
    print('stage5_unit1_bn1:', s5_u1_b1.infer_shape(data=input_shape)[1])
    print('stage5_unit1_relu1:', s5_u1_a1.infer_shape(data=input_shape)[1])
    print('pool1:', pool.infer_shape(data=input_shape)[1])
    print('flatten1:', flatten.infer_shape(data=input_shape)[1])
    print('fc1:', fc.infer_shape(data=input_shape)[1])
    print('softmax:', softmax.infer_shape(data=input_shape)[1])
    # ---------------------------------------------

    print('==> New params:')
    pprint.pprint(zip(softmax.list_arguments(), softmax.infer_shape(data=input_shape)[0]))
    return softmax, arg_new


def main():
    sym, arg_params, aux_params = mx.model.load_checkpoint(sys.argv[1], 0)   # load original model
    print('==> Original params:')
    pprint.pprint(zip(sym.list_arguments(), sym.infer_shape(data=(1, 3, 224, 224))[0]))
    sym_new, arg_new = modify_net(sym, arg_params)
    mx.model.save_checkpoint(sys.argv[1] + '-modified', 0, sym_new, arg_new, aux_params)
    print('New model saved at: {}'.format(sys.argv[1] + '-modified'))


if __name__ == '__main__':
    main()
