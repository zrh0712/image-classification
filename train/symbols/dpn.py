# simply merged different num-layers dual-path networks into one script
# original paper: https://arxiv.org/pdf/1707.01629
# original codes: https://github.com/cypw/DPNs
# @Northrend 2017-08-31 13:01:29
#

import mxnet as mx
from dpn_basic import *

# - - - - - - - - - - - - - - - - - - - - - - -
# Standard Dual Path Unit
def DualPathFactory(data, num_1x1_a, num_3x3_b, num_1x1_c, name, inc, G, _type='normal'):
    kw = 3
    kh = 3
    pw = (kw-1)/2
    ph = (kh-1)/2

    # type
    if _type is 'proj':
        key_stride = 1
        has_proj   = True
    if _type is 'down':
        key_stride = 2
        has_proj   = True
    if _type is 'normal':
        key_stride = 1
        has_proj   = False

    # PROJ
    if type(data) is list:
        data_in  = mx.symbol.Concat(*[data[0], data[1]],  name=('%s_cat-input' % name))
    else:
        data_in  = data

    if has_proj:
        c1x1_w   = BN_AC_Conv( data=data_in, num_filter=(num_1x1_c+2*inc), kernel=( 1, 1), stride=(key_stride, key_stride), name=('%s_c1x1-w(s/%d)' %(name, key_stride)), pad=(0, 0))
        data_o1  = mx.symbol.slice_axis(data=c1x1_w, axis=1, begin=0,         end=num_1x1_c,         name=('%s_c1x1-w(s/%d)-split1' %(name, key_stride)))
        data_o2  = mx.symbol.slice_axis(data=c1x1_w, axis=1, begin=num_1x1_c, end=(num_1x1_c+2*inc), name=('%s_c1x1-w(s/%d)-split2' %(name, key_stride)))
    else:
        data_o1  = data[0]
        data_o2  = data[1]
        
    # MAIN
    c1x1_a = BN_AC_Conv( data=data_in, num_filter=num_1x1_a,       kernel=( 1,  1), pad=( 0,  0), name=('%s_c1x1-a'   % name))
    c3x3_b = BN_AC_Conv( data=c1x1_a,  num_filter=num_3x3_b,       kernel=(kw, kh), pad=(pw, ph), name=('%s_c%dx%d-b' % (name,kw,kh)), stride=(key_stride,key_stride), num_group=G)
    c1x1_c = BN_AC_Conv( data=c3x3_b,  num_filter=(num_1x1_c+inc), kernel=( 1,  1), pad=( 0,  0), name=('%s_c1x1-c'   % name))
    c1x1_c1= mx.symbol.slice_axis(data=c1x1_c, axis=1, begin=0,         end=num_1x1_c,       name=('%s_c1x1-c-split1' % name))
    c1x1_c2= mx.symbol.slice_axis(data=c1x1_c, axis=1, begin=num_1x1_c, end=(num_1x1_c+inc), name=('%s_c1x1-c-split2' % name))

    # OUTPUTS
    summ   = mx.symbol.ElementWiseSum(*[data_o1, c1x1_c1],                        name=('%s_sum' % name))
    dense  = mx.symbol.Concat(        *[data_o2, c1x1_c2],                        name=('%s_cat' % name))

    return [summ, dense]
    

def check_num_layers(num_layers):
    layer_lst = [68,92,98,107,131]
    if num_layers not in layer_lst:
        return False 
    else:
        return True
        

def get_before_pool(num_layers):
    ## define hyper params
    k_R = {68: 128, 92: 96, 98: 160, 107: 200, 131: 160}

    G   = {68: 32, 92: 32, 98: 40, 107: 50, 131: 40}

    k_sec  = {68:{  2: 3,  \
                    3: 4,  \
                    4: 12, \
                    5: 3    },
              92:{  2: 3,  \
                    3: 4,  \
                    4: 20, \
                    5: 3    },
              98:{  2: 3,  \
                    3: 6,  \
                    4: 20, \
                    5: 3    },
              107:{ 2: 4,  \
                    3: 8,  \
                    4: 20, \
                    5: 3    },
              131:{ 2: 4,  \
                    3: 8,  \
                    4: 28, \
                    5: 3    }}

    inc_sec= {68:{  2: 16, \
                    3: 32, \
                    4: 32, \
                    5: 64   },
              92:{  2: 16, \
                    3: 32, \
                    4: 24, \
                    5: 128  },
              98:{  2: 16, \
                    3: 32, \
                    4: 32, \
                    5: 128  },
              107:{ 2: 20, \
                    3: 64, \
                    4: 64, \
                    5: 128  },
              131:{ 2: 16, \
                    3: 32, \
                    4: 32, \
                    5: 128  }}
    
    conv_1_num_filter = {68: 10, 92: 64, 98: 96, 107: 128, 131: 128}
    conv_1_kernel_size = {68: (3, 3)}
    conv_1_kernel_size.update(dict.fromkeys([92, 98, 107, 131], (7, 7)))
    conv_1_pad_size = {68: (1, 1)}
    conv_1_pad_size.update(dict.fromkeys([92, 98, 107, 131], (3, 3)))
    r_denominator = 64 if num_layers == 68 else 256
    
    ## define Dual Path Network
    data = mx.symbol.Variable(name="data")

    # conv1
    conv1_x_1  = Conv(data=data,  num_filter=conv_1_num_filter[num_layers],  kernel=conv_1_kernel_size[num_layers], 
                      name='conv1_x_1', pad=conv_1_pad_size[num_layers], stride=(2,2))
    conv1_x_1  = BN_AC(conv1_x_1, name='conv1_x_1__relu-sp')
    conv1_x_x  = mx.symbol.Pooling(data=conv1_x_1, pool_type="max", kernel=(3, 3), pad=(1,1), stride=(2,2), name="pool1")

    # conv2
    bw = 64 if num_layers == 68 else 256
    inc= inc_sec[num_layers][2]
    R  = (k_R[num_layers]*bw)/r_denominator 
    conv2_x_x  = DualPathFactory(     conv1_x_x,   R,   R,   bw,  'conv2_x__1',           inc,   G[num_layers],  'proj'  )
    for i_ly in range(2, k_sec[num_layers][2]+1):
        conv2_x_x  = DualPathFactory( conv2_x_x,   R,   R,   bw, ('conv2_x__%d'% i_ly),   inc,   G[num_layers],  'normal')

    # conv3
    bw = 128 if num_layers == 68 else 512
    inc= inc_sec[num_layers][3]
    R  = (k_R[num_layers]*bw)/r_denominator
    conv3_x_x  = DualPathFactory(     conv2_x_x,   R,   R,   bw,  'conv3_x__1',           inc,   G[num_layers],  'down'  )
    for i_ly in range(2, k_sec[num_layers][3]+1):
        conv3_x_x  = DualPathFactory( conv3_x_x,   R,   R,   bw, ('conv3_x__%d'% i_ly),   inc,   G[num_layers],  'normal')

    # conv4
    bw = 256 if num_layers == 68 else 1024
    inc= inc_sec[num_layers][4]
    R  = (k_R[num_layers]*bw)/r_denominator
    conv4_x_x  = DualPathFactory(     conv3_x_x,   R,   R,   bw,  'conv4_x__1',           inc,   G[num_layers],  'down'  )
    for i_ly in range(2, k_sec[num_layers][4]+1):
        conv4_x_x  = DualPathFactory( conv4_x_x,   R,   R,   bw, ('conv4_x__%d'% i_ly),   inc,   G[num_layers],  'normal')

    # conv5
    bw = 512 if num_layers == 68 else 2048
    inc= inc_sec[num_layers][5]
    R  = (k_R[num_layers]*bw)/r_denominator
    conv5_x_x  = DualPathFactory(     conv4_x_x,   R,   R,   bw,  'conv5_x__1',           inc,   G[num_layers],  'down'  )
    for i_ly in range(2, k_sec[num_layers][5]+1):
        conv5_x_x  = DualPathFactory( conv5_x_x,   R,   R,   bw, ('conv5_x__%d'% i_ly),   inc,   G[num_layers],  'normal')

    # output: concat
    conv5_x_x  = mx.symbol.Concat(*[conv5_x_x[0], conv5_x_x[1]],  name='conv5_x_x_cat-final')
    conv5_x_x = BN_AC(conv5_x_x, name='conv5_x_x__relu-sp')
    return conv5_x_x


def get_linear(num_classes = 1000, num_layers = 92):
    before_pool = get_before_pool(num_layers)
    # - - - - -
    pool5     = mx.symbol.Pooling(data=before_pool, pool_type="avg", kernel=(7, 7), stride=(1,1), name="pool5")
    flat5     = mx.symbol.Flatten(data=pool5, name='flatten')
    fc6       = mx.symbol.FullyConnected(data=flat5, num_hidden=num_classes, name='fc6')
    return fc6


def get_symbol(num_classes = 1000, num_layers = 92, image_shape = (3,224,224), **kwargs):
    assert check_num_layers(num_layers), 'num_layers={} not supported yet'.format(num_layers)
    fc6       = get_linear(num_classes, num_layers)
    softmax   = mx.symbol.SoftmaxOutput( data=fc6,  name='softmax')
    # sys_out   = softmax
    return softmax


