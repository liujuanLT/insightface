
# def fire_module(inputs,
#                 squeeze_depth,
#                 expand_depth,
#                 reuse=None,
#                 scope=None,
#                 outputs_collections=None):
#     with tf.variable_scope(scope, 'fire', [inputs], reuse=reuse):
#         with slim.arg_scope([slim.conv2d, slim.max_pool2d],
#                             outputs_collections=None):
#             net = squeeze(inputs, squeeze_depth)
#             outputs = expand(net, expand_depth)
#             return outputs

# def squeeze(inputs, num_outputs):
#     return slim.conv2d(inputs, num_outputs, [1, 1], stride=1, scope='squeeze')

# def expand(inputs, num_outputs):
#     with tf.variable_scope('expand'):
#         e1x1 = slim.conv2d(inputs, num_outputs, [1, 1], stride=1, scope='1x1')
#         e3x3 = slim.conv2d(inputs, num_outputs, [3, 3], scope='3x3')
#     return tf.concat([e1x1, e3x3], 3)

# def inference(images, keep_probability, phase_train=True, bottleneck_layer_size=128, weight_decay=0.0, reuse=None):
#     batch_norm_params = {
#         # Decay for the moving averages.
#         'decay': 0.995,
#         # epsilon to prevent 0s in variance.
#         'epsilon': 0.001,
#         # force in-place updates of mean and variance estimates
#         'updates_collections': None,
#         # Moving averages ends up in the trainable variables collection
#         'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
#     }
#     with slim.arg_scope([slim.conv2d, slim.fully_connected],
#                         weights_initializer=slim.xavier_initializer_conv2d(uniform=True),
#                         weights_regularizer=slim.l2_regularizer(weight_decay),
#                         normalizer_fn=slim.batch_norm,
#                         normalizer_params=batch_norm_params):
#         with tf.variable_scope('squeezenet', [images], reuse=reuse):
#             with slim.arg_scope([slim.batch_norm, slim.dropout],
#                                 is_training=phase_train):
#                 net = slim.conv2d(images, 96, [7, 7], stride=2, scope='conv1')
#                 net = slim.max_pool2d(net, [3, 3], stride=2, scope='maxpool1')
#                 net = fire_module(net, 16, 64, scope='fire2')
#                 net = fire_module(net, 16, 64, scope='fire3')
#                 net = fire_module(net, 32, 128, scope='fire4')
#                 net = slim.max_pool2d(net, [2, 2], stride=2, scope='maxpool4')
#                 net = fire_module(net, 32, 128, scope='fire5')
#                 net = fire_module(net, 48, 192, scope='fire6')
#                 net = fire_module(net, 48, 192, scope='fire7')
#                 net = fire_module(net, 64, 256, scope='fire8')
#                 net = slim.max_pool2d(net, [3, 3], stride=2, scope='maxpool8')
#                 net = fire_module(net, 64, 256, scope='fire9')
#                 net = slim.dropout(net, keep_probability)
#                 net = slim.conv2d(net, 1000, [1, 1], activation_fn=None, normalizer_fn=None, scope='conv10')
#                 net = slim.avg_pool2d(net, net.get_shape()[1:3], scope='avgpool10')
#                 net = tf.squeeze(net, [1, 2], name='logits')
#                 net = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None, 
#                         scope='Bottleneck', reuse=False)
#     return net, None



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
import mxnet as mx
import numpy as np
import symbol_utils
import memonger
import sklearn
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import config


def Conv(**kwargs):
    body = mx.sym.Convolution(**kwargs)
    return body

def fire_module(inputs, squeeze_depth, expand_depth, bn_mom, act_type, scope, workspace):
    body = squeeze(inputs, squeeze_depth, bn_mom, act_type, scope, workspace)
    outputs = expand(body, expand_depth, bn_mom, act_type, scope, workspace)
    return outputs

def squeeze(inputs, num_outputs, bn_mom, act_type, scope, workspace):
    # return slim.conv2d(inputs, num_outputs, [1, 1], stride=1, scope= name + '_squeeze')
    net = Conv(data=inputs, num_filter=num_outputs, kernel=(1, 1), stride=(1, 1), pad=(0, 0), no_bias=True, name=scope+'_squeeze_conv',
            workspace=workspace)
    net = mx.sym.BatchNorm(data=net, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=scope+'_squeeze_bn')
    net = Act(data=net, act_type=act_type, name=scope+'_squeeze_relu')
    return net

def expand(inputs, num_outputs, bn_mom, act_type, scope, workspace):
    # e1x1 = slim.conv2d(inputs, num_outputs, [1, 1], stride=1, scope='1x1')
    e1x1 = Conv(data=inputs, num_filter=num_outputs, kernel=(1, 1), stride=(1, 1), pad=(0, 0), no_bias=True, name=scope+'_expand_1x1_conv',
        workspace=workspace)
    e1x1 = mx.sym.BatchNorm(data=e1x1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=scope+'_expand_1x1_bn')
    e1x1 = Act(data=e1x1, act_type=act_type, name=scope+'_expand_1x1_relu')
    # e3x3 = slim.conv2d(inputs, num_outputs, [3, 3], scope='3x3')
    e3x3 = Conv(data=inputs, num_filter=num_outputs, kernel=(3, 3), stride=(1, 1), pad=(1, 1), no_bias=True, name=scope+'_expand_3x3_conv',
        workspace=workspace)
    e3x3 = mx.sym.BatchNorm(data=e3x3, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=scope+'_expand_3x3_bn')
    e3x3 = Act(data=e3x3, act_type=act_type, name=scope+'_expand_3x3_relu')
    # return tf.concat([e1x1, e3x3], 3)
    return mx.symbol.concat(e1x1, e3x3, dim=1)

def Act(data, act_type, name):
    if act_type == 'prelu':
        body = mx.sym.LeakyReLU(data=data, act_type='prelu', name=name)
    else:
        body = mx.symbol.Activation(data=data, act_type=act_type, name=name)
    return body

def get_symbol(keep_probability=0.8):
    bn_mom = config.bn_mom
    act_type = config.net_act
    workspace = config.workspace
    num_classes = config.emb_size
    data = mx.sym.Variable(name='data')
    data = mx.sym.identity(data=data, name='id')
    data = data - 127.5
    data = data * 0.0078125
    net = data
    net = Conv(data=net, num_filter=96, kernel=(7, 7), stride=(2, 2), pad=(3, 3), no_bias=True, name="conv1",
            workspace=workspace) # python def pad=SAME, in=112,112, 3, out=56,56,96
    net = mx.sym.BatchNorm(data=net, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
    net = Act(data=net, act_type=act_type, name='relu1')
    net = mx.sym.Pooling(data=net, kernel=(3, 3), stride=(2, 2), pool_type='max', name = 'maxpool1') # python def pad=valid, out=27,27,96
    net = fire_module(net, 16, 64, bn_mom, act_type, 'fire2', workspace) # out = 27, 27, 128. (after squueze:27,27,16; after expand: out=27, 27, 128(e1x1:27,27,27,64, e3x3:27,27,64))
    net = fire_module(net, 16, 64, bn_mom, act_type, 'fire3', workspace) # out = 27,27,128
    net = fire_module(net, 32, 128, bn_mom, act_type, 'fire4', workspace) # out = 27, 27, 256
    net = mx.sym.Pooling(data=net, kernel=(2, 2), stride=(2, 2), pool_type='max', name = 'maxpool4') # python def pad=valid, out=13,13,256
    net = fire_module(net, 32, 128, bn_mom, act_type, 'fire5', workspace) # out = 13,13,256
    net = fire_module(net, 48, 192, bn_mom, act_type, 'fire6', workspace) # out = 13, 13, 384
    net = fire_module(net, 48, 192, bn_mom, act_type, 'fire7', workspace) # out = 13, 13 384
    net = fire_module(net, 64, 256, bn_mom, act_type, 'fire8', workspace) # out = 13, 13, 512
    net = mx.sym.Pooling(data=net, kernel=(3, 3), stride=(2, 2), pool_type='max', name = 'maxpool8') # python def pad=valid, out = 6, 6, 512
    net = fire_module(net, 64, 256, bn_mom, act_type, 'fire9', workspace) # out = 6,6,512
    net = mx.sym.BatchNorm(data=net, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn_beforelast') # NOTE, tf has no this
    # net = mx.symbol.Dropout(data=net, p=keep_probability) # out = 6,6,512
    # net = Conv(data=net, num_filter=1000, kernel=(1, 1), stride=(1, 1), pad=(0, 0), no_bias=True, name="conv10",
    #         workspace=workspace) # python def pad=SAME, out=6,6,1000
    # net = mx.sym.Pooling(data=net, global_pool=True, pool_type='avg', name = 'avgpool10') # python def pad=valid, out=1,1,1000
    # net = mx.sym.squeeze(data=net, axis=(2,3), name='logits') # out=1000
    net = mx.symbol.Dropout(data=net, p=keep_probability) # NOTE, TF has not this ilne
    net = mx.sym.FullyConnected(data=net, num_hidden=num_classes, name='fc1_output') # out=emb_size
    net = mx.sym.BatchNorm(data=net, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_last') # TODO, fix gamma = true or false?
    return net

    # TODO。然后加上batchnorm和act(done), 然后确认batchnorm和drop_out VS phase,
    # TODO: debug为什么获取不到fc1_output
    # TODO: weight init(model.fit时指定的参数), weight regular (weight decay)
