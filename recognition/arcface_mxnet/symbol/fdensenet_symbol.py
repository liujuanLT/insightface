# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
# pylint: disable= arguments-differ
"""DenseNet, implemented in symbol, refereced from fdensenet.py."""

import sys
import os
import mxnet as mx
import symbol_utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import config

def Conv(**kwargs):
    body = mx.sym.Convolution(**kwargs)
    return body

def Act(data, act_type, name):
    if act_type == 'prelu':
        body = mx.sym.LeakyReLU(data=data, act_type='prelu', name=name)
    else:
        body = mx.symbol.Activation(data=data, act_type=act_type, name=name)
    return body

def _make_dense_block(net, num_layers, bn_size, bn_mom, bn_eps, growth_rate, dropout, act_type, workspace, name):
    for ilayer in range(num_layers):
        out = _make_dense_layer(net, growth_rate, bn_size, bn_mom, bn_eps, dropout, act_type, workspace, name = name + '_layer' + str(ilayer))
        net = mx.symbol.concat(net, out, name = name + '_concat' + str(ilayer))
    return net

def _make_dense_layer(net, growth_rate, bn_size, bn_mom, bn_eps, dropout, act_type, workspace, name):
    net = mx.sym.BatchNorm(data=net,
                            fix_gamma=False,
                            eps=bn_eps,
                            momentum=bn_mom,
                            name=name+'_bn0')
    net = Act(net, act_type, name=name+'_relu0')
    net = Conv(data=net,
        num_filter=bn_size * growth_rate,
        kernel=(1, 1),
        stride=(1, 1),
        pad=(0, 0),
        no_bias=True,
        name=name+"_conv0",
        workspace=workspace)
    net = mx.sym.BatchNorm(data=net,
                            fix_gamma=False,
                            eps=bn_eps,
                            momentum=bn_mom,
                            name=name+'_bn1')
    net = Act(net, act_type, name=name+'_relu1')
    net = Conv(data=net,
        num_filter=growth_rate,
        kernel=(3, 3),
        stride=(1, 1),
        pad=(1, 1),
        no_bias=True,
        name=name+"_conv1",
        workspace=workspace)
    if dropout:
        net = mx.symbol.Dropout(data=net, p=dropout)

    return net


def _make_transition(net, num_output_features, bn_mom, bn_eps, act_type, workspace, name):
    net = mx.sym.BatchNorm(data=net,
                            fix_gamma=False,
                            eps=bn_eps,
                            momentum=bn_mom,
                            name=name+'_bn0')
    net = Act(net, act_type, name=name+'_relu0')
    net = Conv(data=net,
        num_filter=num_output_features,
        kernel=(1, 1),
        stride=(1, 1),
        pad=(0, 0),
        no_bias=True,
        name=name+"_conv0",
        workspace=workspace)
    net = mx.sym.Pooling(data=net, kernel=(2, 2), stride=(2, 2), pad=(0,0), pool_type='avg', name = name+'_maxpool0')
    return net


def get_symbol():
    r"""Densenet-BC model from the
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_ paper.

    Parameters
    ----------
    num_init_features : int
        Number of filters to learn in the first convolution layer.
    growth_rate : int
        Number of filters to add each layer (`k` in the paper).
    block_config : list of int
        List of integers for numbers of layers in each pooling block.
    bn_size : int, default 4
        Multiplicative factor for number of bottle neck layers.
        (i.e. bn_size * k features in the bottleneck layer)
    dropout : float, default 0
        Rate of dropout after each dense layer.
    classes : int, default 1000
        Number of classification classes.
    """
    num_layers = config.num_layers
    workspace = config.workspace
    # Specification
    densenet_spec = {
    121: (64, 32, [6, 12, 24, 16]),
    161: (96, 48, [6, 12, 36, 24]),
    169: (64, 32, [6, 12, 32, 32]),
    201: (64, 32, [6, 12, 48, 32])
    }

    num_init_features, growth_rate, block_config = densenet_spec[num_layers]
    bn_size=4
    bn_mom = 0.9 # default value of mxnet.gluon.nn.BatchNorm
    bn_eps = 1e-5 # default value of mxnet.gluon.nn.BatchNorm
    dropout=config.densenet_dropout
    act_type_local = 'relu'
    act_type_global = config.net_act
    data = mx.sym.Variable(name='data')
    data = mx.sym.identity(data=data, name='id')
    data = data - 127.5
    data = data * 0.0078125
    body = data
    body = Conv(data=body,
            num_filter=num_init_features,
            kernel=(3, 3),
            stride=(1, 1),
            pad=(1, 1),
            no_bias=True,
            name="conv0",
            workspace=workspace)
    body = mx.sym.BatchNorm(data=body,
                            fix_gamma=False,
                            eps=bn_eps,
                            momentum=bn_mom,
                            name='bn0')
    body = Act(data=body, act_type=act_type_local, name='relu0')
    body = mx.sym.Pooling(data=body, kernel=(3, 3), stride=(2, 2), pad=(1,1), pool_type='max', name = 'maxpool0')
    num_features = num_init_features
    for i, num_layers in enumerate(block_config):
        body = _make_dense_block(body, num_layers, bn_size, bn_mom, bn_eps, growth_rate,
                                dropout, act_type_global, workspace, 'stage%d_' % i)
        num_features = num_features + num_layers * growth_rate
        if i != len(block_config) - 1:
            body = _make_transition(body, num_features // 2, bn_mom, bn_eps, act_type_global, workspace, 'transition%d_' % i)
            num_features = num_features // 2
    body = mx.sym.BatchNorm(data=body,
                            fix_gamma=False,
                            eps=bn_eps,
                            momentum=bn_mom,
                            name='bn2')
    body = Act(data=body, act_type=act_type_local, name='relu2')
    #self.features.add(nn.AvgPool2D(pool_size=7))
    #self.features.add(nn.Flatten())
    #self.output = nn.Dense(classes)
    fc1 = symbol_utils.get_fc1(body, config.emb_size, config.net_output)
    return fc1
