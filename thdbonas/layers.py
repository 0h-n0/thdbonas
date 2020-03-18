# -*- coding: utf-8 -*-
#
# Copyright 2018-2020 Data61, CSIRO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import activations, initializers, constraints, regularizers
from tensorflow.keras.layers import Input, Layer, Lambda, Dropout, Reshape


class GraphConvolution(Layer):

    """
    Graph Convolution (GCN) Keras layer.
    The implementation is based on the keras-gcn github repo https://github.com/tkipf/keras-gcn.

    Original paper: Semi-Supervised Classification with Graph Convolutional Networks. Thomas N. Kipf, Max Welling,
    International Conference on Learning Representations (ICLR), 2017 https://github.com/tkipf/gcn

    Notes:
      - The inputs are tensors with a batch dimension of 1:
        Keras requires this batch dimension, and for full-batch methods
        we only have a single "batch".

      - There are three inputs required, the node features, the output
        indices (the nodes that are to be selected in the final layer)
        and the normalized graph Laplacian matrix

      - This class assumes that the normalized Laplacian matrix is passed as
        input to the Keras methods.

      - The output indices are used when ``final_layer=True`` and the returned outputs
        are the final-layer features for the nodes indexed by output indices.

      - If ``final_layer=False`` all the node features are output in the same ordering as
        given by the adjacency matrix.

    Args:
        units (int): dimensionality of output feature vectors
        activation (str or func): nonlinear activation applied to layer's output to obtain output features
        use_bias (bool): toggles an optional bias
        final_layer (bool): If False the layer returns output for all nodes,
                            if True it returns the subset specified by the indices passed to it.
        kernel_initializer (str or func, optional): The initialiser to use for the weights.
        kernel_regularizer (str or func, optional): The regulariser to use for the weights.
        kernel_constraint (str or func, optional): The constraint to use for the weights.
        bias_initializer (str or func, optional): The initialiser to use for the bias.
        bias_regularizer (str or func, optional): The regulariser to use for the bias.
        bias_constraint (str or func, optional): The constraint to use for the bias.
    """

    def __init__(
            self,
            units,
            activation=None,
            use_bias=True,
            final_layer=False,
            input_dim=None,
            kernel_initializer="glorot_uniform",
            kernel_regularizer=None,
            kernel_constraint=None,
            bias_initializer="zeros",
            bias_regularizer=None,
            bias_constraint=None,
            **kwargs
    ):
        if "input_shape" not in kwargs and input_dim is not None:
            kwargs["input_shape"] = (input_dim,)

        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.final_layer = final_layer

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)

        super().__init__(**kwargs)

    def get_config(self):
        """
        Gets class configuration for Keras serialization.
        Used by keras model serialization.

        Returns:
            A dictionary that contains the config of the layer
        """

        config = {
            "units": self.units,
            "use_bias": self.use_bias,
            "final_layer": self.final_layer,
            "activation": activations.serialize(self.activation),
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "bias_constraint": constraints.serialize(self.bias_constraint),
        }

        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shapes):
        """
        Computes the output shape of the layer.
        Assumes the following inputs:

        Args:
            input_shapes (tuple of ints)
                Shape tuples can include None for free dimensions, instead of an integer.

        Returns:
            An input shape tuple.
        """
        feature_shape, out_shape, *As_shapes = input_shapes

        batch_dim = feature_shape[0]
        if self.final_layer:
            out_dim = out_shape[1]
        else:
            out_dim = feature_shape[1]

        return batch_dim, out_dim, self.units

    def build(self, input_shapes):
        """
        Builds the layer

        Args:
            input_shapes (list of int): shapes of the layer's inputs (node features and adjacency matrix)

        """
        feat_shape = input_shapes[0]
        input_dim = int(feat_shape[-1])

        self.kernel = self.add_weight(
            shape=(input_dim, self.units),
            initializer=self.kernel_initializer,
            name="kernel",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                name="bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None
        self.built = True


    def call(self, inputs):
        """
        Applies the layer.

        Args:
            inputs (list): a list of 3 input tensors that includes
                node features (size 1 x N x F),
                output indices (size 1 x M)
                graph adjacency matrix (size N x N),
                where N is the number of nodes in the graph, and
                F is the dimensionality of node features.

        Returns:
            Keras Tensor that represents the output of the layer.
        """
        features, *As = inputs
        batch_dim, n_nodes, _ = K.int_shape(features)
        if batch_dim != 1:
            raise ValueError(
                "Currently full-batch methods only support a batch dimension of one"
            )

        # Remove singleton batch dimension
        features = K.squeeze(features, 0)

        # Calculate the layer operation of GCN
        A = As[0]
        h_graph = K.dot(A, features)
        output = K.dot(h_graph, self.kernel)

        # Add optional bias & apply activation
        if self.bias is not None:
            output += self.bias
        output = self.activation(output)

        return output


class GraphGather(Layer):
    # copy from https://github.com/clinfo/kGCN/blob/master/kgcn/layers.py
    def __init__(self, **kwargs):
        super(GraphGather, self).__init__(**kwargs)

    def build(self, input_shape):  # input: batch_size x node_num x #inputs
        super(GraphGather, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return tf.reduce_sum(inputs, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]
