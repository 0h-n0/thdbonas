import uuid
import typing

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as L

from .trial import Trial
from .layers import GraphConvolution, GraphGather


class BaseSurrogateModel:
    def __init__(self,
                 n_train_epochs: int = 100,
                 save_path: str = 'network-{uuid.uuid1()}.ckpt'):
        self.n_train_epochs = n_train_epochs
        self.save_path = save_path

    def train(self, xtrain=typing.List[Trial], ytrain=typing.List[float], n_epochs: int=None):
        raise NotImplementedError

    def predict(self, xeval=typing.List[Trial]):
        raise NotImplementedError


class GCNSurrogateModel(BaseSurrogateModel):
    def __init__(self,
                 num_nodes: int = 32,
                 input_channels: int = 3,
                 output_channels: int = 1,
                 hidden_channels: int = 64,
                 n_train_epochs: int = 100,
                 save_path=f'/tmp/gcnnet-{uuid.uuid1()}.ckpt'):
        super(GCNSurrogateModel, self).__init__(n_train_epochs, save_path)
        self.gcn1 = GraphConvolution(16, activation='tanh')
        self.gcn2 = GraphConvolution(32, activation='tanh')
        self.gcn3 = GraphConvolution(64, activation='tanh')
        self.gather = GraphGather()
        self.l1 = L.Dense(hidden_channels)
        self.l2 = L.Dense(output_channels)
        self.bases = None
        self.tf_config = tf.compat.v1.ConfigProto(log_device_placement=False,
                                                  gpu_options=tf.compat.v1.GPUOptions(
                                                      allow_growth=True,
                                                  ))
        self._build_graph(input_channels, num_nodes, output_channels)
        self.vars_to_train = tf.compat.v1.trainable_variables()
        self.saver = tf.compat.v1.train.Saver(self.vars_to_train)

    def last_layer(self, inputs):
        features, adj = inputs
        x = self.gcn1([features, adj])
        x = self.gcn2([x, adj])
        x = self.gcn3([x, adj])
        x = self.gather(x)
        x = self.l1(x)
        self.bases = x
        return x

    def __call__(self, inputs, adj):
        x = self.last_layer([inputs, adj])
        x = self.l2(x)
        return x

    def _build_graph(self, input_channels, num_nodes, output_channles):
        self.graph = tf.compat.v1.get_default_graph()
        with self.graph.as_default():
            self.y_plh = tf.compat.v1.placeholder(tf.float32,
                                              shape=[None, output_channles],
                                              name='ytrain')
            self.x_plh = tf.compat.v1.placeholder(tf.float32, shape=[1, num_nodes, input_channels], name='x')
            self.x_adj_plh = tf.compat.v1.placeholder(tf.float32, shape=[1, num_nodes, num_nodes], name='x_adj')
            out = self(self.x_plh, self.x_adj_plh)
            self.mse_loss = tf.reduce_mean(tf.square(self.y_plh - out))
            self.train_loss = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001).minimize(self.mse_loss)

    def train(self, xtrain=typing.List[Trial], ytrain=typing.List[float], n_epochs: int=None):
        # currently only support batch_size == 1
        if not n_epochs is None:
            n_epochs = self.n_train_epochs
        if True:
            bases = []
            with tf.compat.v1.Session(config=self.tf_config) as sess:
                sess.run(tf.compat.v1.global_variables_initializer())
                for _ in range(n_epochs):
                    for trial, y in zip(xtrain, ytrain):
                        _, (adj, features) = trial.graph
                        features = np.expand_dims(features, axis=0) # add batch dimmension
                        adj = np.expand_dims(adj, axis=0) # add batch dimmension
                        y = np.array(y, dtype=np.float32).reshape(1, 1)
                        sess.run(self.train_loss, feed_dict={self.x_plh: features, self.x_adj_plh: adj, self.y_plh: y})


                for trial, y in zip(xtrain, ytrain):
                    _, (adj, features) = trial.graph
                    features = np.expand_dims(features, axis=0) # add batch dimmension
                    adj = np.expand_dims(adj, axis=0) # add batch dimmension
                    y = np.array(y, dtype=np.float32).reshape(1, 1)
                    bases.append(sess.run(self.bases, feed_dict={self.x_plh: features, self.x_adj_plh: adj, self.y_plh: y}))
                bases = np.concatenate(bases)
                self.saver.save(sess, self.save_path)
        return bases

    def predict(self, xeval=typing.List[Trial]):
        bases = []
        with self.graph.as_default():
            with tf.compat.v1.Session(config=self.tf_config) as sess:
                self.saver.restore(sess, self.save_path)
                for trial in xeval:
                    _, (adj, features) = trial.graph
                    features = np.expand_dims(features, axis=0) # add batch dimmension
                    adj = np.expand_dims(adj, axis=0) # add batch dimmension
                    y = np.ones((1, 1), dtype=np.float32) # dummy input
                    bases.append(sess.run(self.bases, feed_dict={self.x_plh: features, self.x_adj_plh: adj, self.y_plh: y}))
                bases = np.concatenate(bases)
        return bases


class SimpleNetwork(BaseSurrogateModel):
    def __init__(self,
                 input_dim: int = 4,
                 output_dim: int = 1,
                 hidden_dim: int = 64,
                 activation='tanh',
                 n_train_epochs: int = 100,
                 save_path=f'/tmp/simplenetwork-{uuid.uuid1()}.ckpt'):
        super(SimpleNetwork, self).__init__(n_train_epochs, save_path)
        self.first_layer = tf.keras.models.Sequential([
            L.Dense(16, activation),
            L.Dense(32, activation),
            L.Dense(hidden_dim, activation)])
        self.last_layer = L.Dense(output_dim)
        self.tf_config = tf.ConfigProto(log_device_placement=False,
                                        gpu_options=tf.GPUOptions(
                                            allow_growth=True,
                                        ))
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_train_epochs = n_train_epochs
        self._build_graph(input_dim, output_dim)
        self.saver = tf.compat.v1.train.Saver()

    def __call__(self, x):
        self.bases = self.first_layer(x)
        return self.last_layer(self.bases)

    def _build_graph(self, xdim: int, ydim: int):
        if True:
            self.y_plh_train = tf.placeholder(tf.float32, shape=[None, ydim], name='ytrain')
            self.x_plh_train = tf.placeholder(tf.float32, shape=[None, xdim], name='xtrain')
            out = self(self.x_plh_train)
            mse_loss = tf.reduce_mean(tf.square(self.y_plh_train - out))
            self.train_loss = tf.train.AdamOptimizer(learning_rate=0.001).minimize(mse_loss)

    def train(self, xtrain=typing.List[Trial], ytrain=typing.List[float], n_epochs: int=None):
        if not n_epochs is None:
            n_epochs = self.n_train_epochs
        if True:
            bases = []
            with tf.Session(config=self.tf_config) as sess:
                sess.run(tf.global_variables_initializer())
                for _ in range(n_epochs):
                    for x, y in zip(xtrain, ytrain):
                        x = x.to_numpy().reshape(1, self.input_dim)
                        y = np.array(y, dtype=np.float32).reshape(1, 1)
                        sess.run(self.train_loss, feed_dict={self.x_plh_train: x, self.y_plh_train: y})
                for x, y in zip(xtrain, ytrain):
                    x = x.to_numpy().reshape(1, self.input_dim)
                    y = np.array(y, dtype=np.float32).reshape(1, 1)
                    bases.append(sess.run(self.bases, feed_dict={self.x_plh_train: x, self.y_plh_train: y}))
                bases = np.concatenate(bases)
                self.saver.save(sess, self.save_path)
        return bases

    def predict(self, xeval=typing.List[Trial]):
        bases = []
        with tf.Session(config=self.tf_config) as sess:
            self.saver.restore(sess, self.save_path)
            for x in xeval:
                x = x.to_numpy().reshape(1, self.input_dim)
                y = np.ones((1, 1), dtype=np.float32) # dummy input
                bases.append(sess.run(self.bases, feed_dict={self.x_plh_train: x, self.y_plh_train: y}))
            bases = np.concatenate(bases)
        return bases
