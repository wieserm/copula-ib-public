import logging

import numpy as np
import tensorflow as tf
from utils.IOTools import IOTools
from tensorflow.contrib import slim as slim
from tqdm import tqdm

from ArtificialDataIterator import ArtificialDataIterator
from utils.Transformation import Transformation


class ArtificialTraining:
    """
    This class implements the basic training procedure for the artificial dataset..
    """

    def __init__(self, dump_path, learning_rate, batch_size, hidden_dim, doTransform):
        """
        Constructor to initialize all class member variables.
        :param dump_path: the path to save the results
        :param learning_rate: the current learning rate
        :param batch_size: the batch size
        :param hidden_dim: size of hidden dim
        :param doTransform: the use of the copula transformation
        """

        self.dump_path = dump_path
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.doTransform = doTransform
        self.logger = logging.getLogger('root')

    def __build_encoder(self, x, z_dim):
        """
        This method builds the encoder network of the model.
        :param x: input data x
        :param z_dim: size of latent dimension z
        :return: first moments of hidden dimension z
        """

        self.logger.debug("create encoder")

        # define network
        net = slim.fully_connected(x, 50, activation_fn=tf.nn.softplus)
        net = slim.fully_connected(net, 50, activation_fn=tf.nn.softplus)

        # get moments
        z_mu = slim.fully_connected(net, z_dim, activation_fn=None)
        return z_mu

    def __build_decoder(self, z, y_dim):
        """
        This method builds the decoder network of the model.
        :param z: hidden code z
        :param y_dim: output dimension y
        :return: the first and second moments of y
        """

        self.logger.debug("create decoder")

        # define network
        net = slim.fully_connected(z, 50, activation_fn=tf.nn.softplus)
        net = slim.fully_connected(net, 50, activation_fn=tf.nn.softplus)

        # get moments
        y_hat = slim.fully_connected(net, y_dim, activation_fn=None)
        y_ls2 = slim.fully_connected(net, y_dim, activation_fn=None)

        return y_hat, y_ls2

    def train(self):
        """
        This methods builds and trains the current model.
        """

        self.logger.info("train model")

        tf.reset_default_graph()

        # define placeholder
        x = tf.placeholder('float32', [None, 10])
        y = tf.placeholder('float32', [None, 10])
        lambda_val = tf.placeholder('float32', [1, 1])

        # build encoder
        z_mu = self.__build_encoder(x, self.hidden_dim)

        # parametrize sparsity layer
        ada = tf.matmul(tf.transpose(z_mu), z_mu) * (1.0 / self.batch_size)
        a_dp_a = tf.diag_part(ada)
        z_ls2 = tf.log(a_dp_a + 1)

        # calc z
        eps = tf.random_normal((self.batch_size, self.hidden_dim), 0, 1, dtype=tf.float32)  # Adding a random number
        z = tf.add(z_mu, eps)

        # build decoder
        y_hat, y_ls2 = self.__build_decoder(z, 10)

        # define loss
        reconstr_loss = lambda_val * tf.reduce_sum(0.5 * y_ls2 + (tf.square(y - y_hat) / (2.0 * tf.exp(y_ls2))), 1)
        latent_loss = 0.5 * tf.reduce_sum(z_ls2)
        total_loss = tf.reduce_mean(reconstr_loss) + latent_loss

        # define optimizer
        optimizer = tf.train.AdamOptimizer(self.learning_rate, epsilon=1e-8).minimize(total_loss)

        # run training
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())

            # init data iterator
            number_of_iterations = 70000

            itx = list()
            ity = list()
            h_y = list()
            nzn = list()
            lambda_list = list()
            latent_list = list()
            lambda_value = 0.4

            # run training procedure
            for epoch in tqdm(range(number_of_iterations)):

                # sample new batch
                x_batch, y_batch, y_orig_batch = ArtificialDataIterator.next_batch(self.batch_size, self.doTransform)

                # run training
                _, loss, ll, rl, sparse_matrix, y_mu, zmu = session.run(
                    (optimizer, total_loss, latent_loss, reconstr_loss, a_dp_a, y_hat, z_mu),
                    feed_dict={x: x_batch, y: y_batch, lambda_val: np.asarray([[lambda_value]])})

                if (epoch % 500 == 0 and epoch > 0):

                    # if latent loss higher 0.1
                    if (np.mean(ll) > 1e-1):

                        # save MI(x,z)
                        itx.append(np.mean(ll))
                        lambda_list.append(lambda_value)
                        latent_list.append(zmu)

                        #calc empirical Y entropy
                        entropy = np.mean(np.absolute(np.asarray(h_y)))

                        print(
                            "Cost: %.2f, I(x,t): %.4f, I(t,y): %4f" % (
                            loss, np.mean(ll), -(np.mean(rl) / lambda_value) - entropy))

                        # save MI (z,y)
                        ity.append(-(np.mean(rl) / lambda_value) - entropy)

                        # save size of used latent dimensions
                        num_latent_dim = len([i for i, v in enumerate(sparse_matrix) if v > 0.25])
                        nzn.append(num_latent_dim)

                        mi_x_t = np.asarray(itx)
                        mi_t_y = np.asarray(ity)
                        nzn_array = np.asarray(nzn)

                        nbins = int(min(12, max(1, np.floor(len(mi_x_t) / 3))))
                        breaks = np.linspace(0.99 * min(mi_x_t), max(mi_x_t), nbins + 1)

                        xl = list()
                        yl = list()
                        yl_means = list()

                        nzn_list = list()

                        kc = 0

                        for k in range(nbins):
                            matchings_indices = [i for i, item in enumerate(mi_x_t) if
                                                 item > breaks[k] and item < breaks[k + 1]]
                            # if more than 3 MI -> create new bin
                            if len(matchings_indices) > 3:
                                xl.append(np.mean(mi_x_t[matchings_indices]))
                                yl.append(mi_t_y[matchings_indices])
                                yl_means.append(np.median(mi_t_y[matchings_indices]))

                                nzn_list.append(np.min(nzn_array[matchings_indices]))
                                kc += 1
                    else:
                        # collect mutual information in order to calculate the empirical entropy of Y
                        h_y.append(-(np.mean(rl) / lambda_value))

                    # increase compression parameter lambda
                    lambda_value = lambda_value * 1.06


            IOTools.save_to_file((yl_means, yl, xl, sparse_matrix, nzn_list, nzn, ity, itx, y_orig_batch, zmu,
                                  Transformation.UniformToOrig(y_orig_batch, y_mu), lambda_list, latent_list),
                                 self.dump_path)