import numpy as np

from utils.Transformation import Transformation


class ArtificialDataIterator:
    """
    This class implements a data iterator that samples batches from an artificial dataset.
    """

    @staticmethod
    def next_batch(batch_size, transform=False, size=8):
        """
        This methods retrieves a new batch.
        :param batch_size: the batch size
        :param transform: usage of the copula transformation
        :param size: #additional dimensions
        :return: x batch, y batch, non transformed y batch
        """
        x1 = np.random.uniform(0, 2, batch_size)
        x2 = np.random.uniform(0, 2, batch_size)

        x1_t = Transformation.to_beta(x1, 10, 0.25)
        x2_t = Transformation.to_beta(x2, 0.25, 0.25)

        x_add = np.zeros((batch_size, size))
        x_add_t = np.zeros((batch_size, size))

        x3 = np.random.uniform(0, 1, batch_size)
        x4 = np.random.uniform(0, 1, batch_size)
        for i in range(size):
            r = np.random.uniform(1)
            tmp = r * x3 + (1 - r) * x4 + 3e-1 * np.random.uniform(0,1,batch_size)
            x_add[:, i] = np.reshape(tmp,(1, batch_size))
            x_add_t[:, i] = Transformation.to_beta(tmp, np.random.uniform(0.2, 10, 1), 0.2)

        temp = np.reshape((x1_t, x2_t), (2, batch_size))
        X = np.transpose(temp)
        X = np.concatenate((X,x_add_t), axis = 1)

        if transform:
            X = Transformation.toUniform(X)

        X = np.asarray(X, dtype='float32')

        z_true = (np.sqrt(np.square(x1) + np.square(x2)))
        z_true = z_true / np.max(z_true) + 1e-2 * np.random.normal(0, 1, batch_size)

        z_true_2 = z_true + np.reshape(x_add[:,0], (1, batch_size))
        z_true_2 = z_true_2 / np.max(z_true_2) + 1e-2 * np.random.uniform(0, 1, batch_size)

        r = z_true_2
        phi = 1.75 * np.pi * z_true

        y1 = r * np.cos(phi)
        y2 = r * np.sin(phi)

        y_add = np.zeros((batch_size, size))
        for i in range(size):
            r <- np.random.uniform(1)
            tmp = r*y1 + (1-r) * y2 + 1e-1*np.random.normal(0,1, batch_size)
            y_add[:,i] = np.reshape(tmp, (1, batch_size))


        y1 = y1 + 1e-1 * np.random.normal(0, 1, batch_size)
        y2 = y2 + 1e-1 * np.random.normal(0, 1, batch_size)

        # Bringing data in the right form
        Y = np.transpose(np.reshape((y1, y2), (2, batch_size)))
        Y = np.concatenate((Y,y_add), axis = 1)

        Y_orig = Y
        Y = Transformation.toUniform(Y)

        Y = np.asarray(Y, dtype='float32')
        Y_orig = np.asarray(Y_orig, dtype='float32')

        return (X, Y, Y_orig)