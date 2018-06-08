import matplotlib

matplotlib.use('Agg')
import platform
import tensorflow as tf
from utils import Logger
from EvaluateArtificialDataset import EvaluateArtificialDataset
from ArtificialTraining import ArtificialTraining
import argparse
import os

# init logger
logger = Logger.setup_custom_logger('root')

logger.info('Initialize Logger')
logger.info('Using python version: %s and Tensorflow version: %s' % (platform.python_version(), tf.__version__))


def train_artificial_dataset(args, do_transform=False):
    """
    This method trains the artificial dataset.
    :param args: command line arguments
    :param do_transform: use copula transformation
    """

    if do_transform:
        trainer = ArtificialTraining(dump_path=args.path + args.copula_file, learning_rate=6e-4, batch_size=500,
                                    hidden_dim=10, doTransform=do_transform)
        trainer.train()
    else:
        trainer = ArtificialTraining(dump_path=args.path + args.file, learning_rate=6e-4,
                                    batch_size=500, hidden_dim=10, doTransform=False)
        trainer.train()


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="run artificial experiment")

    parser.add_argument("--path", default="dumps/", help="Location to save the results", type=str)
    parser.add_argument("--copula_file", default="mi_copula.pickle", help="Copula file", type=str)
    parser.add_argument("--file", default="mi.pickle", help="Non copula file", type=str)
    parser.add_argument("--path_plots", default="plots/", help="Location to save the results", type=str)
    parser.add_argument("--train", default="True", type=str)
    parser.add_argument("--evaluate", default="True", type=str)
    args = parser.parse_args()

    # create folders if not existent
    if not os.path.exists(args.path):
        os.makedirs(args.path)

    if not os.path.exists(args.path_plots):
        os.makedirs(args.path_plots)

    # train artificial dataset
    if args.train == "True":
        logger.info("Train artificial dataset without copula")
        train_artificial_dataset(args, do_transform=False)

        logger.info("Train artificial dataset with copula")
        train_artificial_dataset(args, do_transform=True)

    # evaluate artificial dataset
    if args.evaluate == "True":
        logger.info("Evaluate artificial dataset")
        EvaluateArtificialDataset.evaluate_information_curve(args.path + args.copula_file, args.path + args.file,
                                                             args.path_plots)


if __name__ == '__main__':
    main()
