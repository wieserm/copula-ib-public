import logging
from matplotlib.backends.backend_pdf import PdfPages
from pylab import *
import pickle

logger = logging.getLogger("root")


class EvaluateArtificialDataset:
    """
    This class evaluates the artificial experiments.
    """

    @staticmethod
    def evaluate_information_curve(copula_file, file, path):
        """
        This method plots the information curves between copula and non-copula experiments.
        :param copula_file: the copula file
        :param file: the non-copula file
        """

        logger.info("Evaluate artificial information curve")

        # create pdf
        with PdfPages(path + "ic.pdf") as pdf:

            fig = plt.figure()

            # open copula file
            with open(copula_file, 'r') as f:
                copula_data = pickle.load(f)

                # create boxplots
                bp = plt.boxplot(copula_data[1], showfliers=False, positions=copula_data[2],
                                 labels=[round(elem, 1) for elem in copula_data[2]],
                                 patch_artist=True)
                for box in bp['boxes']:
                    # change outline color
                    box.set(color='#b20000', linewidth=1.25)
                    box.set(facecolor='#ff9999')

                ## change color and linewidth of the whiskers
                for whisker in bp['whiskers']:
                    whisker.set(color='#b20000', linewidth=1.25)
                    whisker.set(linestyle='-')

                for cap in bp['caps']:
                    cap.set(color='#b20000', linewidth=1.25)

                for median in bp['medians']:
                    median.set(color='#b20000', linewidth=1.25)

            # open non-copula file
            with open(file, 'r') as f:
                data = pickle.load(f)

                # create boxplots
                bp = plt.boxplot(data[1], showfliers=False, positions=data[2],
                                 labels=[round(elem, 1) for elem in data[2]], patch_artist=True)
                for box in bp['boxes']:
                    # change outline color
                    box.set(color='#FF9900', linewidth=1.25)
                    box.set(facecolor='#FFDEAD')

                    ## change color and linewidth of the whiskers
                for whisker in bp['whiskers']:
                    whisker.set(color='#FF9900', linewidth=1.25)
                    whisker.set(linestyle='-')

                for cap in bp['caps']:
                    cap.set(color='#FF9900', linewidth=1.25)

                for median in bp['medians']:
                    median.set(color='#FF9900', linewidth=1.25)

            # draw copula and non copula mean lines
            plt.plot(copula_data[2], copula_data[0],
                     color='#b20000', label=r'Information Curve (Copula)', linewidth=1.5)
            plt.plot(data[2], data[0], color='#FF9900',
                     label=r'Information Curve', linewidth=1.5, linestyle=':')

            # plot how many dimensions of the latent space are used (non-copula)
            strings = ["%.d" % number for number in data[4]]
            x_text = data[2]
            y_text = data[0]
            for i in range(len(x_text)):
                plt.text(x_text[i] - 0.2, y_text[i] - 1, strings[i], color='#FF9900')

            # plot how many dimensions of the latent space are used (copula)
            strings = ["%.d" % number for number in copula_data[4]]
            x_text = copula_data[2]
            y_text = copula_data[0]
            for i in range(len(x_text)):
                plt.text(x_text[i] - 0.2, y_text[i] + 1, strings[i], color='#b20000')

            # set legend
            plt.legend(loc="lower right")

            # set axis label
            plt.ylabel(r'$I(y,t)$')
            plt.xlabel(r'$I(x,t)$')

            ax = plt.gca()
            ax.set_axis_bgcolor('#f6f6f6')

            # set background
            ax = plt.gca()
            ax.set_axis_bgcolor('#f6f6f6')

            pdf.savefig(fig)
            plt.show()
            plt.close()
