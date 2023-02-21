import torch
import matplotlib.pyplot as plt
import os


class Helper():
    save_path = None

    def __init__(self, args):
        self.save_path = args.save_path

    def plot_image(self, data, path=None):
        if path is None:
            path = self.save_path

        plt.imsave(path, data)

    def save_results(self, trainer, path=None):
        if path is None:
            path = self.save_path

        # Save the results in a csv file
        if not os.path.exists(path):
            os.makedirs(path)

        # Create a folder per seed
        path = os.path.join(path, str(trainer.seed))

        if not os.path.exists(path):
            os.makedirs(path)

        # Create the csv file
        path = os.path.join(path, 'results.csv')

        # Write the results to the csv file
