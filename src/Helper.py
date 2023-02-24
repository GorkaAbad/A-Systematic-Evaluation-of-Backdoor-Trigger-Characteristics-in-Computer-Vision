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
