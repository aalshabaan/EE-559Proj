import torch
import math

def generate_disc_data(nb_points=1000):
    """
    Generates a set of 1000 2d points sampled uniformly in [0, 1], each with a label 0
    if outside the disk centered at (0.5, 0.5) of radius 1/sqrt(2*pi), and 1 inside.
            Parameters:
                    nb_points (int): Number of samples
            Returns:
                    input_ (torch.FloatTensor): Input data
                    target_ (torch.IntTensor): Target data

    """
    input_ = torch.empty(nb_points, 2).uniform_(0, 1)
    target_ = ((input_-0.5).pow(2).sum(dim=1)<=1/(2*math.pi)).int()
    return input_, target_