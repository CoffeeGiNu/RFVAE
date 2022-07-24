import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from utility import *
from model import *
from dataset import *


parser = argparse.ArgumentParser()

parser.add_argument('-ne', '--num_epochs', default=128, type=int)
parser.add_argument('-bs', '--batch_size', default=32, type=int)
parser.add_argument('-s', '--seed', default=42, type=int)
parser.add_argument('-is', '--in_image_size', default=(28, 28), type=tuple)
parser.add_argument('-ois', '--out_image_size', default=(28, 28), type=image_size)
parser.add_argument('-ic', '--in_channels', default=1, type=int)
parser.add_argument('-ehc', '--enc_hidden_channels', default=64, type=int)
parser.add_argument('-dl', '--dim_latent', default=10, type=int)
parser.add_argument('-dhf', '--dec_hidden_features', default=512, type=int)
parser.add_argument('-oc', '--out_channels', default=1, type=int)
parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float)
parser.add_argument('-ip', '--is_profiler', default=False, type=bool)
parser.add_argument('-es', '--is_early_stopping', default=True, type=bool)
parser.add_argument('-pm', '--path_model', default='./models/', type=str)

args = parser.parse_args()

NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
IN_CHANNELS = args.in_channels
ENC_HIDDEN_CHANNELS = args.enc_hidden_channels
DIM_LATENT = args.dim_latent
DEC_HIDDEN_FEATURES = args.dec_hidden_features
OUT_CHANNELS = args.out_channels
IN_IMAGE_SIZE = args.in_image_size
OUT_IMAGE_SIZE = args.out_image_size
SEED = args.seed
LEARNING_RATE = args.learning_rate
IS_PROFILE = args.is_profiler
IS_EARLY_STOPPING = args.is_early_stopping
PATH_MODEL = args.path_model


if __name__ == "__main__":
    log_dir = "./logs"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_test = load_mnist_dataset(
        split='test', preprocess_fn=cast_and_nomrmalise_images)
    
    model = ResolutionFreeVariationalAutoEncoder(
        in_channels=IN_CHANNELS,
        enc_hidden_channels=ENC_HIDDEN_CHANNELS,
        dim_latent=DIM_LATENT,
        dec_hidden_features=DEC_HIDDEN_FEATURES,
        out_channels=OUT_CHANNELS,
        out_size=OUT_IMAGE_SIZE,
        device=device,
    )
    model.load_state_dict(torch.load(PATH_MODEL))
    model.eval()
    model.to(device)

    for inputs in dataset_test:
        inputs = inputs['image']
        inputs = torch.from_numpy(np.array(inputs)).to(device)
        fig, axes = plt.subplots(2, 10, figsize=(20, 4))
        # for a in axes:
        #     a.set_xticks([])
        #     a.set_yticks([])

        results = model(inputs)
        y = results['outputs']
        # image_x = inputs.cpu().detach().numpy().reshape(-1, IN_CHANNELS, IN_IMAGE_SIZE[0], IN_IMAGE_SIZE[1])
        # image_y = y.cpu().detach().numpy().reshape(-1, OUT_CHANNELS, OUT_IMAGE_SIZE[0], OUT_IMAGE_SIZE[1])
        image_x = inputs.cpu().detach().numpy().reshape(-1, IN_IMAGE_SIZE[0], IN_IMAGE_SIZE[1])
        image_y = y.cpu().detach().numpy().reshape(-1, OUT_IMAGE_SIZE[0], OUT_IMAGE_SIZE[1])
        for j in range(BATCH_SIZE):
            axes[0][j].imshow(image_x[j], "gray")
            axes[0][j].set_xticks([])
            axes[0][j].set_yticks([])
            axes[1][j].imshow(image_y[j], "gray")
            axes[1][j].set_xticks([])
            axes[1][j].set_yticks([])
        fig.savefig(f"./figure/reconstracttion_z{DIM_LATENT}_({OUT_IMAGE_SIZE[0]}_{OUT_IMAGE_SIZE[1]}).png")
        plt.close()
        break