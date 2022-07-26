import torch
import argparse

from model import *
from utility import *


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
OUT_IMAGE_SIZE = args.out_image_size
SEED = args.seed
LEARNING_RATE = args.learning_rate
IS_PROFILE = args.is_profiler
IS_EARLY_STOPPING = args.is_early_stopping
PATH_MODEL = args.path_model


log_dir = "./logs"
seed = 42
fix_seed(seed)
device = 'cpu'
file_model = PATH_MODEL+"64_10_512/checkpoint_z10.pth"

model = ResolutionFreeVariationalAutoEncoder(
    in_channels=IN_CHANNELS,
    enc_hidden_channels=ENC_HIDDEN_CHANNELS,
    dim_latent=DIM_LATENT,
    dec_hidden_features=DEC_HIDDEN_FEATURES,
    out_channels=OUT_CHANNELS,
    out_size=OUT_IMAGE_SIZE,
    device=device,
)
model.load_state_dict(torch.load(file_model))
model.eval()

dummy_input = torch.randn(1, 1, 28, 28, device="cpu")
torch.onnx.export(model, dummy_input, "./models/model.onnx", input_names=["input"], output_names=["outputs", "mean", "logvar", "latent"], verbose=True)