import os
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter

from train import *
from model import *
from utility import *
from dataset import *


parser = argparse.ArgumentParser()

parser.add_argument('-ne', '--num_epochs', default=128, type=int)
parser.add_argument('-bs', '--batch_size', default=32, type=int)
parser.add_argument('-s', '--seed', default=42, type=int)
parser.add_argument('-is', '--image_size', default=(28, 28), type=tuple)
parser.add_argument('-ois', '--out_image_size', default=(28, 28), type=image_size)
parser.add_argument('-ic', '--in_channels', default=1, type=int)
parser.add_argument('-edh', '--enc_hidden_channels', default=32, type=int)
parser.add_argument('-dl', '--dim_latent', default=2, type=int)
parser.add_argument('-dhf', '--dec_hidden_features', default=256, type=int)
parser.add_argument('-oc', '--out_channels', default=1, type=int)
parser.add_argument('-lr', '--learning_rate', default=2e-3, type=float)
parser.add_argument('-ip', '--is_profiler', default=False, type=bool)
parser.add_argument('-es', '--is_early_stopping', default=False, type=bool)
parser.add_argument('-pm', '--path_model', default='./models/checkpoint.pth', type=str)

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

if __name__ == "__main__":
    fix_seed(SEED)
    log_dir = "./logs"
    writer = SummaryWriter(log_dir)
    loss_fn = lambda lower_bound: -sum(lower_bound)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    os.makedirs(log_dir) if not os.path.exists(log_dir) else None
    
    dataset_train = load_mnist_dataset(
        batch_size=BATCH_SIZE, preprocess_fn=cast_and_nomrmalise_images)
    dataset_valid = load_mnist_dataset(
        batch_size=BATCH_SIZE, split='valid', preprocess_fn=cast_and_nomrmalise_images)
    
    model = ResolutionFreeVariationalAutoEncoder(
        in_channels=IN_CHANNELS,
        enc_hidden_channels=ENC_HIDDEN_CHANNELS,
        dim_latent=DIM_LATENT,
        dec_hidden_features=DEC_HIDDEN_FEATURES,
        out_channels=OUT_CHANNELS,
        out_size=OUT_IMAGE_SIZE,
        device=device,
    ).to(device)
    try:
        optimizer = torch.optim.RAdam(
            model.parameters(), 
            lr=LEARNING_RATE, 
            # weight_decay=DECAY, 
            eps=0.000
        )
    except:
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=LEARNING_RATE, 
            # weight_decay=DECAY, 
            eps=0.000
        )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    earlystopping = None
    if IS_EARLY_STOPPING:
        earlystopping = EarlyStopping(path=PATH_MODEL, patience=5)
    criterion = VAELoss()

    if IS_PROFILE:
        with torch.profiler.profile(
            # schedule=torch.profiler.schedule(
            #     wait=2,
            #     warmup=2,
            #     active=6,
            #     repeat=1),
            # use_cuda=(True if device=='cuda' else False),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
            record_shapes=True,
            with_stack=True
        ) as profiler:
            for e in range(NUM_EPOCHS):
                model = epoch_loop(model, dataset_train, optimizer, criterion, device, e, NUM_EPOCHS, BATCH_SIZE, is_train=True, profiler=profiler, writer=writer)
                model = epoch_loop(model, dataset_valid, optimizer, criterion, device, e, NUM_EPOCHS, BATCH_SIZE, is_train=False, earlystopping=earlystopping, profiler=profiler, writer=writer)
                if IS_EARLY_STOPPING:
                    if earlystopping.early_stop:
                        # writer.add_graph(model)
                        writer.close()
                        break
                scheduler.step()
            writer.close()
    else:
        for e in range(NUM_EPOCHS):
            model = epoch_loop(model, dataset_train, optimizer, criterion, device, e, NUM_EPOCHS, BATCH_SIZE, is_train=True, profiler=None, writer=writer)
            model = epoch_loop(model, dataset_valid, optimizer, criterion, device, e, NUM_EPOCHS, BATCH_SIZE, is_train=False, earlystopping=earlystopping, profiler=None, writer=writer)
            if IS_EARLY_STOPPING:
                if earlystopping.early_stop:
                    # writer.add_graph(model)
                    writer.close()
                    break
            scheduler.step()
        writer.close()