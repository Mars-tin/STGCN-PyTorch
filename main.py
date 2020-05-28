import os
import argparse
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from stgcn import STGCN
from GaussianCopula import CopulaLoss
from utils import generate_dataset, load_metr_la_data, get_normalized_adj


use_gpu = True
num_timesteps_input = 12
num_timesteps_output = 3

seed = 7
epochs = 1000
patience = 30
batch_size = 50

parser = argparse.ArgumentParser(description='STGCN')
parser.add_argument('--enable-cuda', action='store_true',
                    help='Enable CUDA')
args = parser.parse_args()
args.device = None
if args.enable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')
print(args.enable_cuda)
print(torch.cuda.is_available())
print(args.device)


def train_epoch(training_input, training_target, batch_size):
    """
    Trains one epoch with the given data.
    :param training_input: Training inputs of shape (num_samples, num_nodes,
    num_timesteps_train, num_features).
    :param training_target: Training targets of shape (num_samples, num_nodes,
    num_timesteps_predict).
    :param batch_size: Batch size to use during training.
    :return: Average loss for this epoch.
    """
    permutation = torch.randperm(training_input.shape[0])

    epoch_training_losses = []
    for i in range(0, training_input.shape[0], batch_size):
        net.train()
        optimizer.zero_grad()

        indices = permutation[i:i + batch_size]
        X_batch, y_batch = training_input[indices], training_target[indices]
        X_batch = X_batch.to(device=args.device)
        y_batch = y_batch.to(device=args.device)

        out = net(A_wave, X_batch)
        loss = loss_criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        epoch_training_losses.append(loss.detach().cpu().numpy())
    return sum(epoch_training_losses)/len(epoch_training_losses)


if __name__ == '__main__':
    torch.manual_seed(seed)

    A, X, means, stds = load_metr_la_data()

    split_line1 = int(X.shape[2] * 0.6)
    split_line2 = int(X.shape[2] * 0.8)

    train_original_data = X[:, :, :split_line1]
    val_original_data = X[:, :, split_line1:split_line2]
    test_original_data = X[:, :, split_line2:]

    training_input, training_target = generate_dataset(train_original_data,
                                                       num_timesteps_input=num_timesteps_input,
                                                       num_timesteps_output=num_timesteps_output)
    val_input, val_target = generate_dataset(val_original_data,
                                             num_timesteps_input=num_timesteps_input,
                                             num_timesteps_output=num_timesteps_output)
    test_input, test_target = generate_dataset(test_original_data,
                                               num_timesteps_input=num_timesteps_input,
                                               num_timesteps_output=num_timesteps_output)

    A_wave = get_normalized_adj(A)
    A_wave = torch.from_numpy(A_wave)

    A_wave = A_wave.to(device=args.device)

    net = STGCN(A_wave.shape[0],
                training_input.shape[3],
                num_timesteps_input,
                num_timesteps_output).to(device=args.device)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_criterion = CopulaLoss()

    training_losses = []
    validation_losses = []
    validation_maes = []

    plt.ion()
    fig, axes = plt.subplots(1, figsize=(10, 5))
    plt.suptitle('STGCN Training')
    axes.set_xlabel('Epoch')
    axes.set_ylabel('Loss')

    rem = patience
    idx = 0
    best_loss = float('inf')

    for epoch in range(epochs):
        # Early stop
        if patience < 0:
            break

        loss = train_epoch(training_input, training_target,
                           batch_size=batch_size)
        training_losses.append(loss)

        # Run validation
        with torch.no_grad():
            net.eval()
            val_input = val_input.to(device=args.device)
            val_target = val_target.to(device=args.device)
            permutation = torch.randperm(training_input.shape[0])

            val_loss_batch = []
            mae_batch = []
            for i in range(0, training_input.shape[0], batch_size):
                indices = permutation[i:i + batch_size]
                X_batch, y_batch = training_input[indices], training_target[indices]
                X_batch = X_batch.to(device=args.device)
                y_batch = y_batch.to(device=args.device)

                out = net(A_wave, X_batch)
                val_loss = loss_criterion(out, y_batch).to(device="cpu")
                val_loss_batch.append(val_loss.detach().numpy().item())

                out_unnormalized = out.detach().cpu().numpy()*stds[0]+means[0]
                target_unnormalized = y_batch.detach().cpu().numpy()*stds[0]+means[0]
                mae = np.mean(np.absolute(out_unnormalized - target_unnormalized))
                mae_batch.append(mae)

            validation_losses.append(np.mean(val_loss_batch))
            validation_maes.append(np.mean(mae_batch))

            out = None
            val_input = val_input.to(device="cpu")
            val_target = val_target.to(device="cpu")

        print("\nEpoch: {}".format(epoch))
        print("Training loss: {}".format(training_losses[-1]))
        print("Validation loss: {}".format(validation_losses[-1]))
        print("Validation MAE: {}".format(validation_maes[-1]))

        checkpoint_path = "checkpoints/"
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        with open("checkpoints/losses.pk", "wb") as fd:
            pk.dump((training_losses, validation_losses, validation_maes), fd)

        valid_loss = validation_losses[-1]
        if valid_loss < best_loss:
            rem = patience
            best_loss = valid_loss
            idx = epoch

        rem -= 1

    print("\nThe MSE loss on test dataset is:", validation_losses[idx])
    print("The MAE on test dataset is:", validation_maes[idx])
    print("This is obtained in epoch", idx)

    plt.plot(training_losses, label="training loss")
    plt.plot(validation_losses, label="validation loss")
    plt.plot(validation_maes, label="validation MAE")
    plt.legend()
    fig.savefig("STGCN_training_{}".format(seed), dpi=200)
    plt.show()
