# 3p
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.interpolate import UnivariateSpline
import torch


class LRFinder:
    def __init__(self, model, optimizer, criterion, cuda=True):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = torch.device('cuda') if (cuda and torch.cuda.is_available()) else torch.device('cpu')
        self.history = {"loss": [], "lr": []}

        # set device
        self.model.to(self.device)
        self.criterion.to(self.device)

    def find(self, train_loader, val_loader=None, num_iter=100, init_value=1e-6, final_value=10., div_th=5, beta=0.98):
        best_loss = float("inf")
        lr = init_value
        self.optimizer.param_groups[0]['lr'] = lr
        lr_update = (final_value / init_value) ** (1/num_iter)  # we use an exponential step mode
        avg_loss = 0

        # iterate over training data
        iterator = iter(train_loader)
        for iteration in tqdm(range(num_iter)):
            # Get a new set of inputs and labels
            try:
                inputs, labels = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                inputs, labels = next(iterator)

            # train model using batch
            if val_loader is None:
                loss = self._train_val_model(inputs, labels, val_loader)
            else:
                loss = self._train_val_model(inputs, labels, val_loader, phase="val")

            # Update the lr for the next step
            self.history["lr"].append(lr)
            lr *= lr_update
            self.optimizer.param_groups[0]['lr'] = lr

            # smooth loss and check for divergence
            avg_loss = beta * avg_loss + (1-beta) * loss
            smoothed_loss = avg_loss / (1 - beta**(iteration+1))
            self.history["loss"].append(smoothed_loss)
            if smoothed_loss > div_th * best_loss:
                break
            elif smoothed_loss < best_loss:
                best_loss = smoothed_loss

        print("LR Finder is complete. See the graph using `.plot()` method.")

    def _train_val_model(self, inputs, labels, val_loader, phase="train"):

        if phase == 'train':
            self.model.train()  # Set model to training mode
        else:
            self.model.eval()   # Set model to evaluate mode

        running_loss = 0.0

        # Iterate over data.
        dataloader = [(inputs, labels)] if phase == 'train' else val_loader
        for inputs, labels in dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                outputs = self.model(inputs)
                predictions = torch.max(outputs, 1)[1]
                loss = self.criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    self.optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = (running_loss / inputs.size(0)) if phase == 'train' else (running_loss / len(dataloader.dataset))
        return epoch_loss

    def plot(self, skip_start=10, skip_end=5, log_lr=True, smooth=True, save_plot_to=None):
        assert (skip_start >= 0 and skip_end >= 0), "skip_start and skip_end must be>=0!"

        lrs = self.history["lr"][skip_start:-skip_end] if skip_end > 0 else self.history["lr"][skip_start:]
        losses = self.history["loss"][skip_start:-skip_end] if skip_end > 0 else self.history["loss"][skip_start:]

        if smooth:
            spl = UnivariateSpline(lrs, losses)
            losses = spl(lrs)

        # get minimum lr over loss and gradient
        mg = (np.gradient(np.array(losses))).argmin()
        ml = np.argmin(losses)
        print(f"Min numerical gradient: {lrs[mg]}")
        print(f"Min loss: {lrs[ml]}")

        # Plot loss as a function of the learning rate
        plt.plot(lrs, losses)
        plt.plot(lrs[mg], losses[mg], markersize=10, marker='o', color='red')
        plt.plot(lrs[ml], losses[ml], markersize=10, marker='x', color='green')
        plt.legend(["Loss", "Min numerical gradient", "Min loss"])
        if log_lr:
            plt.xscale("log")
        plt.xlabel("Learning rate")
        plt.ylabel("Loss")
        if save_plot_to is not None:
            plt.savefig(save_plot_to)
        plt.show()
