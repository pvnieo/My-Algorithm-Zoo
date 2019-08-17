# 3p
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.interpolate import UnivariateSpline
import torch


class LRFinder:
    """Implements Leslie N. Smith's Learning rate finder method. Use `.find()` method to search for the best learning rate.
       use `.plot()` method to plot the loss
    """

    def __init__(self, model, optimizer, criterion, cuda=True):
        """
        Arguments:
            model {torch.nn.Module} -- Used model
            optimizer {torch.optim.Optimizer} -- Used optimizer
            criterion {torch.nn.Module)} -- Loss function

        Keyword Arguments:
            cuda {bool} -- Use cuda if available (default: {True})
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = torch.device('cuda') if (cuda and torch.cuda.is_available()) else torch.device('cpu')
        self.history = {"loss": [], "lr": []}

        # set device
        self.model.to(self.device)
        self.criterion.to(self.device)

    def find(self, train_loader, val_loader=None, num_iter=100, init_value=1e-6, final_value=10., div_th=5, beta=0.98):
        """Performs the learning rate range test.

        Arguments:
            train_loader {torch.utils.data.DataLoader} -- Training set data loader

        Keyword Arguments:
            val_loader {torch.utils.data.DataLoader} -- Validation set dataloader. If None, range test will be performed only with train_loader (default: {None})
            num_iter {int} -- Maximum number of iteration. Determines the discretisation of the interval (default: {100})
            init_value {float} -- Minimun learning rate to start with. (default: {1e-6})
            final_value {float} -- Maximum learnig rate before stopping the range test (default: {10.})
            div_th {int} -- Stop the range test if the loss attains div_th * min_loss (default: {5})
            beta {float} -- Parameter used to smooth the loss. must be in [0, 1) (default: {0.98})
        """
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
        """train the model for one mini-batch if phase==train, perform a validation step otherwise

        Arguments:
            inputs {torch.Tensor} -- Input data
            labels {torch.Tensor} -- Labels of the input data
            val_loader {torch.utils.data.DataLoader} -- Validation set dataloader.

        Keyword Arguments:
            phase {str} -- Either `train` or `val` (default: {"train"})

        Returns:
            {float} -- loss obtained
        """

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
        """Plot the learning rate range test.

        Keyword Arguments:
            skip_start {int} -- number of batches to trim from the start (default: {10})
            skip_end {int} -- number of batches to trim from the end (default: {5})
            log_lr {bool} -- True to plot the learning rate in a logarithmic scale (default: {True})
            smooth {bool} -- True to smooth the loss function using UnivariateSpline smoother (default: {True})
            save_plot_to {[type]} -- Path to where to save the figure. None to disable saving. (default: {None})
        """
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
