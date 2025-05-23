import os
from meegnet.network import Model
import logging
from torch.utils.data import DataLoader

from data_loader import CustomMEGDataset

LOG = logging.getLogger(__name__)

class CustomMEGModel(Model):
    """
    Custom model class for MEEGNet.
    Inherits from the base Model class and accomodates for training and test set already split.
    """

    def train(
        self,
        dataset: CustomMEGDataset,
        batch_size: int = 1,
        patience: int = 10,
        max_epoch: int = None,
        model_path: str = None,
        early_stop: str = "loss",
        num_workers: int = 4,
        continue_training: bool = False,
        verbose: int = 3,
    ) -> None:
        """
        Train the model on the provided dataset.

        Parameters
        ----------
        dataset
            Dataset to train on.
        batch_size : int, optional
            Batch size for training. Defaults to 128.
        patience : int, optional
            Patience for early stopping. Defaults to 10.
        max_epoch : int, optional
            Maximum number of epochs to train. Defaults to None.
        model_path : str, optional
            Path to save the model. Defaults to None.
        num_workers : int, optional
            Number of workers for data loading. Defaults to 4.
        continue_training : bool, optional
            Wether to pick up training from last checkpoint. By default False. Use when
            training stopped abbruptly because of an error, or to re-train or fine-tune
            the model.
        verbose : int, optional
            The verbosity level. By default 3.
            Values under 3 will display less detailed progress during training.
            Values under 2 will not display any update on performance epoch per epoch during training.

        Notes
        -----
        This method trains the model using the provided dataset and hyperparameters.
        It uses early stopping based on the validation loss and saves the model
        periodically.
        """
        assert len(dataset.data) > 0, "Dataset is empty."
        # Check dataset compatibility
        assert (
            dataset.data[0].shape == self.input_size
        ), "Dataset sample size must match network input size."
        assert early_stop in (
            "loss",
            "accuracy",
        ), f"{early_stop} is not a valid early_stop option."

        # Setting model_path
        self.tracker.set_model_path(model_path)
        # Set training mode and batch size
        self.net.train()
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Create data loaders
        LOG.info("Creating DataLoaders...")
        train_index, valid_index, _ = dataset.split_data()
        trainloader = DataLoader(
            dataset.torchDataset(train_index),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True,
        )
        validloader = DataLoader(
            dataset.torchDataset(valid_index),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True,
        )

        # Log training configuration
        LOG.info("Starting Training with:")
        LOG.info(f"Batch size: {batch_size}")
        LOG.info(f"Learning rate: {self.lr}")
        LOG.info(f"Patience: {patience}")
        if max_epoch is not None:
            LOG.info(f"Maximum Epoch: {max_epoch}")

        epoch = 0
        if continue_training:
            epoch = self.tracker.best["epoch"] + 1
            self.tracker.patience_state = 0

        while self.tracker.patience_state < patience and (
            max_epoch is None or epoch < max_epoch
        ):
            self.train_epoch(epoch, trainloader, verbose=verbose)
            train_loss, train_acc = self.evaluate(trainloader)
            valid_loss, valid_acc = self.evaluate(validloader)
            self.tracker.update(
                epoch,
                train_loss,
                train_acc,
                valid_loss,
                valid_acc,
                self.net,
                self.optimizer,
                early_stop=early_stop,
            )
            if verbose >= 2:
                LOG.info(f"Epoch: {epoch}")
                LOG.info(f" [LOSS] TRAIN {train_loss:.4f} / VALID {valid_loss:.4f}")
                LOG.info(f" [ACC] TRAIN {100*train_acc:.2f}% / VALID {100*valid_acc:.2f}%")
            epoch += 1
    
    def test(self, dataset: CustomMEGDataset):
        test_loader = DataLoader(
            dataset= dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )
        test_loss, test_acc = self.evaluate(test_loader)
        LOG.info(f" [LOSS] TEST {test_loss}")
        LOG.info(f" [ACC] TEST {test_acc}")
        return test_loss, test_acc
    