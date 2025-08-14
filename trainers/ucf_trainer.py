import time
import wandb
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from huggingface_hub import upload_file, create_repo
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class UCFTrainingConfig:
    """
    Holds configuration parameters for training a UCF action recognition model.

    This dataclass includes all hyperparameters, DataLoaders, and settings
    required for training, validation, and testing.

    - experiment_name: Name of the experiment for logging and saving.
    - criterion: Loss function used during training.
    - optimizer: Name of the optimizer to use (default: 'Adam').
    - lr: Learning rate for the optimizer (default: 0.001).
    - epochs: Number of training epochs (default: 20).
    - patience: Number of epochs to wait for early stopping (default: 3).
    - train_loader: PyTorch DataLoader for the training dataset.
    - val_loader: PyTorch DataLoader for the validation dataset.
    - test_loader: PyTorch DataLoader for the test dataset.
    - main_save_path: Directory path to save model checkpoints (default: 'pretrain').
    """

    experiment_name: str
    criterion: nn.Module
    optimizer: str = "Adam"
    lr: float = 0.001
    epochs: int = 20
    patience = 3
    train_loader: DataLoader = field(default_factory=DataLoader)
    val_loader: DataLoader = field(default_factory=DataLoader)
    test_loader: DataLoader = field(default_factory=DataLoader)
    main_save_path: str = "pretrain"


class UCFTrainer:
    """
    Trainer class for managing UCF model training, validation, testing, and saving.

    This class handles the training loop, validation evaluation, early stopping,
    test evaluation, logging metrics to Weights & Biases, and saving the trained
    model locally or to Hugging Face Hub.
    """

    def __init__(self, model: nn.Module, config: UCFTrainingConfig):
        """
        Initializes the UCFTrainer.

        :param model: PyTorch model to train.
        :param config: Configuration object containing hyperparameters, DataLoaders,
                    and experiment settings.
        """
        self.model = model
        self.config = config

        self.optimizer: optim.Optimizer = {
            "Adadelta": optim.Adadelta(self.model.parameters(), lr=self.config.lr),
            "Adafactor": optim.Adafactor(self.model.parameters(), lr=self.config.lr),
            "Adagrad": optim.Adagrad(self.model.parameters(), lr=self.config.lr),
            "Adam": optim.Adam(self.model.parameters(), lr=self.config.lr),
            "AdamW": optim.AdamW(self.model.parameters(), lr=self.config.lr),
            "SparseAdam": optim.SparseAdam(self.model.parameters(), lr=self.config.lr),
            "Adamax": optim.Adamax(self.model.parameters(), lr=self.config.lr),
            "ASGD": optim.ASGD(self.model.parameters(), lr=self.config.lr),
            "LBFGS": optim.LBFGS(self.model.parameters(), lr=self.config.lr),
            "NAdam": optim.NAdam(self.model.parameters(), lr=self.config.lr),
            "RAdam": optim.RAdam(self.model.parameters(), lr=self.config.lr),
            "RMSprop": optim.RMSprop(self.model.parameters(), lr=self.config.lr),
            "Rprop": optim.Rprop(self.model.parameters(), lr=self.config.lr),
            "SGD": optim.SGD(self.model.parameters(), lr=self.config.lr),
            "SparseAdam": optim.SparseAdam(self.model.parameters(), lr=self.config.lr),
        }[self.config.optimizer]

        self.run = wandb.init(
            project="smart-surveillance-system",
            name=self.config.experiment_name,
            config={
                "criterion": self.config.criterion._get_name(),
                "optimizer": self.config.optimizer,
                "lr": self.config.lr,
                "epochs": self.config.epochs,
                "strategy": model.strategy,
                "unfreeze_number": model.unfreeze_number,
            },
        )

        self.is_train = False

    @staticmethod
    def __compute_metrics(y_true, y_pred):
        """
        Computes evaluation metrics (accuracy, precision, recall, F1).

        :param y_true: Ground truth labels (list or numpy array).
        :param y_pred: Predicted labels (list or numpy array).
        :return: Dictionary with keys: 'accuracy', 'precision', 'recall', 'f1'.
        """
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }

    def train(self):
        """
        Trains the UCF model with the specified configuration.

        Steps:
        1. Iterates over the number of epochs.
        2. Performs training batch by batch.
        3. Computes and logs batch-level metrics to wandb.
        4. Validates the model on the validation set after each epoch.
        5. Logs epoch-level metrics and saves the best model checkpoint.
        6. Stops early if validation loss does not improve after 'patience' epochs.

        :raises RuntimeError: If an unexpected error occurs during training.
        """
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config.epochs):
            self.model.train()
            start_time = time.time()
            total_loss = 0
            all_preds, all_labels = [], []

            progress_bar = tqdm(
                enumerate(self.config.train_loader, start=1),
                total=len(self.config.train_loader),
                desc=f"Epoch {epoch+1}/{self.config.epochs}",
                unit="batch",
            )

            for batch_idx, (frames, labels) in progress_bar:
                self.optimizer.zero_grad()

                frames = frames.to(device)
                labels = labels.to(device)

                outputs = self.model(frames)
                loss = self.config.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                preds = outputs.argmax(dim=1).cpu().numpy()
                batch_labels = labels.cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch_labels)

                batch_metrics = self.__compute_metrics(batch_labels, preds)

                self.run.log(
                    {
                        "train_batch_loss": loss.item(),
                        "train_batch_accuracy": batch_metrics["accuracy"],
                        "train_batch_precision": batch_metrics["precision"],
                        "train_batch_recall": batch_metrics["recall"],
                        "train_batch_f1": batch_metrics["f1"],
                        "epoch": epoch + 1,
                    }
                )

                avg_loss = total_loss / batch_idx
                progress_bar.set_postfix(
                    {"batch_loss": f"{loss.item():.4f}", "avg_loss": f"{avg_loss:.4f}"}
                )

            train_metrics = self.__compute_metrics(all_labels, all_preds)
            epoch_time = time.time() - start_time

            self.model.eval()
            val_loss = 0
            val_preds, val_labels_list = [], []

            with torch.no_grad():
                val_bar = tqdm(
                    enumerate(self.config.val_loader, start=1),
                    desc="Validating",
                    unit="batch",
                    total=len(self.config.val_loader),
                    ncols=80,
                )

                for batch_idx, (frames, labels) in val_bar:
                    frames = frames.to(device)
                    labels = labels.to(device)

                    outputs = self.model(frames)
                    loss = self.config.criterion(outputs, labels)
                    val_loss += loss.item()

                    preds = outputs.argmax(dim=1).cpu().numpy()
                    batch_labels = labels.cpu().numpy()
                    val_preds.extend(preds)
                    val_labels_list.extend(batch_labels)

                    val_batch_metrics = self.__compute_metrics(batch_labels, preds)

                    self.run.log(
                        {
                            "val_batch_loss": loss.item(),
                            "val_batch_accuracy": val_batch_metrics["accuracy"],
                            "val_batch_precision": val_batch_metrics["precision"],
                            "val_batch_recall": val_batch_metrics["recall"],
                            "val_batch_f1": val_batch_metrics["f1"],
                            "epoch": epoch + 1,
                        }
                    )

            val_loss /= len(self.config.val_loader)
            val_metrics = self.__compute_metrics(val_labels_list, val_preds)

            print(
                f"Epoch [{epoch+1}/{self.config.epochs}] "
                f"Train Loss: {total_loss/len(self.config.train_loader):.4f} | Train Acc: {train_metrics['accuracy']:.4f} | Train Prec: {train_metrics['precision']:.4f} | Train Rec: {train_metrics['recall']:.4f} | Train F1: {train_metrics['f1']:.4f} "
                f"| Val Loss: {val_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | Val Prec: {val_metrics['precision']:.4f} | Val Rec: {val_metrics['recall']:.4f} | Val F1: {val_metrics['f1']:.4f} "
                f"| Time: {epoch_time:.2f}s"
            )

            self.run.log(
                {
                    "train_epoch_loss": total_loss / len(self.config.train_loader),
                    "train_epoch_accuracy": train_metrics["accuracy"],
                    "train_epoch_precision": train_metrics["precision"],
                    "train_epoch_recall": train_metrics["recall"],
                    "train_epoch_f1": train_metrics["f1"],
                    "val_epoch_loss": val_loss,
                    "val_epoch_accuracy": val_metrics["accuracy"],
                    "val_epoch_precision": val_metrics["precision"],
                    "val_epoch_recall": val_metrics["recall"],
                    "val_epoch_f1": val_metrics["f1"],
                    "epoch_time": epoch_time,
                    "epoch": epoch + 1,
                }
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(
                    self.model.state_dict(),
                    f"{self.config.main_save_path}/{self.config.experiment_name}",
                )
                self.run.summary["best_val_loss"] = best_val_loss
                self.run.summary["best_val_accuracy"] = val_metrics["accuracy"]
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break

        self.is_train = True

    def evaluate(self):
        """
        Evaluates the trained model on the test dataset.

        Steps:
        1. Checks if the model has been trained.
        2. Iterates over the test DataLoader.
        3. Computes evaluation metrics: accuracy, precision, recall, F1.
        4. Logs test metrics to wandb and finalizes the wandb run.

        :return: Dictionary containing test metrics: 'accuracy', 'precision', 'recall', 'f1'.
        :raises RuntimeError: If the model has not been trained yet.
        """
        if not self.is_train:
            raise RuntimeError(
                "The model has not been trained yet. Please call the train() method first."
            )

        self.model.eval()

        test_preds, test_labels_list = [], []

        test_bar = tqdm(
            self.config.test_loader,
            desc="Testing",
            total=len(self.config.test_loader),
            ncols=80,
        )

        for frames, labels in test_bar:
            frames = frames.to(device)
            labels = labels.to(device)

            outputs = self.model(frames)
            preds = outputs.argmax(dim=1).cpu().numpy()
            batch_labels = labels.cpu().numpy()
            test_preds.extend(preds)
            test_labels_list.extend(batch_labels)

        test_metrics = self.__compute_metrics(test_labels_list, test_preds)

        self.run.log(
            {
                "test_accuracy": test_metrics["accuracy"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f1": test_metrics["f1"],
            }
        )

        self.run.finish()

        return test_metrics

    def save(self, repo_id):
        """
        Saves the trained model locally and uploads it to Hugging Face Hub.

        :param repo_id: Hugging Face repository ID where the model will be uploaded.
        :return: URL of the commit for the uploaded model on Hugging Face Hub.
        :raises RuntimeError: If the model has not been trained yet.
        """
        if not self.is_train:
            raise RuntimeError(
                "The model has not been trained yet. Please call the train() method first."
            )

        repo = create_repo(repo_id, repo_type="model", private=False)

        commit = upload_file(
            path_or_fileobj=f"{self.config.main_save_path}/{self.config.experiment_name}",
            path_in_repo="ucf_model.pth",
            repo_id=repo.repo_id,
            repo_type="model",
        )

        return commit.commit_url
