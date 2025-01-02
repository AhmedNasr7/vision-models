import os 
from loguru import logger
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.model_utils import accuracy


class Trainer:
    def __init__(self, model, criterion, optimizer, train_loader, val_loader, save_path, device="cuda", save_period=5):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        if not os.path.exists(save_path):
          os.makedirs(save_path, exist_ok=True)
            
        self.save_path = save_path
        self.save_period = save_period

        if device == "cuda" and torch.cuda.is_available():
            self.device = torch.device(device)
        elif device == "mps" and torch.backends.mps.is_available():
            self.device = torch.device(device)
        else:
          self.device = torch.device("cpu")
          logger.warning("Device not available. Using CPU.")

        self.model.to(self.device)
        # Dictionary to store all metrics
        self.history = {
            "epoch": [],
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }


    def forward(self, data, labels):
        outputs = self.model(data)
        loss = self.criterion(outputs, labels)
        return outputs, loss

    def training_step(self, data, labels):
        self.model.train()
        self.optimizer.zero_grad()
        outputs, loss = self.forward(data, labels)
        loss.backward()
        self.optimizer.step()
        return outputs.detach(), loss.detach()

    def evaluation_step(self, data, labels):
        self.model.eval()
        with torch.no_grad():
            outputs, loss = self.forward(data, labels)
        acc = accuracy(outputs, labels) * 100
        return outputs.detach(), loss.detach(), acc

    # def training_loop
    
    def train(self, epochs, start_epoch=0):
        for epoch in range(start_epoch, epochs):
            epoch_metrics = {"train_loss": 0.0, "train_accuracy": 0.0,
                             "val_loss": 0.0, "val_accuracy": 0.0}
            num_batches = len(self.train_loader)
            num_val_batches = len(self.val_loader)

            with tqdm(total=num_batches, desc=f"Training Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
                for i, batch in enumerate(self.train_loader):

                    # Training step
                    images, labels = batch
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs, loss = self.training_step(images, labels)
                    acc = accuracy(outputs, labels) * 100
                    epoch_metrics["train_loss"] += loss.item()
                    epoch_metrics["train_accuracy"] += acc

                    # Update progress bar
                    _train_loss = epoch_metrics["train_loss"] / (i+1)
                    _train_acc = epoch_metrics["train_accuracy"] / (i+1)

                    pbar.set_postfix({
                        "Training Loss": f"{_train_loss:.4f}",
                        "Training Accuracy": f"{_train_acc:.2f}%",
                        "Batch": f"{i+1}/{len(self.train_loader)}"
                    })

                    pbar.update(1)

            with tqdm(total=num_val_batches, desc=f"Validation Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
                for i, val_batch in enumerate(self.val_loader):

                    # Validation step 
                    images, labels = val_batch
                    images, labels = images.to(self.device), labels.to(self.device)
                    _, eval_loss, eval_acc = self.evaluation_step(images, labels)
                    epoch_metrics["val_loss"] += eval_loss.item()
                    epoch_metrics["val_accuracy"] += eval_acc

             
                    _val_loss = epoch_metrics["val_loss"] / (i+1)
                    _val_acc = epoch_metrics["val_accuracy"] / (i+1)

                    pbar.set_postfix({
                        "Training Loss": f"{_train_loss:.4f}",
                        "Training Accuracy": f"{_train_acc:.2f}%",
                        "Validation Loss": f"{_val_loss:.4f}",
                        "Validation Accuracy": f"{_val_acc:.2f}%",
                        "Batch": f"{i+1}/{len(self.train_loader)}"
                    })
                    pbar.update(1)

                    # Free CUDA memory
                    torch.cuda.empty_cache()

            # Calculate epoch averages
            for key in epoch_metrics:
                if key.startswith("val_"):
                    epoch_metrics[key] /= len(self.val_loader)
                else:
                    epoch_metrics[key] /= num_batches
                    
                self.history[key].append(epoch_metrics[key])

            self.history["epoch"] = epoch
            
            if self.save_period > 0 and epoch % self.save_period == 0:
                logger.info("Saving Model")

                state_file_path = os.path.join(self.save_path, f"model_epoch_{epoch}.pth")
                self.save_state_dict(state_file_path)
                logger.info("Model Saved")
            else:
                state_file_path = os.path.join(self.save_path, f"last_model.pth")
                self.save_state_dict(state_file_path)
                logger.info("Model Saved")

            best_accuracy, best_epoch = self.track_best_metric("accuracy")
            logger.info(f"Best val accuracy: {best_accuracy:.2f}% at Epoch {best_epoch}")
            if epoch == best_epoch:
                logger.info("Saving Best Model")
                state_file_path = os.path.join(self.save_path, f"best_model.pth")
                self.save_state_dict(state_file_path)


        logger.info("Finished Training")
        self.plot_metrics()

    def save_state_dict(self, path):

        state_dict = {
            "epoch": self.history["epoch"],
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history
        }

        torch.save(state_dict, path)

    def load_state_dict(self, path):
        self.model.load_state_dict(torch.load(path)["model_state_dict"])

    def resume_training(self, epochs, path="", resume_best=True):

        if resume_best:
            path = os.path.join(self.save_path, f"best_model.pth")

        logger.info(f"Resuming Training from {path}")
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint["history"]
        self.model.to(self.device)

        start_epoch = checkpoint["epoch"] - 1

        self.train(start_epoch=start_epoch, epochs=epochs-start_epoch)


    def track_best_metric(self, metric):
        if metric == "loss":
            best_metric = min(self.history["val_loss"])
            best_epoch = self.history["val_loss"].index(best_metric)
        elif metric == "accuracy":
            best_metric = max(self.history["val_accuracy"])
            best_epoch = self.history["val_accuracy"].index(best_metric)

        else:
            raise ValueError("Invalid metric. Must be 'loss' or 'accuracy'.")

        return best_metric, best_epoch


    def plot_metrics(self):
        epochs = range(1, len(self.history["train_loss"]) + 1)

        # Plot loss
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, self.history["train_loss"], label="Training Loss")
        plt.plot(epochs, self.history["val_loss"], label="Validation Loss")
        plt.title("Loss Over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True) 
        plt.show()

        # Plot accuracy
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, self.history["train_accuracy"], label="Training Accuracy")
        plt.plot(epochs, self.history["val_accuracy"], label="Validation Accuracy")
        plt.title("Accuracy Over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.grid(True)
        plt.show()
