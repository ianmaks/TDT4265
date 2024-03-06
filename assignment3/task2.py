import pathlib
import matplotlib.pyplot as plt
import utils
import torch
from torch import nn, float32
from torchvision.transforms import v2
from dataloaders import load_cifar10
from trainer import Trainer


class ExampleModel(nn.Module):
    def __init__(self, image_channels, num_classes):
        """
        Is called when model is initialized.
        Args:
            image_channels. Number of color channels in image (3)
            num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        # TODO: Implement this function (Task  2a)
        
        
        # Hyperparameters
        num_filters = 32  # Set number of filters in first conv layer
        kernel_size = 3
        padding = 1
        stride = 1
        
        self.num_classes = num_classes
        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=num_filters,
                kernel_size=kernel_size,
                stride=stride,
                padding=1,
            ),
            nn.ReLU(),
            
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=64,
                kernel_size=5,
                stride=stride,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(
                    in_channels=64,
                    out_channels=128,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),    
             
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)       
        )
        # The output of feature_extractor will be [batch_size, num_filters, 4, 4]
        self.num_output_features = 4*4*256
        # Initialize our last fully connected layer
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.num_output_features, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
            # nn.Softmax()
        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        # TODO: Implement this function (Task  2a)
        batch_size = x.shape[0]
        out = self.feature_extractor(x)
        out = self.classifier(out)
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (
            batch_size,
            self.num_classes,
        ), f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out


def create_plots(trainer: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(
        trainer.train_history["loss"], label="Training loss", npoints_to_average=10
    )
    utils.plot_loss(trainer.validation_history["loss"], label="Validation loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(trainer.validation_history["accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()


def main():
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result!
    utils.set_seed(0)
    print(f"Using device: {utils.get_device()}")
    epochs = 10
    batch_size = 64
    learning_rate = 5e-2
    early_stop_count = 4
    dataloaders = load_cifar10(batch_size)
    
    # dataloaders[0].dataset = extend_dataset(dataloaders[0].dataset)
    
    model = ExampleModel(image_channels=3, num_classes=10)
    trainer = Trainer(
        batch_size, learning_rate, early_stop_count, epochs, model, dataloaders
    )
    trainer.train()
    
    create_plots(trainer, "task2")
    
    from trainer import compute_loss_and_accuracy
    # Check test accuracy
    test_loss, test_accuracy = compute_loss_and_accuracy(
        dataloaders[2], trainer.model, trainer.loss_criterion
    )
    print(f"Test Accuracy: {test_accuracy:.3f} "
          f"Test Loss: {test_loss:.3f}")
    
    #Check train accuracy
    train_loss, train_accuracy = compute_loss_and_accuracy(
        dataloaders[0], trainer.model, trainer.loss_criterion
    )
    print(f"Train Accuracy: {train_accuracy:.3f} "
          f"Train Loss: {train_loss:.3f}")
    
    # Check validation accuracy
    val_loss, val_accuracy = compute_loss_and_accuracy(
        dataloaders[1], trainer.model, trainer.loss_criterion
    )
    print(f"Validation Accuracy: {val_accuracy:.3f} "
          f"Validation Loss: {val_loss:.3f}")
    


if __name__ == "__main__":
    main()
