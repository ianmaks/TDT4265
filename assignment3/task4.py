import pathlib
import matplotlib.pyplot as plt
import utils
import torch
from torch import nn, float32
from torchvision.transforms import v2
from torchvision.models import resnet18
from dataloaders_task4 import load_cifar10
from trainer_task4 import Trainer_task4
from task2 import create_plots



class Model(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, 10)  # No need to apply softmax, as this is done in nn.CrossEntropyLoss
        
        for param in self.model.parameters(): # Freeze all parameters
            param.requires_grad = False
        for param in self.model.fc.parameters(): # Unfreeze the last fully-connected layer
            param.requires_grad = True
        for param in self.model.layer4.parameters(): # Unfreeze the last 5 convolutional layers
            param.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        return x
    


def main():
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result!
    utils.set_seed(0)
    print(f"Using device: {utils.get_device()}")
    epochs = 10
    batch_size = 32
    learning_rate = 5e-4
    early_stop_count = 4
    dataloaders = load_cifar10(batch_size)
    
    
    # dataloaders[0].dataset = extend_dataset(dataloaders[0].dataset)
    
    model = Model()
    trainer = Trainer_task4(
        batch_size, learning_rate, early_stop_count, epochs, model, dataloaders
    )
    trainer.train()
    
    create_plots(trainer, "task4")
    
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
