import torch
from torch import nn
from data.dataloader import load_random_data, load_dummy_data, load_imagenet_data

from model.model import Model

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute pred and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        # Backprop
        optimizer.zero_grad() # Clear prev gradients
        loss.backward()       # Compute gradients
        optimizer.step()      # Update model params

        if batch % 5 == 0:
            # Print loss every 100 batches
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    model.eval() # Set model to evaluation
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    # Disable gradient calculation for testing
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == "__main__":
    # HPs
    learning_rate = 0.05
    batch_size = 16
    epochs = 3
    input_shape = (3, 224, 224)
    num_labels = 10
    num_samples = 1000

    # Dataloader
    dataloader = load_dummy_data(num_samples=num_samples, input_shape=input_shape, num_classes=num_labels, batch_size=batch_size)

    # Model, loss, optimizer
    model = Model(labels=num_labels)
    loss_fn = nn.NLLLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train(dataloader, model, loss_fn, optimizer)
        test(dataloader, model, loss_fn)
    print("Training done.")

    # Save the model
    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")