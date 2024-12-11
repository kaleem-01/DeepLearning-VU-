from data_rnn import load_imdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

(x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = load_imdb(final=False)


class RNN(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_classes, pad_idx):
        """
        Parameters:
        - vocab_size (int): Size of the vocabulary.
        - emb_dim (int): Dimensionality of the embeddings.
        - hidden_dim (int): Dimensionality of the hidden layer.
        - num_classes (int): Number of output classes.
        - pad_idx (int): Index for the padding token.
        """
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.linear1 = nn.Linear(emb_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        """
        Forward pass through the network.

        Parameters:
        - x (torch.Tensor): Input tensor of shape (batch, time).

        Returns:
        - output (torch.Tensor): Output tensor of shape (batch, num_classes).
        """
        # Step 1: Embedding layer
        x = self.embedding(x)  # Output shape: (batch, time, emb)
        
        # Step 2: Linear transformation
        x = self.linear1(x)  # Output shape: (batch, time, hidden)
        
        # Step 3: ReLU activation
        x = self.relu(x)  # Output shape: (batch, time, hidden)
        
        # Step 4: Global max pooling along the time dimension
        x = torch.max(x, dim=1)[0]  # Output shape: (batch, hidden)
        
        # Step 5: Linear projection to number of classes
        output = self.linear2(x)  # Output shape: (batch, num_classes)
        
        return output


# Data preparation
def prepare_data(x, y, w2i, batch_size):
    """
    Pads sequences and prepares data loaders.

    Parameters:
    - x (list of list of int): Input sequences.
    - y (list of int): Labels.
    - w2i (dict): Vocabulary mapping (used for padding index).
    - batch_size (int): Batch size.

    Returns:
    - DataLoader object for PyTorch.
    """
    pad_idx = w2i[".pad"]

    # Pad sequences to the maximum length in the batch
    max_len = max(len(seq) for seq in x)
    padded_x = [
        seq + [pad_idx] * (max_len - len(seq)) for seq in x
    ]
    padded_x = torch.tensor(padded_x, dtype=torch.long)
    labels = torch.tensor(y, dtype=torch.long)

    # Create DataLoader
    dataset = TensorDataset(padded_x, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


# Training loop
def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    """
    Trains the model for one epoch.

    Parameters:
    - model: PyTorch model.
    - dataloader: DataLoader for training data.
    - optimizer: Optimizer for training.
    - loss_fn: Loss function.
    - device: Device for computation (CPU/GPU).

    Returns:
    - Average training loss for the epoch.
    """
    model.train()
    total_loss = 0

    for batch_x, batch_y in dataloader:
        # Move data to device
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        # Forward pass
        outputs = model(batch_x)
        loss = loss_fn(outputs, batch_y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


# Validation loop
def evaluate(model, dataloader, device):
    """
    Evaluates the model on validation data.

    Parameters:
    - model: PyTorch model.
    - dataloader: DataLoader for validation data.
    - device: Device for computation (CPU/GPU).

    Returns:
    - Validation accuracy.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            # Move data to device
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Forward pass
            outputs = model(batch_x)
            predictions = torch.argmax(outputs, dim=1)

            # Compute accuracy
            correct += (predictions == batch_y).sum().item()
            total += batch_y.size(0)

    return correct / total




def main():
    # Hyperparameters
    batch_size = 248
    learning_rate = 0.001
    num_epochs = 10  # Train for at least one epoch

    # Prepare data loaders
    train_loader = prepare_data(x_train, y_train, w2i, batch_size)
    val_loader = prepare_data(x_val, y_val, w2i, batch_size)

    # Initialize model, optimizer, and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = len(w2i)
    emb_dim = 300
    hidden_dim = 300
    num_classes = numcls
    pad_idx = w2i[".pad"]

    model = RNN(vocab_size, emb_dim, hidden_dim, num_classes, pad_idx).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Training and validation
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_accuracy = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.2%}")


if __name__ == "__main__":
    main()

# Inspect the result
# print(padded_tensor)
# print(labels_tensor)


# idx = 0
# print([i2w[w] for w in x_train[idx]])
# print(batch_labels[idx])

# ● x_train A python list of lists of integers. Each integer represents a word. Sorted
# from short to long.
# ● y_train The corresponding class labels: 0 for positive, 1 for negative.
# ● x_val Test/validation data. Laid out the same as x_train.
# ● y_val Test/validation labels
# ● i2w A list of strings mapping the integers in the sequences to their original words.
# i2w[141] returns the string containing word 141.
# ● w2i A dictionary mapping the words to their indices. w2i['film'] returns the index
# for the word "film".

# print([i2w[w] for w in x_train[141]])   

# print(w2i[".pad"]) # 0

# print([i2w[w] for w in x_train[0]])   

# print(i2w[0])



