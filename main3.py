import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Define the Elman RNN
# ---------------------------------------------------------------------------
class Elman(nn.Module):
    def __init__(self, insize=300, outsize=300, hsize=300):
        super(Elman, self).__init__()
        self.lin1 = nn.Linear(insize + hsize, hsize)
        self.lin2 = nn.Linear(hsize, outsize)
        
    def forward(self, x, hidden=None):
        b, t, e = x.size()
        if hidden is None:
            hidden = torch.zeros(b, e, dtype=torch.float, device=x.device) 

        outs = []
        for i in range(t):
            inp = torch.cat([x[:, i, :], hidden], dim=1)
            hidden = F.relu(self.lin1(inp))
            out = self.lin2(hidden)
            outs.append(out[:, None, :])

        return torch.cat(outs, dim=1), hidden

# ---------------------------------------------------------------------------
# Load Data (Assuming a function `load_imdb` is available)
# ---------------------------------------------------------------------------
# Please ensure you have the load_imdb function properly defined in data_rnn.py
# Here, we assume it returns (x_train, y_train), (x_val, y_val), (i2w, w2i), numcls
from data_rnn import load_imdb
(x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = load_imdb(final=False)

# ---------------------------------------------------------------------------
# Define the Model
# ---------------------------------------------------------------------------
class Elman_scratch_RNN(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_classes, pad_idx):
        super(Elman_scratch_RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.rnn = Elman(insize=emb_dim, outsize=hidden_dim, hsize=hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        Forward pass:
        x: (batch, time)
        We get embeddings -> (batch, time, emb_dim)
        Pass to Elman RNN -> (batch, time, hidden_dim)
        Project to (batch, time, num_classes)
        Pool over time dimension to get (batch, num_classes)
        """
        x = self.embedding(x)  # (batch, time, emb_dim)
        o_l2, _ = self.rnn(x)  # (batch, time, hidden_dim)
        output = self.linear2(o_l2)  # (batch, time, num_classes)

        # Global max pooling over time dimension
        output, _ = torch.max(output, dim=1)  # (batch, num_classes)

        return output
    

class Elman_torch_RNN(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_classes, pad_idx):
        super(Elman_torch_RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.RNN(insize=emb_dim, outsize=hidden_dim, hsize=hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        Forward pass:
        x: (batch, time)
        We get embeddings -> (batch, time, emb_dim)
        Pass to Elman RNN -> (batch, time, hidden_dim)
        Project to (batch, time, num_classes)
        Pool over time dimension to get (batch, num_classes)
        """
        x = self.embedding(x)  # (batch, time, emb_dim)
        o_l2, _ = self.rnn(x)  # (batch, time, hidden_dim)
        output = self.linear2(o_l2)  # (batch, time, num_classes)

        # Global max pooling over time dimension
        output, _ = torch.max(output, dim=1)  # (batch, num_classes)

        return output
    

class lstm_RNN(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_classes, pad_idx, num_layers_lstm=1):
        super(lstm_RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(input_size = emb_dim, hidden_size = hidden_dim, num_layers = num_layers_lstm, batch_first = True)
        self.linear2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        Forward pass:
        x: (batch, time)
        We get embeddings -> (batch, time, emb_dim)
        Pass to Elman RNN -> (batch, time, hidden_dim)
        Project to (batch, time, num_classes)
        Pool over time dimension to get (batch, num_classes)
        """
        x = self.embedding(x)  # (batch, time, emb_dim)
        o_l2, _ = self.rnn(x)  # (batch, time, hidden_dim)
        output = self.linear2(o_l2)  # (batch, time, num_classes)

        # Global max pooling over time dimension
        output, _ = torch.max(output, dim=1)  # (batch, num_classes)

        return output


class mlp_RNN(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_classes, pad_idx):
        """
        Parameters:
        - vocab_size (int): Size of the vocabulary.
        - emb_dim (int): Dimensionality of the embeddings.
        - hidden_dim (int): Dimensionality of the hidden layer.
        - num_classes (int): Number of output classes.
        - pad_idx (int): Index for the padding token.
        """
        super(mlp_RNN, self).__init__()
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

# ---------------------------------------------------------------------------
# Data Preparation
# ---------------------------------------------------------------------------
def prepare_data(x, y, w2i, batch_size):
    pad_idx = w2i[".pad"]
    max_len = max(len(seq) for seq in x)
    padded_x = [seq + [pad_idx] * (max_len - len(seq)) for seq in x]

    padded_x = torch.tensor(padded_x, dtype=torch.long)
    labels = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(padded_x, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader

# ---------------------------------------------------------------------------
# Training Function
# ---------------------------------------------------------------------------
def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training", leave=True)

    for batch_x, batch_y in progress_bar:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)  # (batch, num_classes)
        loss = loss_fn(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    return total_loss / len(dataloader)

# ---------------------------------------------------------------------------
# Evaluation Function
# ---------------------------------------------------------------------------
def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)  # (batch, num_classes)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == batch_y).sum().item()
            total += batch_y.size(0)

    return correct / total

# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------
def main():
    # Hyperparameters
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 1

    # Prepare data loaders
    train_loader = prepare_data(x_train, y_train, w2i, batch_size)
    val_loader = prepare_data(x_val, y_val, w2i, batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    vocab_size = len(w2i)
    emb_dim = 300
    hidden_dim = 300
    num_classes = numcls
    pad_idx = w2i[".pad"]

    model = RNN(vocab_size, emb_dim, hidden_dim, num_classes, pad_idx).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Training Loss: {train_loss:.4f}")

        val_accuracy = evaluate(model, val_loader, device)
        print(f"Validation Accuracy: {val_accuracy:.2%}")


def train_rnn(batch_size, learning_rate, num_epochs, nn_type):
    # Prepare data loaders
    train_loader = prepare_data(x_train, y_train, w2i, batch_size)
    val_loader = prepare_data(x_val, y_val, w2i, batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    vocab_size = len(w2i)
    emb_dim = 300
    hidden_dim = 300
    num_classes = numcls
    pad_idx = w2i[".pad"]

    if nn_type == "elman_scratch":
        model = Elman_scratch_RNN(vocab_size, emb_dim, hidden_dim, num_classes, pad_idx).to(device)
    elif nn_type == "elman_torch":
        model = Elman_torch_RNN(vocab_size, emb_dim, hidden_dim, num_classes, pad_idx).to(device)
    elif nn_type == "lstm":
        model = lstm_RNN(vocab_size, emb_dim, hidden_dim, num_classes, pad_idx, num_layers_lstm=1).to(device)
    elif nn_type == "mlp":
        model = mlp_RNN(vocab_size, emb_dim, hidden_dim, num_classes, pad_idx).to(device)
    # model = RNN(vocab_size, emb_dim, hidden_dim, num_classes, pad_idx).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Training Loss: {train_loss:.4f}")

        val_accuracy = evaluate(model, val_loader, device)
        print(f"Validation Accuracy: {val_accuracy:.2%}")

    return epoch+1, train_loss, val_accuracy

if __name__ == "__main__":
    # Hyperparameters
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 1
    nn_type = "mlp" # elman_scratch, elman_torch, lstm, mlp
    epoch_num, train_loss, val_accuracy = train_rnn(batch_size, learning_rate, num_epochs, nn_type)
