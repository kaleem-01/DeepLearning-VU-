import torch
import torch.nn.functional as F

def pad_and_convert_to_tensor(batch_data, batch_labels, w2i):
    """
    Pads sequences in a batch to the same length and converts them to PyTorch tensors.

    Parameters:
    - batch_data (list of list of int): A batch of sequences (list of word indices).
    - batch_labels (list of int): Corresponding class labels for the sequences.
    - w2i (dict): Dictionary mapping words to indices.

    Returns:
    - padded_tensor (torch.Tensor): A tensor of padded sequences (batch_size, max_seq_len).
    - labels_tensor (torch.Tensor): A tensor of labels (batch_size).
    """
    # Get the padding token index
    pad_idx = w2i[".pad"]
    
    # Find the maximum sequence length in the batch
    max_seq_len = max(len(seq) for seq in batch_data)
    
    # Pad sequences to the maximum length
    padded_sequences = [
        seq + [pad_idx] * (max_seq_len - len(seq)) for seq in batch_data
    ]
    
    # Convert to PyTorch tensors
    padded_tensor = torch.tensor(padded_sequences, dtype=torch.long)
    labels_tensor = torch.tensor(batch_labels, dtype=torch.long)
    
    return padded_tensor, labels_tensor

def get_loss(num_classes):
    # Example loss computation
    logits = torch.randn(32, num_classes)  # Example logits from the model
    targets = torch.randint(0, num_classes, (32,))  # Example ground truth
    loss = F.cross_entropy(logits, targets)

    # Example accuracy computation
    predictions = torch.argmax(logits, dim=1)  # Predicted classes
    accuracy = (predictions == targets).float().mean().item()

    print(f"Loss: {loss.item()}, Accuracy: {accuracy}")

    return loss, accuracy
