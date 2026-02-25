import numpy.typing as npt
import numpy as np
import torch

def data_loading(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    # 1. compute max index
    max_idx = len(dataset) - context_length - 1
    
    # 2. random sample
    ix = torch.randint(0, max_idx + 1, (batch_size,))
    
    # 3. get the next context_length tokens
    # x starts from ix[i]
    # y starts from ix[i] + 1
    x_list = [torch.from_numpy(dataset[i : i + context_length].astype(np.int64)) for i in ix]
    y_list = [torch.from_numpy(dataset[i + 1 : i + 1 + context_length].astype(np.int64)) for i in ix]
    
    # 4. shape it into (batch_size, context_length) tensor
    x = torch.stack(x_list)
    y = torch.stack(y_list)
    
    # 5. move to cuda
    return x.to(device), y.to(device)