import torch


def get_device() -> str:
    """Determine device to run on (CPU, NVidia GPU, or Apple MPS).

    Returns:
        str: device string
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"
