import torch

def resolve_device(requested_device, logger):
    cuda_available = torch.cuda.is_available()
    logger.info("CUDA available: %s", cuda_available)
    if requested_device == "auto" and cuda_available:
        logger.info("CUDA GPU: %s", torch.cuda.get_device_name(0))
        return torch.device("cuda")
    if (
        requested_device == "auto"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    if requested_device == "auto":
        return torch.device("cpu")
    if requested_device == "cpu":
        return torch.device("cpu")
    if requested_device == "cuda" and not cuda_available:
        logger.warning("CUDA was requested but is not available; falling back to CPU.")
        return torch.device("cpu")
    if requested_device == "cuda":
        logger.info("CUDA GPU: %s", torch.cuda.get_device_name(0))
        return torch.device("cuda")
    if requested_device == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        logger.warning("MPS was requested but is not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested_device)
