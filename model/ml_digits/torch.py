import torch

device: str = (
    torch.accelerator.current_accelerator().type  # type: ignore
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Using {device} device")
