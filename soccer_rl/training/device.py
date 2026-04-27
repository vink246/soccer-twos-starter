import torch


def resolve_device(spec: str) -> torch.device:
    s = (spec or "auto").lower()
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)
