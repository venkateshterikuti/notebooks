import os
import pickle

from .xb import np, to_cpu


def save_params(params: dict, path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    cpu_params = {k: to_cpu(v) for k, v in params.items()}
    with open(path, "wb") as f:
        pickle.dump(cpu_params, f)


def load_params(path: str, device: str = "cpu") -> dict:
    with open(path, "rb") as f:
        cpu_params = pickle.load(f)
    if device == "cuda":
        import cupy

        return {k: cupy.asarray(v) for k, v in cpu_params.items()}
    return cpu_params


def assign_params_to_model(model, params: dict) -> None:
    model_params = model.parameters()
    for k, v in params.items():
        model_params[k][...] = v
