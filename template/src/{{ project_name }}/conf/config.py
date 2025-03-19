from dataclasses import dataclass

@dataclass
class Data:
    data_dir: str
    logs_dir: str
    batch_size: int
    num_workers: int
    latent_dim: int

@dataclass
class Train:
    accelerator: str
    epochs: int

@dataclass
class Optim:
    lr: float

@dataclass
class Config:
    data: Data
    train: Train
    optim: Optim
