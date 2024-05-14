from typing import List, Tuple

from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg, FedAdagrad, FedAdam, FedYogi
from flwr.common import Metrics

import argparse
import torch
import torch.nn as nn
import torchvision.models as models

from flwr.common import ndarrays_to_parameters

from utils import plot_metric_from_history, save_results_as_pickle

from datetime import datetime

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


# Get partition id
parser = argparse.ArgumentParser(description="Flower")
parser.add_argument(
    "--strategy",
    choices=["fedavg", "fedadagrad", "fedadam", "fedyogi"],
    default="fedavg",
    type=str,
    help="Strategy to use for federated learning.",
)
parser.add_argument(
    "--num_rounds",
    default=3,
    type=int,
    help="Number of rounds to simulate.",
)

# Define strategy and resnet18 model for 10 classes 

class ResNet(nn.Module):
    def __init__(self) -> None:
        super(ResNet, self).__init__()
        self.resnet18 = models.resnet18(pretrained=False)
        self.resnet18.fc = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet18(x)

# initial_parameters = ndarrays_to_parameters([val.cpu().numpy() for _, val in resnet18().state_dict().items()])  # type: ignore
initial_parameters = ndarrays_to_parameters([val.cpu().numpy() for _, val in ResNet().state_dict().items()])  # type: ignore

# initial_parameters = [, dtype=float32)]
if parser.parse_known_args()[0].strategy == "fedavg":
    strategy = FedAvg(evaluate_metrics_aggregation_fn=weighted_average)
elif parser.parse_known_args()[0].strategy == "fedadagrad":
    strategy = FedAdagrad(evaluate_metrics_aggregation_fn=weighted_average, initial_parameters=initial_parameters)
elif parser.parse_known_args()[0].strategy == "fedadam":
    strategy = FedAdam(evaluate_metrics_aggregation_fn=weighted_average, initial_parameters=initial_parameters)
elif parser.parse_known_args()[0].strategy == "fedyogi":
    strategy = FedYogi(evaluate_metrics_aggregation_fn=weighted_average, initial_parameters=initial_parameters)
else:
    raise ValueError("Invalid strategy.")

# Define config
config = ServerConfig(num_rounds=parser.parse_known_args()[0].num_rounds)

# ServerApp
app = ServerApp(
    config=config,
    strategy=strategy,
)

# Legacy mode
if __name__ == "__main__":
    from flwr.server import start_server

    history = start_server(
        server_address="0.0.0.0:8080",
        config=config,
        strategy=strategy,
    )

    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = "./results/" + f"{strategy}{time}_results.pkl"

    # save results as a Python pickle using a file_path
    # the directory created by Hydra for each run
    save_results_as_pickle(
        history,
        file_path=save_path,
    )
    # plot results and include them in the readme
    strategy_name = strategy.__class__.__name__
    file_suffix: str = (
        f"_{strategy}"
        f"_{time}"
    )

    plot_metric_from_history(
        history,
        save_path,
        (file_suffix),
    )
