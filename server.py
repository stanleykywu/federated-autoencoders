import flwr as fl
from flwr.server.strategy import FedAvg, Strategy


def main():
    strategy = FedAvg(min_available_clients=1, min_fit_clients=1, min_eval_clients=1)
    fl.server.start_server(
        "localhost:8080", config={"num_rounds": 3}, strategy=strategy
    )


if __name__ == "__main__":
    main()
