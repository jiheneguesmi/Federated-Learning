# Federated Learning with Flower

This project demonstrates the implementation of a federated learning system using the [Flower framework](https://flower.dev/). The notebook walks through the process of setting up a federated learning environment, training a model across multiple simulated clients, and aggregating the results on a central server.

## Features

- **Centralized Training**: A baseline model is trained on a single dataset partition to establish a reference accuracy.
- **Federated Learning**: The Flower framework is used to simulate federated learning with multiple clients, each training on its own local data.
- **Custom Metrics Aggregation**: A weighted average function is implemented to aggregate accuracy metrics across clients.

## Requirements

The following libraries are required to run the notebook:
- `flwr[simulation]`
- `flwr-datasets[vision]`
- `torch`
- `torchvision`
- `matplotlib`

Install the dependencies using the following command:
```bash
pip install -q flwr[simulation] flwr-datasets[vision] torch torchvision matplotlib
```

## Notebook Structure

1. **Step 0: Preparation**
   - Install dependencies and set up the environment.
   - Import necessary libraries and configure the runtime (CPU/GPU).

2. **Step 1: Centralized Training**
   - Define a simple Convolutional Neural Network (CNN) using PyTorch.
   - Train the model on a single data partition to establish a baseline.

3. **Step 2: Federated Learning**
   - Simulate federated learning with 10 clients using the Flower framework.
   - Each client trains the model on its local data and sends updates to the server.
   - The server aggregates the updates using the Federated Averaging (FedAvg) strategy.

4. **Custom Metrics Aggregation**
   - Implement a weighted average function to compute global accuracy from client metrics.

## How to Run

1. Open the notebook `FederatedLearning_Flower.ipynb` in Jupyter Notebook or JupyterLab.
2. Execute the cells sequentially to:
   - Set up the environment.
   - Train the model in a centralized manner.
   - Simulate federated learning with multiple clients.
3. Observe the results, including the aggregated accuracy and loss metrics.

## Results

- The centralized training provides a baseline accuracy for comparison.
- Federated learning demonstrates how a global model can be trained collaboratively without sharing raw data between clients.

## References

- [Flower Documentation](https://flower.dev/docs/)
- [PyTorch Documentation](https://pytorch.org/docs/)

