# OpenShift PyTorch Example

This repository provides an example of how to run a PyTorch training job on OpenShift. The example demonstrates setting up a distributed training job using OpenShift resources and the PyTorchJob API. This repository is used in the walkthrough document ["RoCE Multi node AI training on OpenShift"](https://docs.google.com/document/d/1OkNcrCeYQpHlG9x4rJwdywCWMejI1iWedCKnAm8yfm8/edit).

## Table of Contents

- [Prerequisites](#prerequisites)
- [Directory info](#Directory-info)
- [Setup](#setup)
- [Running the Example](#running-the-example)
- [PyTorch Script Arguments](#pytorch-script-arguments)
- [Convert Script Arguments](#convert-script-arguments)

## Directory info

This repository is organized as follows:

- **docker-image-files/**: Contains Dockerfile and related scripts for building the Docker image used for the PyTorch training job.
  - **Dockerfile**: Defines the environment and dependencies for the PyTorch training container.
  - **entrypoint.sh**: Script that sets up the environment variables and starts the training job.

- **examples/**: Contains example scripts and configurations for running and testing the PyTorch training job.
  - **pytorchjob.yaml**: Defines a basic PyTorchJob resource for running the distributed training job on OpenShift.
  - **pytorch-using-entrypoint.yaml**: Defines the PyTorchJob resource for running the distributed training job by setting the environment variables inside the file and using the entrypoint.sh to execute the training on OpenShift.

## Prerequisites

Before you begin, ensure you have the following prerequisites:

- OpenShift cluster up and running.
- `oc` command-line tool installed and configured.
- docker/podman installed for building the container images.
- Basic knowledge of Kubernetes or OpenShift.

## Setup

1. **Clone the repository:**

    ```bash
    git clone git@github.com:redhat-developer-demos/openshift-distrbuted-resnet-training.git
    cd openshift-distrbuted-resnet-training
    ```

2. **Build the Docker image:**

    ```bash
    docker build -t <your-dockerhub-username>/pytorch-training:latest .
    ```

3. **Push the Docker image to your Docker registry:**

    ```bash
    docker push <your-dockerhub-username>/pytorch-training:latest
    ```

## Running the Example

1. **Apply the Kubernetes resources:**

    ```bash
    oc create -f pytorchjob.yaml
    ```

2. **Verify the job is running:**

    ```bash
    oc get pods
    ```

3. **Check the logs of the training job (for the job with the entry_point.sh):**

    ```bash
    oc logs <pod-name>
    ```

## PyTorch Script Arguments

The `main.py` script accepts the following arguments:

- **`--backend`**: Backend to use for distributed training (default: `nccl`)
- **`--batch_size`**: Input batch size for training (default: `64`)
- **`--data_path`**: Path to the dataset (required)
- **`--num_train_epochs`**: Number of training epochs (default: `1`)
- **`--learning_rate`**: Learning rate for optimizer (default: `0.001`)
- **`--weight_decay`**: Weight decay for optimizer (default: `0.0`)
- **`--gradient_accumulation_steps`**: Gradient accumulation steps (default: `1`)
- **`--evaluation_strategy`**: Evaluation strategy (default: `no`)
- **`--save_strategy`**: Save strategy (default: `epoch`)
- **`--lr_scheduler_type`**: Type of learning rate scheduler (default: `cosine`)
- **`--pretrained_weights`**: Path to pre-trained weights (default: `''`)
- **`--num_workers`**: Number of DataLoader workers (default: `2`)
- **`--max_samples`**: Maximum number of samples per epoch (-1 for full dataset, only works with a single node) (default: `-1`)
- **`--print_interval`**: Interval for printing metrics (in batches) (default: `10`)
- **`--use_syn`**: Use synthetic data (default: `False`)
- **`--output_dir`**: Output directory for saving models (default: `.`)

Example usage:

```bash
torchrun --nproc_per_node=1 --nnodes=3 --node_rank=2 --master_addr=192.168.1.5 --master_port=23456 main.py --backend=nccl --batch_size=128 --data_path=/mnt/storage/dataset/cifar10_imagefolder --num_train_epochs=1 --learning_rate=0.001 --num_workers=5 --print_interval=5 --output_dir /mnt/storage/

```

## Convert Script Arguments

The `convert.py` script accepts the following arguments:

- **`--root_dir`**: Root directory of the dataset batches (required).
- **`--output_dir`**: Output directory for the ImageFolder format (required).

Example usage:

```bash
python convert_cifar10_to_imagefolder.py --root_dir /path/to/cifar-10-batches-py --output_dir /path/to/output
