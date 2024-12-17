import argparse  # For parsing command-line arguments
import os  # For interacting with the operating system
import time  # For time-related operations
import subprocess  # For running subprocesses
import torch  # PyTorch deep learning framework
import torch.distributed as dist  # For distributed training
import torch.nn as nn  # Neural network modules
import torch.optim as optim  # Optimization algorithms
import torchvision.transforms as transforms  # Image transformations
from torch.utils.data import DataLoader  # Data loading utilities
from torch.optim.lr_scheduler import CosineAnnealingLR  # Learning rate scheduler
from resnest.torch import resnest50_fast_1s1x64d  # ResNeSt model (assumed)
import torchvision.datasets as datasets  # Standard datasets
from torch.utils.data.distributed import DistributedSampler  # Distributed data sampler

def get_gpu_utilization_and_memory(gpu_index):
    command = f"nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,nounits,noheader -i {gpu_index}"
    result = subprocess.check_output(command.split()).decode('utf-8').strip().split(',')
    gpu_utilization = float(result[0])  # GPU utilization percentage
    gpu_memory_usage = float(result[1])  # GPU memory usage in MB
    return gpu_utilization, gpu_memory_usage

def get_cgroup_metrics():
    cpu_usage_path = '/sys/fs/cgroup/cpu/cpuacct.usage'
    mem_usage_path = '/sys/fs/cgroup/memory/memory.usage_in_bytes'

    with open(cpu_usage_path, 'r') as f:
        cpu_usage = int(f.read().strip()) / 1e9  # Convert nanoseconds to seconds
    with open(mem_usage_path, 'r') as f:
        mem_usage = int(f.read().strip()) / (1024 ** 2)  # Convert bytes to MB

    return round(cpu_usage, 2), round(mem_usage, 2)

def train(model, train_loader, optimizer, criterion, device, epoch, args):
    model.train()  # Setting the model to train mode
    start_time = time.time()  # saving start time
    total_correct = 0  # Initializing total correct predictions
    total_samples = 0  # Initializing total processed samples

    dataset_size = len(train_loader.dataset)  # Calculate the total number of samples in the dataset
    max_samples = args.max_samples if args.max_samples != -1 else dataset_size  # Set max_samples based on the argument or dataset size

    gpu_device = torch.device('cuda')  # the GPU device, in our case it's cuda
    model.to(gpu_device)  # Move model to the GPU device once

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        if max_samples != -1 and total_samples >= max_samples:
            break  # Stop when max_samples reached

        inputs, labels = inputs.to(gpu_device), labels.to(gpu_device)  # Move inputs and labels to GPU
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Calculate loss

        optimizer.zero_grad()  # Zero gradients
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        _, predicted = outputs.max(1)  # Get the index of the max log-probability
        total_correct += predicted.eq(labels).sum().item()  # Count correct predictions
        total_samples += labels.size(0)  # Count processed samples

        if batch_idx % args.print_interval == 0:
            elapsed_time = time.time() - start_time  # Calculate elapsed time
            accuracy = 100. * total_correct / total_samples  # Calculate accuracy
            gpu_utilization, gpu_memory_usage = get_gpu_utilization_and_memory(0)  # Get GPU metrics (assuming single GPU)
            cpu_usage, memory_usage = get_cgroup_metrics()  # Get CPU and memory metrics

            if batch_idx == 0:
                # Print CSV header
                print('epoch,batch_idx,num_samples,percentage_completed,loss,time_elapsed,learning_rate,accuracy,gpu_utilization,gpu_memory_usage,cpu_usage,mem_usage')

            # Print CSV formatted metrics
            percentage_completed = (total_samples / max_samples) * 100
            print(f'{epoch},{batch_idx},{total_samples},{percentage_completed:.2f}%,{loss.item():.6f},{elapsed_time:.2f},{optimizer.param_groups[0]["lr"]:.6f},{accuracy:.2f},{gpu_utilization},{gpu_memory_usage},{cpu_usage},{memory_usage}')

def main():
    parser = argparse.ArgumentParser(description='Distributed Training Example')
    parser.add_argument('--backend', type=str, default='nccl', help='Backend to use for distributed training')
    parser.add_argument('--batch_size', type=int, default=64, help='Input batch size for training')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--num_train_epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for optimizer')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--evaluation_strategy', type=str, default='no', help='Evaluation strategy')
    parser.add_argument('--save_strategy', type=str, default='epoch', help='Save strategy')
    parser.add_argument('--lr_scheduler_type', type=str, default='cosine', help='Type of learning rate scheduler')
    parser.add_argument('--pretrained_weights', type=str, default='', help='Path to pre-trained weights')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of DataLoader workers')
    parser.add_argument('--max_samples', type=int, default=-1, help='Maximum number of samples per epoch (-1 for full dataset, only works with a single node)')
    parser.add_argument('--print_interval', type=int, default=10, help='Interval for printing metrics (in batches)')
    parser.add_argument('--use_syn', action='store_true', help='Use synthetic data')
    parser.add_argument('--output_dir', type=str, default='.', help='Output directory for saving models')
    args = parser.parse_args()

    dist.init_process_group(backend=args.backend)  # Initialize the process group for distributed training
    world_size = dist.get_world_size()  # Get the total number of nodes
    if args.max_samples != -1 and world_size > 1:
        print("Error: --max_samples cannot be used with more than one node.")
        exit(1)

    # Set up dataset and model
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # default values, see  https://pytorch.org/vision/main/generated/torchvision.transforms.Normalize.html
    ])

    trainset = datasets.ImageFolder(root=os.path.join(args.data_path, 'train'), transform=transform)
    model = resnest50_fast_1s1x64d(pretrained=False)  # Loading ResNeSt model

    sampler = DistributedSampler(trainset) if dist.is_initialized() else None  # Use DistributedSampler for distributed training

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=(sampler is None),
                              num_workers=args.num_workers, sampler=sampler)  # DataLoader for training data

    model = model.cuda()  # Move ResNet model to GPU
    model = torch.nn.parallel.DistributedDataParallel(model)  # Wrap model for distributed training

    criterion = nn.CrossEntropyLoss()  # Loss function

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)  # Optimizer
    if args.lr_scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.num_train_epochs)  # Learning rate scheduler

    for epoch in range(args.num_train_epochs):
        train(model, train_loader, optimizer, criterion, torch.device('cuda'), epoch, args)  # Training function

        if args.lr_scheduler_type == 'cosine':
            scheduler.step()  # Scheduler step

        if args.save_strategy == 'epoch':
            os.makedirs(args.output_dir, exist_ok=True)  # Create the output directory if it doesn't exist
            torch.save(model.state_dict(), os.path.join(args.output_dir, f'model_epoch_{epoch}.pth'))  # Save model after each epoch

if __name__ == '__main__':
    main()  # Main function entry point
