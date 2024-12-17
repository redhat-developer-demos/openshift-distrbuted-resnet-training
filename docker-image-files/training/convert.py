import os
import pickle
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

def load_cifar_batch(file):
    try:
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    except FileNotFoundError:
        print(f"File {file} not found.")
        return None
    except Exception as e:
        print(f"Error loading file {file}: {e}")
        return None

def convert_cifar10_to_imagefolder(root_dir, output_dir):
    meta = load_cifar_batch(os.path.join(root_dir, 'batches.meta'))
    if meta is None:
        return
    classes = [label.decode('utf-8') for label in meta[b'label_names']]

    for cls in classes:
        os.makedirs(os.path.join(output_dir, 'train', cls), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'test', cls), exist_ok=True)

    train_batches = [f for f in os.listdir(root_dir) if f.startswith('data_batch_')]

    for batch_file in train_batches:
        data_dict = load_cifar_batch(os.path.join(root_dir, batch_file))
        if data_dict is None:
            continue
        data = data_dict[b'data']
        labels = data_dict[b'labels']

        batch_number = batch_file.split('_')[-1]
        for i in tqdm(range(len(data)), desc=f'Converting {batch_file}'):
            img = data[i]
            img = img.reshape(3, 32, 32).transpose(1, 2, 0)
            img = Image.fromarray(img)
            label = labels[i]
            img.save(os.path.join(output_dir, 'train', classes[label], f'batch{batch_number}_img{i}.png'))

    test_dict = load_cifar_batch(os.path.join(root_dir, 'test_batch'))
    if test_dict is None:
        return
    data = test_dict[b'data']
    labels = test_dict[b'labels']

    for i in tqdm(range(len(data)), desc='Converting test batch'):
        img = data[i]
        img = img.reshape(3, 32, 32).transpose(1, 2, 0)
        img = Image.fromarray(img)
        label = labels[i]
        img.save(os.path.join(output_dir, 'test', classes[label], f'test_img{i}.png'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert CIFAR-10 dataset to ImageFolder format.')
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory of the dataset batches.')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for the ImageFolder format.')
    
    args = parser.parse_args()
    
    convert_cifar10_to_imagefolder(args.root_dir, args.output_dir)
