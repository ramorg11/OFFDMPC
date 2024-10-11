import argparse
import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from utils import load_image
from VuNet import VuNet

if __name__ == "__main__":
    # Command-line argument parser setup
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to the configuration YAML file", default="config_vunet.yml")
    args = parser.parse_args()

    # Loading configuration from YAML file
    with open(args.config, "r") as config_file:
        config = yaml.safe_load(config_file)

    # Instantiate the VuNet model with input and output channels
    vunet = VuNet(in_ch=3, out_ch=2)
    
    # Extracting necessary information from the config file
    DATASET = config['image_dir']
    model_file_vunet = config['weights_file']
    
    # Loading and preprocessing the input image
    image = load_image(DATASET, 500)
    normalized_image = (image - 128) / 128.0
    np_image = np.array(normalized_image)
    resized_img_float32 = np_image.astype(np.float32)[np.newaxis, ...]
    resized_img_torch = torch.tensor(resized_img_float32, dtype=torch.float32).permute(0, 3, 1, 2)
    
    # Chosing the corresponding velocity commands
    vresc = 0 * torch.ones((1, 1, 1, 1), dtype=torch.float32)
    wresc = 1 * torch.ones((1, 1, 1, 1), dtype=torch.float32)
    
    # Load pre-trained weights into the VuNet model
    state_dict = torch.load(model_file_vunet)
    vunet.load_state_dict(state_dict)
    
    # Forward pass through the model to get predictions
    x = vunet(resized_img_torch, vresc, wresc)
    x = x.permute(0, 2, 3, 1)
    
    # Perform grid sampling to obtain the predicted image
    pred_vunet = torch.nn.functional.grid_sample(resized_img_torch, x)
    pred_vunet = pred_vunet.permute(0, 2, 3, 1)
    prediction = (pred_vunet[0] * 128) + 128
    
    # Plot the results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(prediction.detach().numpy().astype(np.uint8))
    plt.title('Predicted Image')
    
    plt.subplot(1, 2, 2)
    plt.imshow(image.astype(np.uint8))
    plt.title('Real Image')
    
    plt.show()
