import argparse
import yaml
from utils import load_image
from VelocityNet import VelocityNet
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import transforms

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to the configuration file", default="config_VelocityNet.yml")
    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config, "r") as config_file:
        config = yaml.safe_load(config_file)

    # Extract necessary parameters from the configuration
    
    DATASET = config['image_dir']
    saved_weights_velocitynet = config['weights_velocitynet_dir']
    
    # Load images from the dataset
    img1 = load_image(DATASET, 318)
    img2 = load_image(DATASET, 318)   
    
    # Convert images to tensors and move to device
    current_image = torch.from_numpy(img1).unsqueeze(0)
    current_image = current_image.permute((0, 3, 1, 2))
    
    future_image1 = torch.from_numpy(img2).unsqueeze(0)
    future_image1 = future_image1.permute((0, 3, 1, 2))
    
    # Concatenate images
    concat = torch.cat((current_image, future_image1), dim=1)
    
    # Instantiate VelocityNet model
    velocitynet = VelocityNet(in_ch=6, out_ch=4)
    
    # Load pre-trained weights for the VelocityNet
    state_dict = torch.load(saved_weights_velocitynet)
    velocitynet.load_state_dict(state_dict)
    
    # Pass the concatenated images through the VelocityNet model
    velocity = velocitynet(concat)
    
    # Extract individual components of velocity
    v11 = velocity[:, 0:1, :, :]
    v21 = velocity[:, 1:2, :, :]
    w11 = velocity[:, 2:3, :, :]
    w21 = velocity[:, 3:4, :, :]
    
    # Print a sample of velocity components
    print(v11[0, 0, 0, 0], w11[0, 0, 0, 0])
