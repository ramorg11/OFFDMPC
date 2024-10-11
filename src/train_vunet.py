import argparse
import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

from torchvision import transforms
from utils import load_image, read_speed, write_loss_to_file
from VuNet import VuNet

if __name__ == "__main__":
    # Creating an argument parser instance
    parser = argparse.ArgumentParser()
    # Adding arguments for configuration file path, storing loss, and plotting loss
    parser.add_argument("--config", help="Path to the configuration YAML file", default="config_vunet.yml")
    parser.add_argument("--store_loss", help="Whether to store loss or not", action="store_true")
    parser.add_argument("--plot_loss", help="Whether to plot loss or not", action="store_true")
    # Parsing command-line arguments
    args = parser.parse_args()

    # Reading configuration from the specified YAML file
    with open(args.config, "r") as config_file:
        config = yaml.safe_load(config_file)

    # Setting up file paths and directories from the configuration
    
    image_dir= config['image_dir']
    velocity_file = config['velocity_file']  # File containing speed data
    weights_file = config['weights_file']  # File to save trained model weights
    loss_file = config['loss_file']  # File to store loss values during training
    
    
    # Initializing the VuNet model
    vunet = VuNet(in_ch=3, out_ch=2)
    
    # Setting up loss function and optimizer
    criterion = nn.L1Loss()
    optimizer = optim.Adam(vunet.parameters(), lr=config['learning_rate'])
    
    # Training parameters
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    num_batches = config['num_iterations']
    device = torch.device('cuda')
    
    # Moving model to GPU if available
    vunet.to(device)
    
    # Initializing speed scalars
    vresc = torch.ones((1, 1, 1, 1), dtype=torch.float32).to(device)
    wresc = torch.ones((1, 1, 1, 1), dtype=torch.float32).to(device)
    
    loss_array = []  # Array to store loss values
    for epoch in range(num_epochs):
        epoch_loss = 0.0
    
        for batch_idx in range(num_batches):
            batch_loss = 0.0
            for i in range(batch_size):
                
                # normalize between -1 and 1
                current_image = (load_image(image_dir, batch_idx * batch_size + i+1)-128)/128.0
                future_image = (load_image(image_dir, batch_idx * batch_size + i +11)-128)/128.0
                future_image = torch.tensor(future_image, dtype=torch.float32).unsqueeze(0).to(device)
                
                # Converting current image for processing
                resized_img1 = np.array(current_image)
                resized_img_float32 = resized_img1.astype(np.float32)[np.newaxis, ...]
                resized_img_torch = torch.tensor(resized_img_float32, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
                
                # Reading speed information
                vresc[0][0][0], wresc[0][0][0] = read_speed(velocity_file, batch_idx * batch_size + i + 1)
               
                
                # Apply the Encoder and Decoder
                x = vunet(resized_img_torch, vresc, wresc)
    
                x=x.permute(0, 2, 3, 1)
    
                pred_vunet = torch.nn.functional.grid_sample(resized_img_torch, x)
                pred_vunet = pred_vunet.permute(0, 2, 3, 1)
                # Transpose the output tensor
    
                # Compute the loss
                loss = criterion(pred_vunet, future_image)                
                batch_loss += loss
    
            # Backpropagation
            batch_loss.backward()
            
            # Update model parameters
            optimizer.step()
            optimizer.zero_grad()
    
            batch_loss /= batch_size
            epoch_loss += batch_loss
    
        # Calculate average loss for the epoch
        average_epoch_loss = epoch_loss / num_batches
        
        loss_array.append(average_epoch_loss.item())  # Append loss value to the array
        
        # Print the average loss for this epoch
        print(f"Epoch {epoch + 1}, Loss: {average_epoch_loss}")
        
        if args.store_loss:
            write_loss_to_file(average_epoch_loss, loss_file)
           
    print("Training finished.")
    
    # Saving model weights
    torch.save(vunet.state_dict(), weights_file)
    
    # Plotting loss if specified
    if args.plot_loss:
        plt.plot(loss_array)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss over Epochs')
        plt.show()
        


