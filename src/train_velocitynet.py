import argparse
import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import transforms
from utils import load_image, read_speed, write_loss_to_file
from VuNet import VuNet
from VelocityNet import VelocityNet
import random

if __name__ == "__main__":
    # Command-line argument parser setup
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to the configuration YAML file", default="config_VelocityNet.yml")
    parser.add_argument("--store_loss", help="Whether to store loss or not", action="store_true")
    parser.add_argument("--plot_loss", help="Whether to plot loss or not", action="store_true")
    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config, "r") as config_file:
        config = yaml.safe_load(config_file)

    # Define paths and files from configuration
    
    DATASET = config['image_dir']
    speed_file = config['velocity_file']  
    model_file_vunet = config['weights_vunet_dir']
    loss_file = config['loss_file']
    save_weights_velocitynet = config['weights_velocitynet_dir']
    
    # Initialize models
    vunet = VuNet(in_ch=3, out_ch=2)
    velocitynet = VelocityNet(in_ch=6, out_ch=4)
    
    # Load pre-trained weights for vunet
    state_dict = torch.load(model_file_vunet)
    vunet.load_state_dict(state_dict)
    
    # Set up the optimizer
    optimizer = torch.optim.Adam(velocitynet.parameters(), lr=config['learning_rate'])
    
    # Set training parameters
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    num_batches = config['num_iterations']
    
    # Define ranges for dataset splitting
    ranges = [
        range(58, 902)
    ]    
    #ranges = [
    #    range(58, 98),
    #    range(127, 391),
    #    range(420, 450),
    #    range(484, 543),
    #    range(1310, 1430)   
    #]
    
    # Set device to GPU if available
    device = torch.device("cuda")
    velocitynet.to(device)
    vunet.to(device)
    criterion = nn.L1Loss()
    
    loss_array = []  # Array to store loss values
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        # Iterate over batches for each epoch
        for batch_idx in range(num_batches):
            optimizer.zero_grad()
    
            # Load one image at a time for the current batch
            batch_loss = 0.0
            for i in range(batch_size):
                # Randomly choose a range from defined ranges
                chosen_range = random.choice(ranges)
                a = random.choice(chosen_range)
    
                # Load and preprocess images
                current_image = load_image(DATASET, a)
                future_image1 = load_image(DATASET, a + 10)
                future_image2 = load_image(DATASET, a + 20)
    
                # Read speed values from file
                vc1 = read_speed(speed_file, a)[0]
                wc1 = read_speed(speed_file, a)[1]
                vc2 = read_speed(speed_file, a + 10)[0]
                wc2 = read_speed(speed_file, a + 10)[1]
    
                # Reshape speed values to match network input shape
                vc1 = vc1.view(1, 1, 1, 1).to(device)
                vc2 = vc2.view(1, 1, 1, 1).to(device)
                wc1 = wc1.view(1, 1, 1, 1).to(device)
                wc2 = wc2.view(1, 1, 1, 1).to(device)
    
                # Convert images to tensors and move to device
                current_image = torch.from_numpy(current_image).unsqueeze(0).to(device)
                current_image = current_image.permute((0, 3, 1, 2))
    
                future_image1 = torch.from_numpy(future_image1).unsqueeze(0).to(device)
                future_image1 = future_image1.permute((0, 3, 1, 2))
    
                future_image2 = torch.from_numpy(future_image2).unsqueeze(0).to(device)
                future_image2 = future_image2.permute((0, 3, 1, 2))
    
                # Concatenate images
                concat1 = torch.cat((current_image, future_image2), dim=1)
                concat2 = torch.cat((future_image2, current_image), dim=1)
    
                # Pass through MPC network
                polinet1 = velocitynet(concat1)
                polinet2 = velocitynet(concat2)
    
                v11 = polinet1[:, 0:1, :, :]
                v21 = polinet1[:, 1:2, :, :]
                w11 = polinet1[:, 2:3, :, :]
                w21 = polinet1[:, 3:4, :, :]
    
                # Apply vunet for the first set of predictions
                x11 = vunet(current_image, v11, w11)
                x11 = x11.permute(0, 2, 3, 1)
                pred_vunet11 = torch.nn.functional.grid_sample(current_image, x11)
    
                x12 = vunet(pred_vunet11, v21, w21)
                x12 = x12.permute(0, 2, 3, 1)
                pred_vunet12 = torch.nn.functional.grid_sample(pred_vunet11, x12)
                
                # Define the loss for the first forward calculation
                J_img1 = (1/(2*128*128*3))*(criterion(pred_vunet11, future_image1) + criterion(pred_vunet12, future_image2))
                J_ref1 = 0.5*(torch.nn.functional.mse_loss(vc1, v11) + torch.nn.functional.mse_loss(vc2, v21) + torch.nn.functional.mse_loss(wc1, w11) + torch.nn.functional.mse_loss(wc2, w21))
    
                v12 = polinet2[:, 0:1, :, :]
                v22 = polinet2[:, 1:2, :, :]
                w12 = polinet2[:, 2:3, :, :]
                w22 = polinet2[:, 3:4, :, :]
    
                # Apply vunet for the second set of predictions
                x21 = vunet(future_image2, v12, w12)
                x21 = x21.permute(0, 2, 3, 1)
                pred_vunet21 = torch.nn.functional.grid_sample(future_image2, x21)
    
                x22 = vunet(pred_vunet21, v22, w12)
                x22 = x22.permute(0, 2, 3, 1)
                pred_vunet22 = torch.nn.functional.grid_sample(pred_vunet21, x22)
    
                # Define the loss for the second forward calculation
                J_img2 = (1/(2*128*128*3))*(criterion(pred_vunet21, future_image1) + criterion(pred_vunet22, current_image))
                J_ref2 = 0.5*(torch.nn.functional.mse_loss(-vc2, v12) + torch.nn.functional.mse_loss(-vc1, v22) + torch.nn.functional.mse_loss(-wc2, w12) + torch.nn.functional.mse_loss(w22, -wc1))
    
                # Calculate total loss 
                loss = 0.5*(J_img1 + J_img2) + 0.1*(J_ref1 + J_ref2)
                batch_loss += loss
    
            # Backpropagation
            batch_loss.backward()
            optimizer.step()
    
            # Accumulate loss for the epoch
            epoch_loss += batch_loss
    
        # Calculate average epoch loss
        epoch_loss /= num_batches
        loss_array.append(epoch_loss.item()) 
        # Print epoch loss
        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {epoch_loss}")
    
        # Store epoch loss if specified
        if args.store_loss:
            write_loss_to_file(epoch_loss, loss_file)
    
    print("Training finished.")
    
    # Save trained model weights
    torch.save(velocitynet.state_dict(), save_weights_velocitynet)
    
    # Plotting loss if specified   
    if args.plot_loss:
        plt.plot(loss_array)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss over Epochs')
        plt.show()

	
