from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch


def load_image(IMAGE_DATASET,index):
    image_filename = IMAGE_DATASET+f"/{index}.jpg"
    try:
        # Open the image using PIL
        with Image.open(image_filename) as img:
            # Resize the image to the target size (128x128)
            img = img.resize((128, 128))
            # Convert the image to numpy array of type float32
            image_data = np.array(img).astype(np.float32)
        return image_data
    except FileNotFoundError:
        print(f"Image '{image_filename}' not found.")
        return None
        
def read_speed(file_path, index):
    with open(file_path, mode='r') as f:
        # Skip the header line
        f.readline()
        for i, line in enumerate(f, start=1):
            if i == index:
                image_count, linear_velocity, angular_velocity = line.strip().split('\t')
                linear_velocity = float(linear_velocity)
                angular_velocity = float(angular_velocity)
                # Create a torch tensor with float32 values
                return torch.tensor([linear_velocity, angular_velocity], dtype=torch.float32)

    # If the index is not found, return None
    return None

def write_loss_to_file(loss, file_path):
    # Check if the file exists
    try:
        with open(file_path, "a") as f:
            f.write(f"{loss}\n")
    except FileNotFoundError:
        # If the file doesn't exist, create it and then write to it
        with open(file_path, "w") as f:
            f.write(f"{loss}\n")

        

    
    
        
