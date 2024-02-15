# Maximum dimensions across all files: X: 330, Y: 337, Z: 22
import os
import tifffile
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import csv
device = torch.device('cuda:0')  if torch.cuda.is_available() else torch.device('mps:0') if torch.backends.mps.is_available() else torch.device('cpu')
# device = 'cpu'
print(device)

def create_image_mapping(directory):
    """
    Create a mapping of image names to their tensors and placeholders for ratios.

    Parameters:
    - directory (str): Path to the directory containing TIFF files.

    Returns:
    - dict: A dictionary mapping image names to dictionaries containing image tensors and ratio placeholders.
    """

    # Function to load a TIFF image as a tensor
    def load_tiff_as_tensor(file_path):
        image_array = tifffile.imread(file_path)
        # Convert the array to float32 before creating the tensor
        image_array = image_array.astype('float32')
        return torch.from_numpy(image_array).float()  # .float() is now redundant but kept for clarity


    # Initialize the mapping structure
    image_mapping = {}

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        # Check if the current object is a file and has a .tif extension
        if os.path.isfile(file_path) and filename.lower().endswith('.tif'):
            # Load the image tensor
            image_tensor = load_tiff_as_tensor(file_path)
            
            # Extract the image name without the extension
            image_name = os.path.splitext(filename)[0]
            
            # Add to the mapping
            image_mapping[image_name] = {
                'tensor': image_tensor,
                'ratio': None  # Placeholder for the ratio, to be filled in later
            }

    return image_mapping

def update_image_mapping_with_ratios(image_mapping, csv_file):
    """
    Update the image_mapping dictionary with ratios read from a CSV file and drop entries not present in the CSV.

    Parameters:
    - image_mapping (dict): The mapping of image names to their tensors and placeholders for ratios.
    - csv_file (str): Path to the CSV file containing image names and their corresponding ratios.

    Returns:
    - dict: The updated image_mapping dictionary with ratios filled in and non-listed entries removed.
    """

    # Read the CSV file into a DataFrame
    ratios_df = pd.read_csv(csv_file)
    
    # Create a set of image names from the CSV file (without the file extension)
    csv_image_names = set(ratios_df['file_name'].str.split('.').str[0])
    
    # Iterate over the rows of the DataFrame
    for index, row in ratios_df.iterrows():
        image_name = row['file_name'].split('.')[0]  # Assuming the file name includes the extension '.tif'
        ratio = row['element_pixel_intensity_405_488_ratio']
        
        # Update the ratio in the image_mapping if the image name exists
        if image_name in image_mapping:
            image_mapping[image_name]['ratio'] = ratio

    # Filter the image_mapping to keep only those entries that are listed in the CSV file
    image_mapping = {key: value for key, value in image_mapping.items() if key in csv_image_names}

    return image_mapping

class Simple3DCNN(nn.Module):
    def __init__(self, in_channels=1, init_features=32, num_classes=1):
        super(Simple3DCNN, self).__init__()
        features = init_features
        self.encoder1 = self.conv_block(in_channels, features, kernel_size=3, stride=1, padding=1)
        self.encoder2 = self.conv_block(features, features * 2, kernel_size=3, stride=2, padding=1)
        self.encoder3 = self.conv_block(features * 2, features * 4, kernel_size=3, stride=2, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))  # Ensure output size is 1x1x1
        self.fc = nn.Linear(features * 4, num_classes)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        pooled = self.adaptive_pool(enc3)
        out = pooled.view(pooled.size(0), -1)  # Flatten
        return self.fc(out)

    def forward_features(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        pooled = self.adaptive_pool(enc3)
        features = pooled.view(pooled.size(0), -1)  # Flatten
        return features


    def conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        return block
    
def train_model_regression(model, optimizer, criterion, data_loader, epochs=10, device='cpu'):
    model.to(device)
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, ((data, ratios), image_names) in enumerate(data_loader):  # Include batch_idx here
            data = data.to(device)
            ratios = ratios.to(device).unsqueeze(1).float()  # Ensure ratios are float and have correct shape

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, ratios)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # if batch_idx % 10 == 9:  # print every 10 mini-batches
            if batch_idx % 1 == 0: # print every mini-batch
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, batch_idx + 1, running_loss / 10))
                running_loss = 0.0


class ImageRatioDataset(Dataset):
    
    def __init__(self, image_mapping, max_size=None):
        self.image_mapping = image_mapping
        self.image_names = list(image_mapping.keys())
        self.max_size = max_size if max_size else self.calculate_max_size()

    def __len__(self):
        return len(self.image_mapping)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image = self.image_mapping[image_name]['tensor']
        ratio = torch.tensor(self.image_mapping[image_name]['ratio'], dtype=torch.float)

        # Pad the image to max_size
        padded_image = self.pad_to_max_size(image.unsqueeze(0), self.max_size)  # Add channel dimension

        return (padded_image, ratio), image_name  # Return a tuple with the image data and the image name

    def pad_to_max_size(self, image, max_size):
        # Add a channel dimension to the image if not already present
        if image.dim() == 3:
            image = image.unsqueeze(0)  # Shape becomes [1, depth, height, width]

        # Calculate the padding needed to match max_size for each spatial dimension
        padding = []
        for dim in range(3):  # Iterate over depth (D), height (H), and width (W) dimensions
            total_pad = max_size[dim] - image.size(dim + 1)  # +1 to skip the channel dimension
            pad_before = total_pad // 2
            pad_after = total_pad - pad_before
            padding.extend([pad_before, pad_after])

        # Rearrange padding order to match PyTorch's F.pad requirement: [padLeft, padRight, padTop, padBottom, padFront, padBack]
        padding = padding[::-1]  # Reverse to get [padFront, padBack, padTop, padBottom, padLeft, padRight]

        # Apply padding
        padded_image = F.pad(image, padding, "constant", 0)

        return padded_image

    def calculate_max_size(self):
        max_size = [0, 0, 0]  # Initialize max size for Z, Y, X dimensions
        for image_name in self.image_names:
            image = self.image_mapping[image_name]['tensor']
            # Update max size if current image is larger in any dimension
            for dim in range(3):
                if image.size(dim) > max_size[dim]:
                    max_size[dim] = image.size(dim)
        return tuple(max_size)

def export_feature_vectors(model, data_loader, output_csv='./image_feature_vectors.csv'):
    model.eval()  # Set the model to evaluation mode
    feature_vectors = []

    with torch.no_grad():  # No need to track gradients
        for ((data, _), image_names) in data_loader:  # Unpacking structure should match the DataLoader output
            data = data.to(device)
            features = model.forward_features(data)
            features = features.cpu().numpy()  # Move features to CPU and convert to numpy array

            for i, feature_vector in enumerate(features):
                feature_vectors.append([image_names[i]] + feature_vector.tolist())

    # Write to CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image Name'] + [f'Dim {i+1}' for i in range(features.shape[1])])  # Header
        writer.writerows(feature_vectors)

# Initialize the model, optimizer, and loss function
model = Simple3DCNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Create the dataset and data loader
directory = "./Images_Of_Networks/tiff"
image_mapping = create_image_mapping(directory)
csv_file = "./imageNames_ratios.csv"
image_mapping = update_image_mapping_with_ratios(image_mapping, csv_file)
dataset = ImageRatioDataset(image_mapping)
data_loader = DataLoader(dataset, batch_size=4, shuffle=False)

# Train the model
train_model_regression(model, optimizer, criterion, data_loader)

# Call the function after training
export_feature_vectors(model, data_loader)
