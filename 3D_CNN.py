# Maximum dimensions across all files: X: 330, Y: 337, Z: 22
import os
import tifffile
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import csv
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

load_existing_weights = True
weights_path = "3D_CNN.pth"

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
        ratio = row['cc_pixel_intensity_405_488_ratio']
        
        # Update the ratio in the image_mapping if the image name exists
        if image_name in image_mapping:
            image_mapping[image_name]['ratio'] = ratio

    # Filter the image_mapping to keep only those entries that are listed in the CSV file
    new_image_mapping = {key: value for key, value in image_mapping.items() if key in csv_image_names}

    return new_image_mapping

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
    
def train_model_regression(model, optimizer, criterion, data_loader, epochs=10, device=device):
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
        
def train_test_split_dataset(image_mapping, test_size=0.2):
    image_names = list(image_mapping.keys())
    ratios = [image_mapping[name]['ratio'] for name in image_names]
    train_names, test_names, train_ratios, test_ratios = train_test_split(image_names, ratios, test_size=test_size, random_state=42)
    
    train_mapping = {name: image_mapping[name] for name in train_names}
    test_mapping = {name: image_mapping[name] for name in test_names}
    return train_mapping, test_mapping

def train_and_evaluate_model(model, train_loader, test_loader, optimizer, criterion, epochs=10, device=device):
    model.to(device)
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for ((data, ratios), _) in train_loader:
            data, ratios = data.to(device), ratios.to(device).unsqueeze(1).float()
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, ratios)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Evaluate on the test set
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for ((data, ratios), _) in test_loader:
                data, ratios = data.to(device), ratios.to(device).unsqueeze(1).float()
                outputs = model(data)
                loss = criterion(outputs, ratios)
                test_loss += loss.item()
        
        test_loss /= len(test_loader)
        test_losses.append(test_loss)

        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.3f}, Test Loss: {test_loss:.3f}')
        
        # Plotting train and test losses over epochs
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train vs Test Loss')
    plt.legend()
    plt.show()
    
def plot_error_frequency(y_pred, y_test, classifier_title, rounding):
    rounding_ = rounding//2
    
    # Calculate errors
    errors = (abs(y_pred - y_test) < rounding_)

    # Extract labels where errors occurred
    error_labels = y_test[errors]

    # Calculate frequency of each label
    unique_labels, counts = np.unique(error_labels, return_counts=True)

    # Creating a dictionary of label frequencies for errors
    label_error_frequency = dict(zip(unique_labels, counts))

    # Finding the range of all possible labels (from min to max in y_test and y_pred)
    all_labels = np.unique(np.concatenate((y_test, y_pred)))
    min_label = min(all_labels)
    max_label = max(all_labels)

    # Filling in frequencies for labels that did not have errors
    for label in range(min_label, max_label + 1):
        if label not in label_error_frequency:
            label_error_frequency[label] = 0

    # Sorting the dictionary for plotting
    sorted_label_error_frequency = dict(sorted(label_error_frequency.items()))

    xticks = np.arange(min_label, max_label + 1, 50)

    # Plotting the error frequency
    plt.figure(figsize=(10, 6))
    plt.bar(sorted_label_error_frequency.keys(), sorted_label_error_frequency.values())
    plt.xlabel('Prediction')
    plt.ylabel('Error Frequency')
    plt.title('Error Frequency Distribution for ' + str(classifier_title) + ' for errors >= {} ({} in either direction)'.format(rounding, rounding_))
    plt.xticks(xticks, rotation='vertical')  # Set x-ticks to show every 50th label and rotate them vertically
    plt.show()

    # Calculating mean and standard deviation of absolute errors
    absolute_errors = np.abs(y_pred - y_test)
    mean_error = np.mean(absolute_errors)
    std_error = np.std(absolute_errors)
    print("The mean error is", mean_error, "and the standard deviation of the errors is", std_error)

    # Plotting absolute error for each label
    plt.figure(figsize=(10, 6))
    for label in all_labels:
        label_errors = absolute_errors[y_test == label]
        plt.bar([label]*len(label_errors), label_errors, color='r')
    plt.xlabel('Prediction')
    plt.ylabel('Absolute Error')
    plt.title('Absolute Error for Each Label in ' + str(classifier_title))
    plt.xticks(xticks, rotation='vertical')
    plt.show()

def calculate_accuracy(y_test, y_pred, rounding):
    y_test = y_test
    rounding = rounding//2
    # print(y_pred)
    assert(len(y_test) == len(y_pred))
    num_predictions = len(y_test)
    num_correct_predictions = 0
    for i in range(len(y_test)):
        if abs(y_test[i] - y_pred[i]) <= rounding:
            num_correct_predictions += 1
    
    return num_correct_predictions/num_predictions


# Used the plotly library as this is much nicer looking, and the graph is more interactive and works well with Jupyter Notebooks

def plot_absolute_error_plotly(y_pred_model1, y_pred_model2, y_test, model_title1, model_title_2):
    def calculate_error_frequency(y_pred):
        incorrect = y_pred != y_test

        unique_predictions, counts = np.unique(y_pred[incorrect], return_counts=True)
        error_freq = dict(zip(unique_predictions, counts))

        return error_freq

    error_freq_svc = calculate_error_frequency(y_pred_model1)
    error_freq_rf = calculate_error_frequency(y_pred_model2)

    all_labels = np.unique(np.concatenate((y_pred_model1, y_pred_model2)))
    all_labels.sort()

    svc_freq = [error_freq_svc.get(label, 0) for label in all_labels]
    rf_freq = [error_freq_rf.get(label, 0) for label in all_labels]

    fig = go.Figure(data=[
        go.Bar(name=model_title1, x=all_labels, y=svc_freq),
        go.Bar(name=model_title_2, x=all_labels, y=rf_freq)
    ])

    fig.update_layout(
        barmode='group',
        title='Prediction Error Frequency',
        xaxis_title='Predicted Label',
        yaxis_title='Frequency of Incorrect Predictions',
        xaxis=dict(type='category')
    )

    fig.show()


def plot_error_label_frequency_plotly(y_pred_model1, y_pred_model2, y_test, model_title1, model_title2):
    labels = np.unique(y_test)
    errors_svc = [np.sum((y_pred_model1 != y_test) & (y_test == label)) for label in labels]
    errors_rf = [np.sum((y_pred_model2 != y_test) & (y_test == label)) for label in labels]

    bar_width = max(0.5, 30 / len(labels))

    fig = go.Figure(data=[
        go.Bar(name=model_title1, x=labels, y=errors_svc, width=bar_width, marker_color='blue', marker_line_color='blue', marker_line_width=1.5, opacity=1),
        go.Bar(name=model_title2, x=labels, y=errors_rf, width=bar_width, marker_color='red', marker_line_color='red', marker_line_width=1.5, opacity=1)
    ])
    
    fig.update_layout(
        barmode='group',
        title='Error Frequency per Label',
        xaxis_title='Labels',
        yaxis_title='Number of Errors',
        xaxis=dict(tickmode='array', tickvals=labels[::10]),
        plot_bgcolor='rgba(255, 255, 255, 1)',
        font=dict(
            size=12,  # You can adjust the size of the text here for better visibility
        )
    )
    fig.show()

# Data distribution
# Basically plot the frequency of ocurrence of each value
# of the functional parameter in the dataset inputted to the models

# This is only plotting the subset in the training set, and not in the entire dataset

def plot_data_distribution(y):
    # Calculate the frequency of each value in the y_test array
    unique_values, counts = np.unique(y, return_counts=True)

    # Create the bar chart
    fig = go.Figure(data=go.Bar(x=unique_values, y=counts))

    # Update the layout for a better visual representation
    fig.update_layout(
        title='Data Distribution',
        xaxis_title='Unique Values',
        yaxis_title='Frequency',
        xaxis=dict(type='category'),  # Treat unique values as discrete categories
        yaxis=dict(title='Frequency'),  # You can also set the range or scale of the y-axis if needed
    )

    fig.show()
    print("The mean of the data distribution is", np.mean(y), "and the standard deviation is", np.std(y))

# Initialize the model, optimizer, and loss function
model = Simple3DCNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Create the dataset and data loader
directory = "./Images_Of_Networks/tiff"
image_mapping = create_image_mapping(directory)
csv_file = "./imageNames_ratios.csv"
train_csv_file = "./imageNames_ratios_train.csv"
test_csv_file = "./imageNames_ratios_test.csv"
image_mapping = update_image_mapping_with_ratios(image_mapping, csv_file)
dataset = ImageRatioDataset(image_mapping)
# train_mapping, test_mapping = train_test_split_dataset(image_mapping)
train_mapping = update_image_mapping_with_ratios(image_mapping, train_csv_file)
test_mapping = update_image_mapping_with_ratios(image_mapping, test_csv_file)
train_dataset = ImageRatioDataset(train_mapping)
test_dataset = ImageRatioDataset(test_mapping)

# Create data loaders
dataset_loader = DataLoader(dataset, batch_size=4, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

if os.path.exists(weights_path) and load_existing_weights:
    model.load_state_dict(torch.load(weights_path))
    model = model.to(device)
    print("Loaded existing model weights from '3D_CNN.pth'.")
else:
    # Call the training function
    train_and_evaluate_model(model, train_loader, test_loader, optimizer, criterion)

    torch.save(model.state_dict(), weights_path)
    print(f"Model weights saved to '{weights_path}' after training.")

# Proceed with the rest of the work
print("Exporting feature vectors")
export_feature_vectors(model, dataset_loader, "./image_feature_vectors.csv")

print("Proceeding with rest of the graphing...")
y_preds, y_trues = [], []
with torch.no_grad():
    for ((data, ratios), _) in train_loader:
        data = data.to(device)
        outputs = model(data)
        y_preds.extend(outputs.view(-1).cpu().numpy())
        y_trues.extend(ratios.view(-1).cpu().numpy())

y_preds = np.rint(np.array(y_preds)).astype(int)
y_trues = np.rint(np.array(y_trues)).astype(int)

# Use the provided functions to analyze the model's performance
plot_error_frequency(y_preds, y_trues, "3D CNN", 50)
accuracy = calculate_accuracy(y_trues, y_preds, 50)
print(f"Model accuracy on training data (with rounding factor 50): {accuracy*100:.2f}%")
plot_data_distribution(y_trues)