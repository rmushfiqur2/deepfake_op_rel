import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.partially_freezed_resnet import get_resnet
from utils.losses import ContrastiveLossWithMargin, EuclideanLoss
from datasets.base_unsupervised_deepfake import DatasetBaseUnsupervised_deepfake
from datasets.unsupervised_dataset_deepfake import MultiCropDatasetDeepfake


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

my_model = get_resnet(device)
my_model.load_state_dict(torch.load('finetuned-backbone.pth'))
my_model.to(device)

# Step 1: Prepare the dataset
# Create an instance of your custom dataset
deepfake_compact_dataset = DatasetBaseUnsupervised_deepfake('data_deepfake/compact-test/')
resized_augmented_dataset = MultiCropDatasetDeepfake(deepfake_compact_dataset,[224],[1],[0.8],[1.0])

# Step 3: Specify loss function and optimizer
criterion = EuclideanLoss()  # Mean squared error loss
criterion_contrast = ContrastiveLossWithMargin(50)
# optimizer = optim.SGD(my_model.parameters(), lr=0.001)  # Stochastic Gradient Descent

# Create a DataLoader
batch_size = 32
dataloader = DataLoader(resized_augmented_dataset, batch_size=batch_size, shuffle=False)

# Iterate over the DataLoader
for i, (batch_raw_im, batch_recons_im, batch_raw_features, batch_recons_features) in enumerate(dataloader):
    # Process the batch_data and batch_labels here
    #print(f"number of crops: {len(batch_raw_im)}")
    #print(f"Batch data shape: {batch_raw_im[0].shape}, Batch labels shape: {torch.squeeze(batch_raw_features).shape}")
    batch_raw_features = torch.squeeze(batch_raw_features)
    batch_recons_features = torch.squeeze(batch_recons_features)
    out_feature_raw = my_model(batch_raw_im[0].to(device)) # single crops
    out_feature_recons = my_model(batch_recons_im[0].to(device)) # # single crops
    loss_raw_teacher_net = criterion(out_feature_raw, batch_raw_features)
    loss_recons_teacher_net = criterion(out_feature_recons, batch_recons_features)
    loss_contrastive, euclidean_distance = criterion_contrast(out_feature_raw, out_feature_recons)