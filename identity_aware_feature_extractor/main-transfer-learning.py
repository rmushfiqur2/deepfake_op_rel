import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.partially_freezed_resnet import get_resnet
from utils.losses import ContrastiveLossWithMargin, EuclideanLoss
from datasets.base_unsupervised_deepfake import DatasetBaseUnsupervised_deepfake
from datasets.unsupervised_dataset_deepfake import MultiCropDatasetDeepfake
import numpy as np

from tensorboardX import SummaryWriter

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device(
    'cpu')

my_model = get_resnet(device)
my_model.load_state_dict(torch.load('../finetuned-backbone.pth'))
my_model.to(device)

# Step 1: Prepare the dataset
# Create an instance of your custom dataset
deepfake_compact_dataset = DatasetBaseUnsupervised_deepfake(
    '../compact-train-swapped/')  # 4 elements
#deepfake_compact_dataset = DatasetBaseUnsupervised_deepfake_idempotence('data_deepfake/compact-test-swapped/') 6 elements
# only one crop is supported
resized_augmented_dataset = MultiCropDatasetDeepfake(
    deepfake_compact_dataset, [224], [1], [0.8], [1.0],
    augment=True,
    consistent_crop_augment=True)

# Step 3: Specify loss function and optimizer
criterion = EuclideanLoss()  # Mean squared error loss
criterion_contrast = ContrastiveLossWithMargin(50)
optimizer = optim.SGD(my_model.parameters(),
                      lr=0.001)  # Stochastic Gradient Descent

# Step 4: Training loop
num_epochs = 1500

# Create a DataLoader
batch_size = 32
dataloader = DataLoader(resized_augmented_dataset,
                        batch_size=batch_size,
                        shuffle=True)

writer = SummaryWriter('logs')
prev_loss = 1e9
# Iterate over the DataLoader
for epoch_num in range(num_epochs):
    epoch_losses = []
    distances = []
    losses_teacher_raw = []
    losses_teacher_recons = []
    for i, (batch_raw_im, batch_recons_im, batch_raw_features,
            batch_recons_features) in enumerate(dataloader):
        # Process the batch_data and batch_labels here
        #print(f"number of crops: {len(batch_raw_im)}")
        #print(f"Batch data shape: {batch_raw_im[0].shape}, Batch labels shape: {torch.squeeze(batch_raw_features).shape}")
        batch_raw_features = torch.squeeze(batch_raw_features)
        batch_recons_features = torch.squeeze(batch_recons_features)
        out_feature_raw = my_model(batch_raw_im[0].to(device))  # single crops
        out_feature_recons = my_model(
            batch_recons_im[0].to(device))  # # single crops
        #loss_raw_teacher_net = 0#criterion(out_feature_raw, batch_raw_features)
        #loss_recons_teacher_net = 0#criterion(out_feature_recons, batch_recons_features)
        #active_loss = loss_raw_teacher_net + loss_recons_teacher_net
        loss_contrastive, euclidean_distance = criterion_contrast(
            out_feature_raw, out_feature_recons)
        active_loss = loss_contrastive
        epoch_losses.append(active_loss.item())
        distances.append(euclidean_distance.item())
        #losses_teacher_raw.append(loss_raw_teacher_net.item())
        #losses_teacher_recons.append(loss_recons_teacher_net.item())

        # Backward pass and optimization
        optimizer.zero_grad()
        active_loss.backward()
        optimizer.step()

    if np.mean(epoch_losses) < prev_loss:
        torch.save(my_model.state_dict(), '../finetuned-backbone.pth')
        prev_loss = np.mean(epoch_losses)

    # Log scalars
    writer.add_scalar('Loss', np.mean(epoch_losses), epoch_num)
    writer.add_scalar('Contrastive distance', np.mean(distances), epoch_num)
    # Print the loss at the end of each epoch
    print(
        f'Epoch [{epoch_num + 1}/{num_epochs}], Loss: {np.mean(epoch_losses)}, Distance: {np.mean(distances)}'
    )
    #print(f'Epoch [{epoch_num + 1}/{num_epochs}], Loss1: {np.mean(losses_teacher_raw)}, Loss2: {np.mean(losses_teacher_recons)}, Distance: {np.mean(distances)}')

writer.close()
