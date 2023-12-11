import torch
import os
from torch import nn, optim
from dataloader import CalligraphyDataset, transform
from model import Generator, Discriminator
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from pathlib import Path
from math import log10
import matplotlib.pyplot as plt
import numpy as np
from piq import ssim
# Parameters
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
device = torch.device("cuda:0")
seed = 777
torch.cuda.manual_seed(seed)

batch_size = 64
lr = 0.0001
beta1 = 0.5
epochs = 1000
smooth = 0.0

# setting up the logging
datetime_str = datetime.now().strftime('%b%d_%H-%M-%S')
logging_dir = None
if logging_dir is None:
    runs_dir = Path("./") / Path(f"runs/")
    runs_dir.mkdir(exist_ok=True)
    logging_dir = runs_dir / Path("semi-cursive")
    logging_dir.mkdir(exist_ok=True)
    logging_dir = str(logging_dir.absolute())

writer = SummaryWriter(log_dir=logging_dir)


def save_image(partial_images, inpainted_images, original_images, filepath, epoch):
    current_batch_size = original_images.size(0)
    idx = np.random.randint(0, current_batch_size)
    fig, ax = plt.subplots(1, 3, figsize=(24, 8))
    ax[0].imshow(partial_images[idx].detach().cpu().numpy().squeeze(), vmin=0, vmax=1, cmap='gray')
    ax[0].title.set_text('Partial Img')
    ax[0].title.set_fontsize(28)
    ax[0].axis('off')
    ax[1].imshow(inpainted_images[idx].detach().cpu().numpy().squeeze(), vmin=0, vmax=1, cmap='gray')
    ax[1].title.set_text('Inpaint Pred')
    ax[1].title.set_fontsize(28)
    ax[1].axis('off')
    ax[2].imshow(original_images[idx].detach().cpu().numpy().squeeze(), vmin=0, vmax=1, cmap='gray')
    ax[2].title.set_text('GT Img')
    ax[2].title.set_fontsize(28)
    ax[2].axis('off')
    plt.savefig(os.path.join(filepath, f'epoch_{epoch}.png'))
    plt.close()


##### visilization folder
os.makedirs('saved_images', exist_ok=True)
os.makedirs('test_results', exist_ok=True)

# Initialize the dataset and dataloader
train_ratio = 0.95
dataset = CalligraphyDataset(calligraphy_type='semi-cursive', transform=transform)

train_size = int(train_ratio * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print(len(train_dataloader), len(test_dataloader))


# Initialize models
g_model = Generator().to(device)
d_model = Discriminator().to(device)

# Loss function
# criterion = nn.BCEWithLogitsLoss()
criterion = nn.BCELoss()
mseloss = torch.nn.MSELoss(reduction='mean')

# Optimizers
optimizer_g = optim.Adam(g_model.parameters(), lr=lr)  #, betas=(beta1, 0.999))
optimizer_d = optim.Adam(d_model.parameters(), lr=lr)  #, betas=(beta1, 0.999))


def test_model(epoch, g_model, test_dataloader, writer):
    g_model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            original_images = batch["original_image"].to(device)
            partial_images = batch["partial_image"].to(device)

            inpainted_images = g_model(partial_images)

            # print(original_images.shape, partial_images.shape, inpainted_images.shape)

            loss = mseloss(inpainted_images, original_images)
            psnr = 10 * log10(1 /loss.item())
            ssim_v = ssim(inpainted_images.clamp(0.0, 1.0), original_images)

            writer.add_scalar('Test/Loss', loss.item(), epoch * len(test_dataloader) + i)
            writer.add_scalar('Test/PSNR', psnr, epoch * len(test_dataloader) + i)
            writer.add_scalar('Test/SSIM', ssim_v.item(), epoch * len(test_dataloader) + i)

        save_image(partial_images, inpainted_images, original_images, 'test_results', epoch)


# Training loop
for epoch in range(epochs):
    g_model.train()
    d_model.train()

    print(f"Starting epoch {epoch+1}/{epochs}...")
    for i, batch in enumerate(train_dataloader):
        original_images = batch["original_image"].to(device)
        partial_images = batch["partial_image"].to(device)

        current_batch_size = original_images.size(0)
        
        # Prepare labels for real and fake images
        real_labels = torch.full((current_batch_size,), 1 - smooth, device=device)
        fake_labels = torch.full((current_batch_size,), smooth, device=device)

        # Train Discriminator with real images
        for d_parameters in d_model.parameters():
            d_parameters.requires_grad = True

        d_model.zero_grad()
        outputs_real = d_model(original_images).squeeze()
        # print(outputs_real.shape, real_labels.shape)
        loss_real = criterion(outputs_real, real_labels)
        loss_real.backward()

        # Generate inpainted images
        inpainted_images = g_model(partial_images)
        outputs_fake = d_model(inpainted_images.detach()).squeeze()
        loss_fake = criterion(outputs_fake, fake_labels)
        loss_fake.backward()

        # Combine and backpropagate for discriminator
        loss_d = loss_real + loss_fake
        optimizer_d.step()

        # Train Generator
        for d_parameters in d_model.parameters():
            d_parameters.requires_grad = False

        g_model.zero_grad()

        outputs_fake_for_gen = d_model(inpainted_images).squeeze()
        loss_g1 = criterion(outputs_fake_for_gen, real_labels)

        output = g_model(partial_images)
        loss_g2 = mseloss(output, original_images)

        loss_g = loss_g1 * 0.01 + loss_g2
        loss_g.backward()
        optimizer_g.step()

        print(f"Epoch {epoch+1}, Batch {i+1}/{len(train_dataloader)}, D_loss: {loss_d.item():.4f}, G_loss: {loss_g.item():.4f}")

        ### log in tensorboard
        if i % 10 == 0:
            iters = epoch * len(train_dataloader) + i
            writer.add_scalar("Train/D_Loss", loss_d.item(), iters)
            writer.add_scalar("Train/G_Loss", loss_g.item(), iters)


    ###### save training images #####
    save_image(partial_images, inpainted_images, original_images, 'saved_images', epoch)

    test_model(epoch, g_model, test_dataloader, writer)



    print(f"Epoch {epoch+1} completed.")

    print("Training completed. Saving models...")
    torch.save(g_model.state_dict(), 'g_model.pth')
    torch.save(d_model.state_dict(), 'd_model.pth')
    print("Models saved.")


