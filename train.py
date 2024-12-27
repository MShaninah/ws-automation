import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import string

# Enable benchmark mode for optimized training
torch.backends.cudnn.benchmark = True

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Define the CAPTCHA dataset
class CaptchaDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(".png")]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert("RGB")

        label = self.image_files[idx].split("_")[0]  # Extract label from filename
        length = len(label)  # Get the length of the label

        if self.transform:
            image = self.transform(image)

        return image, label, length

# Define the CNN model
class CaptchaModel(nn.Module):
    def __init__(self, max_chars, num_classes):
        super(CaptchaModel, self).__init__()

        from torchvision.models import ResNet34_Weights
        self.cnn = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        self.cnn.fc = nn.Sequential(
            nn.Linear(self.cnn.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, max_chars * num_classes)
        )
        self.max_chars = max_chars
        self.num_classes = num_classes

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1, self.max_chars, self.num_classes)
        return x

# Define a function to calculate accuracy
def calculate_accuracy(output, labels, lengths):
    predictions = output.argmax(dim=2)
    correct = 0
    total = 0

    for i in range(len(labels)):
        pred_label = ''.join([idx_to_char[predictions[i, j].item()] for j in range(lengths[i])])
        if pred_label == labels[i]:
            correct += 1
        total += 1

    return correct / total

# Define a function to calculate per-character accuracy
def calculate_character_accuracy(output, labels, lengths):
    predictions = output.argmax(dim=2)
    correct_chars = 0
    total_chars = 0

    for i, length in enumerate(lengths):
        for j in range(length):
            if predictions[i, j].item() == char_to_idx[labels[i][j]]:
                correct_chars += 1
            total_chars += 1

    return correct_chars / total_chars

# Preprocessing and dataset setup
transform = transforms.Compose([
    transforms.Resize((90, 280)),  # Match the generated CAPTCHA dimensions
    transforms.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.1),  # Increased variability
    transforms.RandomAffine(
        degrees=60,  # Match rotation range from CAPTCHA generator
        translate=(0.3, 0.3),  # Match translation variability
        scale=(0.6, 1.4),  # Match scaling variability
        shear=25  # Match shearing variability
    ),
    transforms.GaussianBlur(kernel_size=3),  # Add random blurring
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values
])

train_dataset = CaptchaDataset("generated_captchas/train", transform=transform)
val_dataset = CaptchaDataset("generated_captchas/val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Character set and mappings
characters = string.ascii_letters + string.digits
char_to_idx = {char: idx for idx, char in enumerate(characters)}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}

# Model, loss, and optimizer setup
max_chars = 6  # Max length of CAPTCHA
num_classes = len(characters)

model = CaptchaModel(max_chars=max_chars, num_classes=num_classes).to(device)

# Initialize the fully connected layer
for m in model.cnn.fc:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

# Optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)  # Reduced learning rate
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

# Loss function
criterion = nn.CrossEntropyLoss()

# Load the checkpoint if it exists
checkpoint_path = "captcha_solver_final_checkpoint.pth"
start_epoch = 0
if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = int(checkpoint.get('epoch', 0))
    print(f"Resumed from epoch {start_epoch + 1}.")
else:
    print("No checkpoint found. Starting training from scratch.")

# Early stopping parameters
early_stop_tolerance = 50
best_val_accuracy = 0.0
no_improvement_epochs = 0

# Training loop
num_epochs = 50  # Increased epochs
for epoch in range(start_epoch, num_epochs):
    model.train()
    running_loss = 0.0
    for i, (images, labels, lengths) in enumerate(train_loader):
        images = images.to(device)

        # Encode labels as indices dynamically based on lengths
        max_length = max(lengths)
        encoded_labels = torch.zeros(len(labels), max_length, dtype=torch.long).to(device)
        for i, (label, length) in enumerate(zip(labels, lengths)):
            for j, char in enumerate(label[:length]):
                encoded_labels[i, j] = char_to_idx[char]

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type):
            outputs = model(images)
            outputs = outputs[:, :max_length, :]
            loss = criterion(outputs.reshape(-1, num_classes), encoded_labels.reshape(-1))

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Debugging: Print sample labels and outputs
        if epoch == start_epoch and i == 0:
            print(f"Sample encoded labels: {encoded_labels[0]}")

    scheduler.step()
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")

    # Validation
    model.eval()
    with torch.no_grad():
        val_accuracy = 0.0
        char_accuracy = 0.0
        for images, labels, lengths in val_loader:
            images = images.to(device)
            outputs = model(images)
            val_accuracy += calculate_accuracy(outputs, labels, lengths)
            char_accuracy += calculate_character_accuracy(outputs, labels, lengths)

            # Debug: Print some predictions during validation
            predictions = outputs.argmax(dim=2)
            for i, length in enumerate(lengths):
                predicted_label = ''.join([idx_to_char[predictions[i, j].item()] for j in range(length)])
                print(f"True: {labels[i]}, Predicted: {predicted_label}")

        val_accuracy /= len(val_loader)
        char_accuracy /= len(val_loader)
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Character Accuracy: {char_accuracy:.4f}")

    # Early stopping check
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        no_improvement_epochs = 0
        print("Validation accuracy improved. Saving model...")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch + 1
        }, "captcha_solver_final_checkpoint.pth")
    else:
        no_improvement_epochs += 1
        print(f"No improvement in validation accuracy for {no_improvement_epochs} epochs.")

    if no_improvement_epochs >= early_stop_tolerance:
        print("Early stopping triggered.")
        break

print("Training complete.")

# Load the trained model and optimizer (example usage)
def load_trained_model_and_optimizer(model_path):
    checkpoint = torch.load(model_path, map_location=device)

    # Load model state
    model = CaptchaModel(max_chars=max_chars, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Load optimizer state
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer

# Example: Loading and using the model and optimizer
# trained_model, trained_optimizer = load_trained
