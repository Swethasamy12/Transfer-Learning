# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for classification using VGG-19 architecture.
## Problem Statement and Dataset
Image classification from scratch requires a huge dataset and long training times. To overcome this, transfer learning can be applied using pre-trained models like VGG-19, which has already learned feature representations from a large dataset (ImageNet).

Problem Statement: Build an image classifier using VGG-19 pre-trained architecture, fine-tuned for a custom dataset (e.g., CIFAR-10, Flowers dataset, or any small image dataset). Dataset: A dataset consisting of multiple image classes (e.g., train, test, and validation sets). For example, CIFAR-10 (10 classes of small images) or a custom dataset with multiple classes.

## DESIGN STEPS
STEP 1:
Import required libraries, load the dataset, and define training & testing datasets.

STEP 2:
Initialize the model, loss function, and optimizer. Use CrossEntropyLoss for multi-class classification and Adam optimizer for efficient training.

STEP 3:
Train the model using the training dataset with forward and backward propagation.

STEP 4:
Evaluate the model on the testing dataset to measure accuracy and performance.

STEP 5:
Make predictions on new data using the trained model

## PROGRAM
Include your code here
```
# Load Pretrained Model and Modify for Transfer Learning

from torchvision.models import VGG19_Weights
model=models.vgg19(weights=VGG19_Weights.DEFAULT)

# Modify the final fully connected layer to match the dataset classes

num_classes = len(train_dataset.classes)
model.classifier[6] = nn.Linear(4096, num_classes)

# Include the Loss function and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier[-1].parameters(), lr=0.001)

# Train the model

def train_model(model, train_loader,test_loader,num_epochs=10):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # Compute validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_losses.append(val_loss / len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')
    # Plot training and validation loss
    print("Name: SWETHA C")
    print("Register Number: 212224230283")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

```

## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot

<img width="730" height="722" alt="Screenshot 2026-03-24 121957" src="https://github.com/user-attachments/assets/519ec98e-2abd-497a-ae7a-619207683ca5" />


### Confusion Matrix
<img width="656" height="768" alt="Screenshot 2026-03-24 122011" src="https://github.com/user-attachments/assets/21ce4a6e-65e9-4167-a353-f24f68fe8e28" />


### Classification Report
<img width="406" height="378" alt="Screenshot 2026-03-24 122023" src="https://github.com/user-attachments/assets/da2457cf-6dd8-4506-a55e-0170b2157a73" />


### New Sample Prediction

<img width="410" height="375" alt="Screenshot 2026-03-24 122037" src="https://github.com/user-attachments/assets/67007b33-5576-4ddd-9399-be3523e979ec" />

## RESULT
Thus, the Transfer Learning for classification using the VGG-19 architecture has been successfully implemented.
