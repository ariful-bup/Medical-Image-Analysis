import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FeedbackDataset(Dataset):
    """Dataset for fine-tuning using feedback data."""
    def __init__(self, feedback_folder, transform=None):
        self.feedback_folder = feedback_folder
        self.transform = transform
        self.samples = self._load_samples()
    
    def _load_samples(self):
        samples = []
        for label in ["positive", "negative"]:
            label_folder = os.path.join(self.feedback_folder, label)
            if not os.path.exists(label_folder):
                continue
            
            for filename in os.listdir(label_folder):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    image_path = os.path.join(label_folder, filename)
                    label_id = 1 if label == "positive" else 0  # 1 for positive, 0 for negative
                    samples.append((image_path, label_id))
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def load_model(model_path='resnet50_finetuned.pth'):
    """Load the pretrained ResNet50 model."""
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # 2 classes: positive and negative
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded from {model_path}")
    except:
        print("No pretrained model found. Using initial model.")
    
    model = model.to(device)
    model.eval()
    return model

def get_transform():
    """Get the transformation pipeline for images."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def predict(model, image):
    """Make prediction for a single image."""
    transform = get_transform()
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    return {
        'class': 'Positive' if predicted.item() == 1 else 'Negative',
        'confidence': confidence.item() * 100
    }

def fine_tune_model(model, feedback_folder, epochs=20, batch_size=16, lr=0.001):
    """Fine-tune the model using feedback data."""
    model.train()  # Set model to training mode
    
    # Load feedback dataset
    dataset = FeedbackDataset(feedback_folder, transform=get_transform())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Fine-tuning loop
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader):.4f}")
    
    # Save the fine-tuned model
    torch.save(model.state_dict(), 'resnet50_.pth')
    print("Fine-tuning complete. Model saved.")