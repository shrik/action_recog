from action_model import ActionModel
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from action_dataset import load_dataset, ActionDataset


NUM_FRAMES = 5

# Training function
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = running_loss / len(train_loader)
        accuracy = 100. * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
        evaluate_model(model, test_loader, device)
        torch.save(model, f'models/model_{epoch}.pth')
    return model

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    false_positive = 0
    false_negative = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Calculate false positives and negatives
            false_positive += ((predicted == 1) & (labels == 0)).sum().item()
            false_negative += ((predicted == 0) & (labels == 1)).sum().item()
            
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    
    accuracy = 100. * correct / total
    epoch_loss = running_loss / len(test_loader)
    precision = 100. * correct / (correct + false_positive) if (correct + false_positive) > 0 else 0
    recall = 100. * correct / (correct + false_negative) if (correct + false_negative) > 0 else 0
    
    print(f'Test Loss: {epoch_loss:.4f}, '
          f'Accuracy: {accuracy:.2f}%, Precision: {precision:.2f}%, '
          f'Recall: {recall:.2f}%')
    
# Example usage:
if __name__ == "__main__":
    CROP_SIZE = 224
    device = torch.device("cuda")
    # model = ActionModel(num_frames=NUM_FRAMES).to(device)
    model = torch.load(f'models/model_49.pth').to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    configfile = 'config.yml'
    train_data, train_labels, test_data, test_labels = load_dataset(configfile, num_frames=NUM_FRAMES)

    train_loader = DataLoader(ActionDataset(train_data, train_labels, crop_size=CROP_SIZE), batch_size=16, shuffle=True)
    test_loader = DataLoader(ActionDataset(test_data, test_labels, crop_size=CROP_SIZE), batch_size=16, shuffle=False)
    
    model = train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=50, device=device)
    # evaluate_model(model, test_loader, device)

