import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pickle
from tqdm import tqdm

# paths to the data location

DATA_PATHS = {
    'train_features': '/Users/muskan/Documents/Deep Learning/Project DL/data/train/features.pkl',
    'train_labels': '/Users/muskan/Documents/Deep Learning/Project DL/data/train/labels.pkl',
    'dev_features': '/Users/muskan/Documents/Deep Learning/Project DL/data/dev/features.pkl',
    'dev_labels': '/Users/muskan/Documents/Deep Learning/Project DL/data/dev/labels.pkl',
    'test_features': '/Users/muskan/Documents/Deep Learning/Project DL/data/test1/features.pkl'
}

CONFIG = {
    'input_dim': 180,        
    'hidden_dim': 128,
    'num_layers': 2,
    'dropout': 0.3,
    'batch_size': 32,
    'num_epochs': 20,
    'lr': 0.001,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# Dataset

class AudioDataset(Dataset):
    def __init__(self, features_file, labels_file=None):
        with open(features_file, 'rb') as f:
            self.features_df = pd.read_pickle(f)
        
        if labels_file:
            with open(labels_file, 'rb') as f:
                labels_df = pd.read_pickle(f)
            self.data = pd.merge(self.features_df, labels_df, on='uttid')
            self.has_labels = True
        else:
            self.data = self.features_df
            self.has_labels = False
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        uttid = row['uttid']
        features = row['features']
        
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)
        
        if self.has_labels:
            label = torch.tensor(row['label'], dtype=torch.float32)
            return uttid, features, label
        return uttid, features

def collate_fn(batch):
    """Padding sequences to same length in batch"""
    if len(batch[0]) == 3:
        uttids, features, labels = zip(*batch)
        labels = torch.stack(labels)
    else:
        uttids, features = zip(*batch)
        labels = None
    
    # Pad to max length
    max_len = max(f.shape[1] for f in features)
    feature_dim = features[0].shape[0]
    
    padded = []
    for f in features:
        if f.shape[1] < max_len:
            pad = torch.zeros(feature_dim, max_len - f.shape[1])
            f = torch.cat([f, pad], dim=1)
        padded.append(f)
    
    padded = torch.stack(padded)
    
    if labels is not None:
        return uttids, padded, labels
    return uttids, padded


# Model

class DeepfakeDetector(nn.Module):
    def __init__(self, input_dim=180, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        # x shape: [batch, features, time] -> [batch, time, features]
        x = x.transpose(1, 2)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Attention
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        attended = torch.sum(lstm_out * attn_weights, dim=1)
        
        # Classify
        output = self.classifier(attended)
        return output.squeeze(-1)


#Training

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for uttids, features, labels in pbar:
        features = features.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.1f}%'})
    
    return total_loss / len(dataloader), 100 * correct / total

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for uttids, features, labels in tqdm(dataloader, desc="Evaluating"):
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    return total_loss / len(dataloader), 100 * correct / total

def predict(model, dataloader, device):
    model.eval()
    predictions = []
    uttids_list = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            if len(batch) == 3:
                uttids, features, _ = batch
            else:
                uttids, features = batch
            
            features = features.to(device)
            outputs = model(features)
            probs = torch.sigmoid(outputs)
            
            predictions.extend(probs.cpu().numpy())
            uttids_list.extend(uttids)
    
    return uttids_list, predictions



def main():
    print("="*60)
    print("Audio Deepfake Detection - Data Training")
    print("="*60)
    print(f"Device: {CONFIG['device']}")
    print(f"Input dimensions: {CONFIG['input_dim']}")
    print(f"Batch size: {CONFIG['batch_size']}")
    print(f"Epochs: {CONFIG['num_epochs']}")
    print("="*60)
    
    # Loading data
    print("\nLoading data...")
    train_dataset = AudioDataset(DATA_PATHS['train_features'], DATA_PATHS['train_labels'])



    dev_dataset = AudioDataset(DATA_PATHS['dev_features'], DATA_PATHS['dev_labels'])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                             shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=CONFIG['batch_size'], 
                           shuffle=False, collate_fn=collate_fn)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Dev samples: {len(dev_dataset)}")
    
    #Creating model

    model = DeepfakeDetector(
        input_dim=CONFIG['input_dim'],
        hidden_dim=CONFIG['hidden_dim'],
        num_layers=CONFIG['num_layers'],
        dropout=CONFIG['dropout']
    ).to(CONFIG['device'])
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0
    
    print("\nStarting training...\n")
    for epoch in range(CONFIG['num_epochs']):
        print(f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
        print("-" * 60)
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, CONFIG['device'])
        val_loss, val_acc = evaluate(model, dev_loader, criterion, CONFIG['device'])
        
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), 'best_model.pt')
            print("✅ New best model saved!")
        print()
    
    print("="*60)
    print(f"Training completed!")
    print(f"Best epoch: {best_epoch+1}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("="*60)
    
    # Making predictions on test set
    print("\nMaking predictions on test set...")
    model.load_state_dict(torch.load('best_model.pt'))
    
    test_dataset = AudioDataset(DATA_PATHS['test_features'], labels_file=None)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], 
                            shuffle=False, collate_fn=collate_fn)
    
    uttids, predictions = predict(model, test_loader, CONFIG['device'])
    
    # Saving predictions
    predictions_df = pd.DataFrame({
        'uttid': uttids,
        'predictions': predictions
    })

    print(f"{predictions_df.head()}")
    
    with open('predictions.pkl', 'wb') as f:
        pickle.dump(predictions_df, f)
    
    print(f"\n✅ Predictions saved to 'predictions.pkl'")
    print(f"Number of predictions: {len(predictions_df)}")
    print("\n" + "="*60)
    print("Next steps:")
    print("1. Evaluate: python scripts/evaluation.py predictions.pkl", DATA_PATHS['dev_labels'])
    print("2. Submit: python scripts/generate_submission.py", DATA_PATHS['test_features'], "predictions.pkl st196668 Muskan Verma mvx")
    print("="*60)

if __name__ == "__main__":
    main()
