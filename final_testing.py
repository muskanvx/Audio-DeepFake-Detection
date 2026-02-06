import torch
import pandas as pd
import pickle
from train_real_data import DeepfakeDetector, AudioDataset, collate_fn, predict
from torch.utils.data import DataLoader

# Load your trained model
model = DeepfakeDetector(input_dim=180)
model.load_state_dict(torch.load('best_model.pt'))

# Load FINAL test data
final_test_dataset = AudioDataset(
    '/Users/muskan/Documents/Deep Learning/Project DL/data/final_test.pkl',
    labels_file=None
)
final_test_loader = DataLoader(final_test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# Predict
print('Making predictions on final test set:')
uttids, preds = predict(model, final_test_loader, 'cpu')

# Saving
df = pd.DataFrame({'uttid': uttids, 'predictions': preds})
with open('predictions_final.pkl', 'wb') as f:
    pickle.dump(df, f)
    
print(f'✅ Saved predictions_final.pkl')
print(f'Total predictions: {len(uttids)}')
print(f'First 10 uttids: {uttids[:10]}')
