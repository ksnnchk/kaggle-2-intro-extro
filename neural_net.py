import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size1=128, hidden_size2=64, dropout_prob=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.dropout2 = nn.Dropout(p=dropout_prob)
        self.fc3 = nn.Linear(hidden_size2, 1)

        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = nn.ReLU()(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = nn.ReLU()(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return torch.sigmoid(x)

def prepare_data(tr_X, tr_y, val_X, val_y, batch_size=64):
    X_train = torch.tensor(tr_X.values if hasattr(tr_X, 'values') else tr_X, dtype=torch.float32)
    y_train = torch.tensor(tr_y.values if hasattr(tr_y, 'values') else tr_y, dtype=torch.float32).view(-1, 1)
    X_val = torch.tensor(val_X.values if hasattr(val_X, 'values') else val_X, dtype=torch.float32)
    y_val = torch.tensor(val_y.values if hasattr(val_y, 'values') else val_y, dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader

def calculate_metrics(y_true, y_pred):
    y_pred_class = (y_pred > 0.5).astype(int)
    return {
        'accuracy': accuracy_score(y_true, y_pred_class)
    }

import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.tuner import Tuner

class LightningNeuralNet(pl.LightningModule):
    def __init__(self, input_size, hidden_size1=128, hidden_size2=64, dropout_prob=0.5, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = NeuralNet(input_size, hidden_size1, hidden_size2, dropout_prob)
        self.criterion = nn.BCELoss()

        self.train_acc = torchmetrics.Accuracy(task='binary')
        self.val_acc = torchmetrics.Accuracy(task='binary')
        self.val_precision = torchmetrics.Precision(task='binary')
        self.val_recall = torchmetrics.Recall(task='binary')
        self.val_f1 = torchmetrics.F1Score(task='binary')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        self.train_acc(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        self.val_acc(y_hat, y)
        self.val_precision(y_hat, y)
        self.val_recall(y_hat, y)
        self.val_f1(y_hat, y)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, prog_bar=True)
        self.log('val_precision', self.val_precision, prog_bar=True)
        self.log('val_recall', self.val_recall)
        self.log('val_f1', self.val_f1)

        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                              lr=self.hparams.lr,
                              weight_decay=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=5,
            factor=0.5
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }


train_loader, val_loader = prepare_data(tr_X, tr_y, val_X, val_y)

lightning_model = LightningNeuralNet(input_size=tr_X.shape[1])

trainer = pl.Trainer(
    max_epochs=200,
    accelerator='auto'
)

tuner = Tuner(trainer)

lr_finder = tuner.lr_find(
    lightning_model,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader,
    min_lr=1e-5,
    max_lr=1e+2,
    num_training=200
)

fig = lr_finder.plot(suggest=True)
fig.show()

new_lr = lr_finder.suggestion()
print(f"Recommended learning rate: {new_lr:.2e}")

lightning_model.hparams.lr = new_lr

trainer.fit(lightning_model, train_loader, val_loader)

test1 = test[['Time_spent_Alone',	'Stage_fear',	'Social_event_attendance',	'Going_outside',	'Drained_after_socializing',	'Friends_circle_size',	'Post_frequency']].copy()

for col in ['Time_spent_Alone',	'Social_event_attendance',	'Going_outside',	'Friends_circle_size',	'Post_frequency']:
  test1[col] = test1[col].fillna(test1[col].mean())

for col1 in ['Stage_fear',	'Drained_after_socializing']:
  test1[col1] = test1[col1].apply(lambda x: 0 if x == 'No' else 1 if x == 'Yes' else None)
  test1[col1] = test1[col1].fillna(test1[col1].mode()[0])


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

lightning_model = lightning_model.to(device)
lightning_model.eval()

class TestDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

X_test = torch.tensor(test1.values if hasattr(test1, 'values') else test1,
                     dtype=torch.float32)
test_dataset = TestDataset(X_test)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

predictions = []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        preds = lightning_model(batch)
        predictions.append(preds.cpu())

pers = torch.cat(predictions).numpy().flatten()

final_ie = test[['id']].copy()
final_ie['Personality'] = ['Introvert' if p < 0.5 else 'Extrovert' for p in pers]

final_ie.to_csv('final_ie.csv', index=False)
