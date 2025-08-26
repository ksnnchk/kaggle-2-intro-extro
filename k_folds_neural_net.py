import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size1=64, dropout_prob=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(hidden_size1, 1)

        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = nn.ReLU()(x)
        x = self.dropout1(x)
        x = self.fc2(x)
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



import torch
import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer
import torchmetrics

class BigFivePersonalityPredictor(pl.LightningModule):
    def __init__(self, num_features, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.bigfive_net = AutoModel.from_pretrained(
            "Nasserelsaman/microsoft-finetuned-personality",
            output_hidden_states=True
        )

        for param in self.bigfive_net.parameters():
            param.requires_grad = True

        self.neural_net = NeuralNet(
            input_size=num_features,
            hidden_size1=64,
            dropout_prob=0.5
        )


        self.train_acc = torchmetrics.Accuracy(task='binary')
        self.val_acc = torchmetrics.Accuracy(task='binary')
        self.val_f1 = torchmetrics.F1Score(task='binary')

    def forward(self, x):

        return self.neural_net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.BCELoss()(y_hat, y.float())

        self.train_acc(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.BCELoss()(y_hat, y.float())

        self.val_acc(y_hat, y)
        self.val_f1(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, prog_bar=True)
        self.log('val_f1', self.val_f1, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=1e-4,
            weight_decay=0.01
        )
        return optimizer

from huggingface_hub import notebook_login
notebook_login()

import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import Subset, DataLoader
import torchmetrics
from pytorch_lightning.tuner import Tuner

k_folds = 15
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

X = torch.tensor(train1.values if hasattr(train1, 'values') else train1, dtype=torch.float32)
y = torch.tensor(y0.values if hasattr(y0, 'values') else y, dtype=torch.float32).view(-1, 1)
dataset = TensorDataset(X, y)

fold_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    print(f"\nFold {fold + 1}/{k_folds}")
    print("-" * 50)

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=64)

    model = BigFivePersonalityPredictor(num_features=X.shape[1])

    trainer = pl.Trainer(
        max_epochs=100,
        callbacks=[
            pl.callbacks.EarlyStopping(monitor='val_loss', patience=7),
            pl.callbacks.ModelCheckpoint(
                monitor='val_f1',
                filename=f'best_model_fold_{fold}',
                mode='max'
            )
        ],
        accelerator='auto',
        enable_progress_bar=True,
        logger=pl.loggers.CSVLogger(save_dir="logs", name=f"fold_{fold}"),
    )

    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(model, train_loader, val_loader)
    model.hparams.lr = lr_finder.suggestion()
    print(f"Suggested LR for fold {fold}: {model.hparams.lr:.2e}")

    trainer.fit(model, train_loader, val_loader)

    val_metrics = trainer.validate(model, val_loader)[0]
    fold_results.append(val_metrics)

    best_model_path = trainer.checkpoint_callback.best_model_path
    print(f"Best model saved at: {best_model_path}")

print("\nFinal Results:")
for i, metrics in enumerate(fold_results):
    print(f"Fold {i + 1}:")
    print(f"Val F1: {metrics['val_f1']:.4f}, Val Acc: {metrics['val_acc']:.4f}")
    print("-" * 40)

avg_f1 = np.mean([m['val_f1'] for m in fold_results])
avg_acc = np.mean([m['val_acc'] for m in fold_results])
print(f"\nAverage Val F1: {avg_f1:.4f}")
print(f"Average Val Acc: {avg_acc:.4f}")


test1 = test[['Time_spent_Alone',	'Stage_fear',	'Social_event_attendance',	'Going_outside',	'Drained_after_socializing',	'Friends_circle_size',	'Post_frequency']].copy()

for col in ['Time_spent_Alone',	'Social_event_attendance',	'Going_outside',	'Friends_circle_size',	'Post_frequency']:
  test1[col] = test1[col].fillna(test1[col].mean())

for col1 in ['Stage_fear',	'Drained_after_socializing']:
  test1[col1] = test1[col1].apply(lambda x: 0 if x == 'No' else 1 if x == 'Yes' else None)
  test1[col1] = test1[col1].fillna(test1[col1].mode()[0])


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

best_model_path = trainer.checkpoint_callback.best_model_path
model = BigFivePersonalityPredictor.load_from_checkpoint(best_model_path, num_features=X.shape[1])
model = model.to(device)
model.eval()

class TestDataset(torch.utils.data.Dataset):
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
        preds = model(batch)
        predictions.append(preds.cpu())

pers = torch.cat(predictions).numpy().flatten()

final_ie = test[['id']].copy()
final_ie['Personality'] = ['Introvert' if p < 0.5 else 'Extrovert' for p in pers]

final_ie.to_csv('final_ie.csv', index=False)
