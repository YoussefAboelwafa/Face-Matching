import torch.nn as nn


class AgeModel(nn.Module):
    def __init__(self, embedding_dim=512, num_classes=91, dropout=0.3):
        super(AgeModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),
            
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        
        self.classifier = nn.Linear(64, num_classes)
        self.regressor = nn.Linear(64, 1)

    def forward(self, x):
        features = self.network(x)
        class_output = self.classifier(features)
        reg_output = self.regressor(features).squeeze()
        return class_output, reg_output


class OrdinalLoss(nn.Module):
    def __init__(self, num_classes=91, alpha=0.5):
        super(OrdinalLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, class_output, reg_output, targets):
        ce_loss = self.ce_loss(class_output, targets)
        mse_loss = self.mse_loss(reg_output, targets.float())
        return self.alpha * ce_loss + (1 - self.alpha) * mse_loss
