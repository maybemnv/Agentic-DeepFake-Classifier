"""
Network wrapper for FaceForensics++ models.
"""
import torch
import torch.nn as nn
from .xception import xception

class TransferModel(nn.Module):
    """
    Simple wrapper to match the FaceForensics++ model structure expected by weights.
    """
    def __init__(self, model_choice, num_out_classes=2, dropout=0.0):
        super(TransferModel, self).__init__()
        self.model_choice = model_choice
        
        if model_choice == 'xception':
            self.model = xception(num_classes=1000, pretrained=None) # We load our own weights
            # Replace the last layer
            num_ftrs = self.model.fc.in_features
            if not dropout:
                self.model.fc = nn.Linear(num_ftrs, num_out_classes)
            else:
                self.model.fc = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(num_ftrs, num_out_classes)
                )
        else:
            raise Exception("Choose valid model, e.g. xception")

    def set_trainable(self, trainable):
        for param in self.parameters():
            param.requires_grad = trainable

    def forward(self, x):
        return self.model(x)

def model_selection(modelname, num_out_classes, dropout=None):
    """
    Factory function to create the model.
    """
    if modelname == 'xception':
        return TransferModel(modelname, num_out_classes, dropout)
    else:
        raise NotImplementedError(modelname)
