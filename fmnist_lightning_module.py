from pathlib import Path
from lightning.pytorch import LightningModule
import timm
from torch.optim import Adam
from torch import nn
from torchmetrics import Precision, Recall
import torch
from lightning.pytorch import Trainer

class FMNISTLightningModule(LightningModule):
    """A LightningModule for Fashion MNIST classification using mobilenetv3_small"""
    def __init__(self):
        """Initialize the model, loss function, and metrics."""
        super().__init__()
        self.model = timm.create_model('mobilenetv3_small_100', pretrained=False, num_classes=10, in_chans=1)
        self.loss = nn.CrossEntropyLoss()
        self.val_precision = Precision(task="multiclass", num_classes=10, average='macro')
        self.val_recall = Recall(task="multiclass", num_classes=10, average='macro')
        self.test_precision = Precision(task="multiclass", num_classes=10, average='macro')
        self.test_recall = Recall(task="multiclass", num_classes=10, average='macro')

    def training_step(self, batch, batch_idx):
        logits = self.model(batch[0])
        loss = self.loss(logits, batch[1])
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        logits = self.model(batch[0])
        loss = self.loss(logits, batch[1])
        self.val_precision(logits, batch[1])
        self.val_recall(logits, batch[1])
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        self.log("val_precision", self.val_precision, on_step=False, on_epoch=True)
        self.log("val_recall", self.val_recall, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        logits = self.model(batch[0])
        loss = self.loss(logits, batch[1])
        self.test_precision(logits, batch[1])
        self.test_recall(logits, batch[1])
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_precision", self.test_precision, on_step=False, on_epoch=True)
        self.log("test_recall", self.test_recall, on_step=False, on_epoch=True)

    def predict_step(self, batch, batch_idx=None):
        logits = self.model(batch)
        probs = torch.softmax(logits, dim=1)
        confidence, predicted_idx = torch.max(probs, 1)
        return predicted_idx, confidence

    def configure_optimizers(self):
        optim = Adam(self.parameters(), lr=1e-3)
        return optim
        
        
        
if __name__ == "__main__":
    from fmnist_datamodule import FashionMNISTDataModule
    from torchvision.transforms.v2 import PILToTensor, Compose, ToDtype
    from lightning.pytorch.callbacks import ModelCheckpoint
    lm = FMNISTLightningModule()
    transform = Compose([
        PILToTensor(),
        ToDtype(torch.float32, scale=True)
    ])

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints/",
        filename="best-fmnist-model",
        save_top_k=1,
        mode="min"
    )

    trainer = Trainer(accelerator="gpu",
                      max_epochs=20,
                      log_every_n_steps=5,
                      callbacks=[checkpoint_callback])
    dm = FashionMNISTDataModule(Path(__file__).parent / "dataset",
                                 train_transform=transform,
                                 val_test_transform=transform,
                                 train_batch_size=1024,
                                 train_num_workers=4,
                                 val_batch_size=1024,
                                 val_num_workers=4,
                                 test_batch_size=2048,
                                 test_num_workers=4)
    trainer.fit(lm, datamodule=dm)
    print(trainer.test(lm, datamodule=dm, ckpt_path="best"))
    
    