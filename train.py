from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from rnn_text_classification import LitRNNForSeqClassifier, train_dataloader, valid_dataloader, test_dataloader

if __name__ == "__main__":
    wandb_logger = WandbLogger(project="proj_rnn_cls")

    # model
    lit_rnn_cls = LitRNNForSeqClassifier(
        num_classes=5, cell="gru", embed_dim=1024, hidden_dim=512,
        num_layers=8, dropout=0.1, bidirectional=True
    )

    # train model
    trainer = pl.Trainer(
        max_epochs=20, logger=wandb_logger, devices=2, accelerator="gpu", strategy="ddp",
        callbacks=[EarlyStopping(monitor="valid/acc_epoch", min_delta=0.00, patience=5, verbose=False, mode="max")]
    )
    trainer.fit(model=lit_rnn_cls, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
    trainer.test(model=lit_rnn_cls, dataloaders=test_dataloader)
