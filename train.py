import subprocess
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset_utils import PromptTrainDataset
from net.DyNet_large import DyNet_large
from utils.schedulers import LinearWarmupCosineAnnealingLR
from options import options as opt
import lightning as pl
from lightning.pytorch.loggers import WandbLogger,TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from torchmetrics.image import PeakSignalNoiseRatio



class DyNetModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = DyNet_large(decoder=True)
        self.loss_fn  = nn.L1Loss()
        self.psnr = PeakSignalNoiseRatio()
    
    def forward(self,x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):

        degrad_patch, clean_patch, clean_name = batch
        restored = self.net(degrad_patch)
        restored = restored.clamp(0.0, 1.0)

        loss = self.loss_fn(restored,clean_patch)
        self.log('train_loss', loss, on_step=True, prog_bar=True, sync_dist=True)
        return loss
    
    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=15,max_epochs=150)

        return [optimizer],[scheduler]



def main():
    print("Options")
    print(opt)
    if opt.wblogger is not None:
        logger  = WandbLogger(project=opt.wblogger,name="DyNet-Training")
    else:
        logger = TensorBoardLogger(save_dir = "logs/")

    trainset = PromptTrainDataset(opt)
    checkpoint_callback = ModelCheckpoint(dirpath = opt.ckpt_dir,every_n_epochs = 1,save_top_k=-1)
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=opt.num_workers)
    
    model = DyNetModel().cuda()
    
    trainer = pl.Trainer(max_epochs=opt.epochs,accelerator="gpu",devices=opt.num_gpus,strategy="ddp_find_unused_parameters_true",logger=logger,callbacks=[checkpoint_callback])
    trainer.fit(model, trainloader)


if __name__ == '__main__':
    main()



