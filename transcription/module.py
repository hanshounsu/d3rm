from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import time

import torch
import torch.nn as nn
from torch.nn import functional as F

from termcolor import colored

import numpy as np

from transcription.diffusion.trainer import DiscreteDiffusion
from transcription.constants import HOP
from transcription.evaluate import evaluate

from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable, ReduceLROnPlateau

torch.autograd.set_detect_anomaly(True)

def remove_progress(captured_out):
    lines = (line for line in captured_out.splitlines() if ('it/s]' not in line) and ('s/it]' not in line))
    return '\n'.join(lines)
   

class D3RM(DiscreteDiffusion):
    def __init__(self,
                encoder: nn.Module,
                decoder: nn.Module,
                encoder_parameters: str,
                pretrained_encoder_path: str,
                freeze_encoder: bool,
                test_save_path: str,
                optimizer: OptimizerCallable = torch.optim.AdamW,
                scheduler: LRSchedulerCallable = ReduceLROnPlateau,
                 *args, **kwargs):
        super().__init__(encoder=encoder, decoder=decoder, *args, **kwargs)
        self.save_hyperparameters() # for wandb logging
        self.num_classes = 6
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.test_save_path = test_save_path
        if encoder_parameters == "pretrained":
            ckpt = torch.load(pretrained_encoder_path)
            self.encoder.load_state_dict(ckpt['model_state_dict'], strict=False)
            print(colored('Load pretrained encoder', 'green', attrs=['bold']))
        else: print(colored('Loading random initialized encoder', 'green', attrs=['bold']))
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print(colored('Freeze encoder', 'blue', attrs=['bold']))
        else: print(colored('Train encoder', 'blue', attrs=['bold']))
        self.step = 0
        self.validation_step_outputs = defaultdict(list)

    def training_step(self, batch, batch_idx):
        audio = batch['audio'].to(self.device) # B x L
        label = batch['label'][:,1:,:].to(self.device) # B x T x 88
        # _ = batch['velocity'].to(device)

        # forward
        label = label.reshape(label.shape[0], -1)
        disc_diffusion_loss = self(label, audio, return_loss=True)
        self.log('train/diffusion_loss', disc_diffusion_loss['loss'].mean(), prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'])

        return disc_diffusion_loss['loss']
    
    def validation_step(self, batch, batch_idx):
        audio = batch['audio'].to(self.device)
        label = batch['label'][:,1:,:].to(self.device)
        shape = label.shape
        # Forward step (for loss calculation)
        label = label.reshape(label.shape[0], -1)
        disc_diffusion_loss = self(label, audio, return_loss=True)
        frame_out, _ = self.sample_func(audio) # frame out: B x T x 88 x C
        accuracy = (frame_out == label).float()
        validation_metric = defaultdict(list)
        for n in range(audio.shape[0]):
            sample = frame_out.reshape(*shape)[n] 
            metrics = evaluate(sample, label.reshape(*shape)[n], band_eval=False)
            for k, v in metrics.items(): validation_metric[k].append(v)
        # for k, v in validation_metric.items():
        #     validation_metric[k] = torch.tensor(np.mean(np.concatenate(v)), device=self.device)
        validation_metric['val/accuracy_loss'] = accuracy.mean(dim=-1)
        validation_metric['val/diffusion_loss'] = disc_diffusion_loss['loss'].unsqueeze(0)
        # self.log_dict(validation_metric, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        for k, v in validation_metric.items():
            self.validation_step_outputs[k].extend(v)
    
    def on_validation_epoch_end(self):
        validation_metric_mean = defaultdict(list)
        for k, v in self.validation_step_outputs.items():
            if 'loss' in k:
                validation_metric_mean[k] = torch.mean(torch.stack(v))
            else:
                validation_metric_mean[k] = torch.tensor(np.mean(np.concatenate(v)), device=self.device)
        self.log_dict(validation_metric_mean, prog_bar=True, logger=True, sync_dist=True)
        self.validation_step_outputs.clear()
    
    def test_step(self, batch, batch_idx):
        audio = batch['audio'].to(self.device)
        B, audio_len = audio.shape[0], audio.shape[1]
        test_metric = defaultdict(list)

        frame_outs = [] 
        n_step = (audio_len - 1) // HOP + 1
        shape = (audio.shape[0], n_step, 88)
        seg_len = 313
        overlap = 50

        n_seg = (n_step - overlap) // (seg_len - overlap) + 1

        frame_outs = torch.zeros(shape, dtype=torch.int)

        for seg in tqdm(range(n_seg)):
        # for seg in tqdm(range(2)):
            start = seg * (seg_len - overlap)
            end = start + seg_len
            if end > n_step:
                pad_len = n_step*HOP - audio_len
                start = n_step - seg_len
                end = n_step
                audio_pad = F.pad(audio[:, int(start*HOP):], (0, pad_len))
                audio_pad = audio_pad.to(self.device)
            else:
                audio_pad = audio[:, int(start*HOP):int(end*HOP)].to(self.device)

            frame_out, _ = self.sample_func(audio_pad)
            sample = frame_out.reshape(frame_out.shape[0], -1, 88).detach()
            if seg == 0:
                frame_outs[:, :end-overlap//2] = sample[:, :-overlap//2]
            elif seg == n_seg - 1:
                frame_outs[:, start+overlap//2:] = sample[:, overlap//2:n_step-start]
            else:
                frame_outs[:, start+overlap//2:end-overlap//2] = sample[:, overlap//2:-overlap//2]
        
        # frame out: B x T x 88 x C
        # vel_outs = []
        # obtain metrics on full-length audio for batches
        for n in range(batch['audio'].shape[0]):
            step_len = batch['step_len'][n]
            frame = frame_outs[n][:step_len].detach().cpu()
            metrics = evaluate(frame, batch['label'][n][1:].detach().cpu()[:-1], band_eval=False)
            for k, v in metrics.items():
                test_metric[k].append(v)
            print(f'\n note f1: {metrics["metric/note/f1"][0]:.4f}, note_with_offsets_f1: {metrics["metric_note_with_offsets_f1"][0]:.4f}', batch['path'])
            print(f'\n note precision: {metrics["metric/note/precision"][0]:.4f}, note_with_offsets_precision : {metrics["metric/note-with-offsets/precision"][0]:.4f}', batch['path'])
            print(f'\n note recall: {metrics["metric/note/recall"][0]:.4f}, note_with_offsets_recall : {metrics["metric/note-with-offsets/recall"][0]:.4f}', batch['path'])
        
        # self.log_dict(test_metric, prog_bar=True, logger=True, on_step=False, on_epoch=True,sync_dist=True)

        Path(self.test_save_path).mkdir(parents=True, exist_ok=True)
        for n in range(len(frame_outs)):
            pred = frame_outs[n].detach().cpu().numpy()
            np.savez(Path(self.test_save_path) / (Path(batch['path'][n]).stem + '.npz'), pred=pred)
            print(f'Saved test result to {Path(self.test_save_path) / (Path(batch["path"][n]).stem + ".npz")}')

    def sample_func(self, audio, visualize_denoising=False):
        tic = time.time()
        with torch.no_grad():
            samples, labels = self.sample(audio,
                                        filter_ratio=0,
                                        visualize_denoising=visualize_denoising
                                        ) 
        return samples['label_token'], labels

    def configure_optimizers(self):
        optimizer = self.optimizer(list(self.encoder.parameters()) + list(self.decoder.parameters()))
        scheduler = self.scheduler(optimizer)
        return {"optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": scheduler.monitor,
                    "interval": "step",
                    "frequency": 1,
                    }}