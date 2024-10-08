# D3RM : A Discrete Denoising Diffusion Refinement Model for Piano Transcription

Regarding the reproducement of the paper, please let me know your concerns and feel free to comment them in the `Issues` part.

<img src="https://github.com/hanshounsu/d3rm/blob/main/images/Absorbing%20state.png?raw=true" height="400"/> <img src="https://github.com/hanshounsu/d3rm/blob/main/images/Model%20architecture.png" height="400"/>


## Installation

```shell
git clone https://github.com/hanshounsu/d3rm.git
pip -r install requirements.txt
```

## Model Download
* Pretrained NAR-HC baseline model [[link](https://drive.google.com/file/d/1puA0CkXGioXs9OrS1w-AvwN71yi2cxae/view?usp=sharing)]
* Pretrained D3RM model [[link]()]

## Download MAESTRO

## Training the model
```shell
python -m transcription.train_diffusion --config=configs/VQ_Diffusion_S.json --no-ddp --finetune -b 8 
```

## Inference
```shell
CUDA_VISIBLE_DEVICES="1" python -m transcription.train_diffusion --config=configs/VQ_Diffusion_S.json --no-ddp --finetune --eval --resume_dir runs/DiscDiff_240901-102544_91259ght
```

## Additional Samples

## Acknowledge

<!-- 1. We had a consistent design of [FunASR](https://github.com/alibaba/FunASR), including dataloader, model definition and so on. -->
<!-- 2. We borrowed a lot of code from [Kaldi](http://kaldi-asr.org/) for data preparation. -->
<!-- 4. We borrowed the design of model architecture from [Enocdec](https://github.com/facebookresearch/encodec) and [Enocdec_Trainner](https://github.com/Mikxox/EnCodec_Trainer). -->

## License
This project is licensed under [The MIT License](https://opensource.org/licenses/MIT). 
FunCodec also contains various third-party components and some code modified from other repos 
under other open source licenses.

## Citations

``` bibtex
@misc{hskim2023d3rm,
      title={D3RM : A Discrete Denoising Diffusion Refinement Model for Piano Transcription},
      author={Hounsu Kim, Taegyun Kwon, Juhan Nam},
      year={2024},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={cs.Sound}
}
```


https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.csv
https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.json
https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip
https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip