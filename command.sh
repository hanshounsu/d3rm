CUDA_VISIBLE_DEVICES="1" python -m transcription.train_diffusion --config=configs/VQ_Diffusion_S.json --no-ddp --finetune -b 8 
CUDA_VISIBLE_DEVICES="1" python -m transcription.train_diffusion --config=configs/VQ_Diffusion_S.json --no-ddp --finetune --eval --resume_dir runs/DiscDiff_240308-223748_q0mqyaco
CUDA_VISIBLE_DEVICES="1" python -m transcription.train_diffusion --config=configs/VQ_Diffusion_S.json --no-ddp --finetune --eval --resume_dir runs/DiscDiff_240326-202512_p2rai79d
CUDA_VISIBLE_DEVICES="1" python -m transcription.train_diffusion --config=configs/VQ_Diffusion_S.json --no-ddp --finetune --eval --resume_dir runs/DiscDiff_240814-020753_w4514bhv --resume_id w4514bhv