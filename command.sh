CUDA_VISIBLE_DEVICES="1" python -m transcription.train_diffusion --config=configs/VQ_Diffusion_S.json --no-ddp --finetune -b 8 
CUDA_VISIBLE_DEVICES="1" python -m transcription.train_diffusion --config=configs/VQ_Diffusion_S.json --no-ddp --finetune --eval --resume_dir runs/DiscDiff_240308-223748_q0mqyaco
CUDA_VISIBLE_DEVICES="1" python -m transcription.train_diffusion --config=configs/VQ_Diffusion_S.json --no-ddp --finetune --eval --resume_dir runs/DiscDiff_240326-202512_p2rai79d
CUDA_VISIBLE_DEVICES="1" python -m transcription.train_diffusion --config=configs/VQ_Diffusion_S.json --no-ddp --finetune --eval --resume_dir runs/DiscDiff_240814-020753_w4514bhv --resume_id w4514bhv


python3 main_cli.py test -c ./logs/2026-01-30T22-29-05/config.yaml --ckpt_path /home/hounsu/subprojects/d3rm/checkpoints/2026-01-30T22-29-05/step=0170000-metric_note_with_offsets_f1=0.9464.ckpt
python -m transcription.precise_evaluation /home/hounsu/subprojects/d3rm/results/2026-01-30T22-29-05 MAESTRO_V3 test