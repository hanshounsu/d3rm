# For inference on MAESTRO_V3 test set
python3 main_cli.py test -c ./logs/2026-01-30T22-29-05/config.yaml --ckpt_path /home/hounsu/subprojects/d3rm/checkpoints/2026-01-30T22-29-05/step=0170000-metric_note_with_offsets_f1=0.9464.ckpt
# For obtaining metrics on inference results (on the .npz files)
python -m transcription.precise_evaluation /home/hounsu/subprojects/d3rm/results/2026-01-30T22-29-05 MAESTRO_V3 test