CUDA_VISIBLE_DEVICES=0 python inference.py \
      --num_layers=48 --latent_dim=12 \
      --output_folder="demo/output/let_it_be" \
      --chord_path="demo/input/let_it_be.flac.chord.lab" \
      --midi_path="demo/input/let_it_be.mid" \
      --audio_path="demo/input/let_it_be.flac" \
      --model_path="weights/diff_9_end.pth" \
      --prompt_path="demo/input/let_it_be.prompt.txt" \
      --offset=81
