# Coco-mulla

Offical repository of [Content-based Controls For Music Large Language Modeling](https://arxiv.org/abs/2310.17162). You can visit our demo page [here](https://kikyo-16.github.io/coco-mulla/).


## Quick Start

### Requirements
The project requires Python 3.11. See `requirements.txt`.

    pip install -r requirements.txt

### Inference

    python inference.py --num_layers=48 --latent_dim=12 \
                        --output_folder=your_output_folder \
                        --model_path=your_model_weight_path \
                        --audio_path=your_audio_path \
                        --midi_path=your_midi_path \
                        --chord_path=your_chord_path \
                        --prompt_path=your_prompt_path \
                        --onset=your_onset  
See `demo` folder for the input data format. `Onset` should be a number indicating the starting second of the input audio.


### Model Weights
We provide with model weights with `r=0.2` and `r=0.4`. Download them via [model weights](https://drive.google.com/drive/folders/1o5xiD5unoDG5L3CSvcsxDR3Z02d8EEJf?usp=sharing). 

# How to cite
    @misc{lin2023contentbased,
      title={Content-based Controls For Music Large Language Modeling}, 
      author={Liwei Lin and Gus Xia and Junyan Jiang and Yixiao Zhang},
      year={2023},
      eprint={2310.17162},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
