import argparse
import librosa

from coco_mulla.models import CoCoMulla
from coco_mulla.utilities import *
from coco_mulla.utilities.encodec_utils import extract_rvq, save_rvq
from coco_mulla.utilities.symbolic_utils import process_midi, process_chord

from coco_mulla.utilities.sep_utils import separate
from config import TrainCfg
import torch.nn.functional as F

device = get_device()


def generate(model_path, batch):
    model = CoCoMulla(TrainCfg.sample_sec,
                      num_layers=args.num_layers,
                      latent_dim=args.latent_dim).to(device)
    model.load_weights(model_path)
    model.eval()
    with torch.no_grad():
        gen_tokens = model(**batch)

    return gen_tokens


def generate_mask(xlen):
    names = ["chord-only", "chord-drums", "chord-midi", "chord-drums-midi"]
    mask = torch.zeros([4, 2, xlen]).to(device)
    mask[1, 1] = 1
    mask[2, 0] = 1
    mask[3] += 1
    return mask, names


def load_data(audio_path, chord_path, midi_path, offset):
    sr = TrainCfg.sample_rate
    res = TrainCfg.frame_res
    sample_sec = TrainCfg.sample_sec

    wav, _ = librosa.load(audio_path, sr=sr, mono=True)
    wav = np2torch(wav).to(device)[None, None, ...]
    wavs = separate(wav, sr)
    drums_rvq = extract_rvq(wavs["drums"], sr=sr)
    chord, _ = process_chord(chord_path)
    flatten_midi_path = midi_path + ".piano.mid"
    midi, _ = process_midi(midi_path)



    chord = crop(chord[None, ...], "chord", sample_sec, res)
    pad_chord = chord.sum(-1, keepdims=True) == 0
    chord = np.concatenate([chord, pad_chord], -1)

    midi = crop(midi[None, ...], "midi", sample_sec, res,offset=offset)
    drums_rvq = crop(drums_rvq[None, ...], "drums_rvq", sample_sec, res, offset=offset)

    chord = torch.from_numpy(chord).to(device).float()
    midi = torch.from_numpy(midi).to(device).float()
    drums_rvq = drums_rvq.to(device).long()

    return drums_rvq, midi, chord


def crop(x, mode, sample_sec, res, offset=0):
    xlen = x.shape[1] if mode == "chord" or mode == "midi" else x.shape[-1]
    sample_len = int(sample_sec * res) + 1
    if xlen < sample_len:
        if mode == "chord" or mode == "midi":
            x = np.pad(x, ((0, 0), (0, sample_len - xlen), (0, 0)))
        else:
            x = F.pad(x, (0, sample_len - xlen), "constant", 0)
        return x

    st = offset * res
    ed = int((offset + sample_sec) * res) + 1
    if mode == "chord" or mode == "midi":
        assert x.shape[1] > st
        return x[:, st: ed]
    assert x.shape[2] > ed
    return x[:, :, st: ed]


def save_pred(output_folder, tags, pred):
    mkdir(output_folder)
    output_list = [os.path.join(output_folder, tag) for tag in tags]
    save_rvq(output_list=output_list, tokens=pred)


def wrap_batch(drums_rvq, midi, chord, cond_mask, prompt):
    num_samples = len(cond_mask)
    midi = midi.repeat(num_samples, 1, 1)
    chord = chord.repeat(num_samples, 1, 1)
    drums_rvq = drums_rvq.repeat(num_samples, 1, 1)
    prompt = [prompt] * num_samples
    batch = {
        "seq": None,
        "desc": prompt,
        "chords": chord,
        "num_samples": num_samples,
        "cond_mask": cond_mask,
        "drums": drums_rvq,
        "piano_roll": midi,
        "mode": "inference",
    }
    return batch


def inference(args):
    drums_rvq, midi, chord = load_data(audio_path=args.audio_path,
                                       chord_path=args.chord_path,
                                       midi_path=args.midi_path,
                                       offset=args.offset)
    cond_mask, names = generate_mask(drums_rvq.shape[-1])
    batch = wrap_batch(drums_rvq, midi, chord, cond_mask, read_lst(args.prompt_path)[0])
    pred = generate(model_path=args.model_path,
                    batch=batch)
    save_pred(output_folder=args.output_folder,
              tags=names,
              pred=pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_folder', type=str)
    parser.add_argument('-n', '--num_layers', type=int)
    parser.add_argument('-l', '--latent_dim', type=int)
    parser.add_argument('-a', '--audio_path', type=str, default=None)
    parser.add_argument('-c', '--chord_path', type=str, default=None)
    parser.add_argument('-m', '--midi_path', type=str, default=None)
    parser.add_argument('-d', '--drums_path', type=str, default=None)
    parser.add_argument('-e', '--model_path', type=str)
    parser.add_argument('-p', '--prompt_path', type=str)
    parser.add_argument('-f', '--offset', type=int)

    args = parser.parse_args()
    inference(args)
