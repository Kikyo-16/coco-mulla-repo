import numpy as np

from ..utilities.symbolic_utils import reduce_piano, process_chord, process_midi
from ..utilities.sep_utils import separate
from ..utilities.encodec_utils import extract_rvq, extract_musicgen_emb
from ..utilities import *
import librosa
from config import *

device = get_device()


def process_single(output_folder, path_dict, fname):
    output_folder = os.path.join(output_folder, fname)
    print("begin", output_folder)
    mkdir(output_folder)
    audio_path = path_dict["audio"]
    chord_path = path_dict["chord"]
    midi_path = path_dict["midi"]
    sr = 48000

    drums_output_path = os.path.join(output_folder, "drums_rvq.npy")
    mix_output_path = os.path.join(output_folder, "mix_rvq.npy")
    chord_output_path = os.path.join(output_folder, "chord.npy")
    midi_output_path = os.path.join(output_folder, "midi.npy")
    print(mix_output_path)
    if os.path.exists(mix_output_path):
        print(mix_output_path, "skip")
        return

    flatten_midi_path = midi_path+".piano.mid"
    if not os.path.exists(flatten_midi_path):
        reduce_piano(midi_path, reduced_path=flatten_midi_path)

    wav, _ = librosa.load(audio_path, sr=sr, mono=True)

    wav = np2torch(wav).to(device)[None, None, ...]
    wavs = separate(wav, sr)
    print("separate", output_folder)

    drums_rvq = extract_rvq(wavs["drums"], sr=sr)
    mix_rvq = extract_rvq(wav, sr=sr)

    chord, _ = process_chord(chord_path)
    piano_roll, _ = process_midi(flatten_midi_path)
    max_len = len(drums_rvq[0])

    if len(chord) < max_len:
        t_len = len(chord)
        chord = np.pad(chord, ((0, max_len - len(chord)), (0, 0)), "constant", constant_values=(0, 0))
        chord[t_len:, -1] = 1
    else:
        chord = chord[:max_len]
    piano_roll = np.pad(piano_roll, ((0, max_len - len(piano_roll)), (0, 0)), "constant", constant_values=(0, 0))

    np.save(chord_output_path, chord)
    np.save(midi_output_path, piano_roll)




    np.save(drums_output_path, drums_rvq.cpu().numpy())
    np.save(mix_output_path, mix_rvq.cpu().numpy())


def scan_audio(audio_folder, low, up):
    res = {}
    for song in os.listdir(audio_folder):
        fname = song.split("Track")[-1]
        audio_path = os.path.join(audio_folder, song, "mix.flac")
        midi_path = os.path.join(audio_folder, song, "all_src.mid")

        if int(fname) < low or int(fname) >= up:
            continue
        res[fname] = {
            "audio": audio_path,
            "midi": midi_path,
            "chord": audio_path + ".chord.lab",
        }
    return res


def process_all(audio_folder, output_folder, low, up):
    data = scan_audio(audio_folder, low, up)
    for song_name in data:
        process_single(output_folder, data[song_name], song_name)
