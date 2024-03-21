import math
from torch.utils.data import Dataset as BaseDataset
from ..utilities import *


def load_data_from_path(path, idx, sec):
    with open(path, "r") as f:
        lines = f.readlines()
    data = []
    data_index = []
    for i, line in enumerate(lines):
        line = line.rstrip()
        f_path = line.split("\t\t")[0]
        onset = float(line.split("\t\t")[1])
        offset = float(line.split("\t\t")[2])
        data += [{"path": f_path,
                  "data": {
                      "piano_roll":
                          np.load(os.path.join(f_path, "midi.npy"))
                  }}]
        x_len = data[i]["data"]["piano_roll"].shape[0] / 50
        if offset == -1 or x_len < offset:
            offset = x_len
        onset = math.ceil(onset)
        offset = int(offset)
        data_index += [[idx, i, j] for j in range(onset, offset - sec, 10)]
    return data, data_index


class Dataset(BaseDataset):
    def __init__(self, path_lst, cfg, rid, sampling_prob=None, sampling_strategy=None, inference=False):
        super(Dataset, self).__init__()
        self.rid = rid
        self.rng = np.random.RandomState(42 + rid * 100)
        self.cfg = cfg
        self.data = []
        self.data_index = []

        for i, path in enumerate(path_lst):
            data, data_index = load_data_from_path(path, i, cfg.sample_sec)
            self.data.append(data)
            self.data_index += data_index

        self.f_len = len(self.data_index)
        print("num of files", self.f_len)

        self.epoch = 0
        self.f_offset = 0
        self.inference = inference

        self.descs = [
            "catchy song",
            "melodic music piece",
            "a song",
            "music tracks",
        ]
        self.sampling_strategy = sampling_strategy
        if sampling_prob is None:
            sampling_prob = [0., 0.8]
        self.sampling_prob = sampling_prob
        print("samling strategy", self.sampling_strategy, sampling_prob)

    def get_prompt(self):

        prompts = self.descs
        return prompts[self.rng.randint(len(prompts))]

    def load_data(self, set_id, song_id):
        data = self.data[set_id]
        if "chords" not in data[song_id]["data"]:
            piano_roll = data[song_id]["data"]["piano_roll"]
            chord_path = os.path.join(data[song_id]["path"], "chord.npy")
            drums_path = os.path.join(data[song_id]["path"], "drums_rvq.npy")

            drums = np.load(drums_path)

            chords = np.load(chord_path)
            mix = np.load(os.path.join(data[song_id]["path"], "mix_rvq.npy"))

            if chords.shape[0] < piano_roll.shape[0]:
                chords = pad(chords, piano_roll.shape[0], 0, 2)

            result = {
                "mix": mix,
                "chords": chords,
                "piano_roll": piano_roll,
                "drums": drums,
            }
            data[song_id]["data"] = result
        return data[song_id]["data"]

    def track_based_sampling(self, seg_len):
        n = 2
        cond_mask = np.ones([n, seg_len])
        r = self.rng.randint(0, 4)
        if r == 0:
            cond_mask = cond_mask *0
        elif r == 1:
            cond_mask[0] = 0
        elif r == 2:
            cond_mask[1] = 0
        else:
            assert r == 3
        return cond_mask


    def prob_based_sampling(self, seg_len, sampling_prob):
        n = 2
        cond_mask = np.ones([n, seg_len])
        r = self.rng.rand()
        if r < sampling_prob[0]:
            cond_mask = cond_mask * 0.
        else:
            r = self.rng.rand(n)
            for i in range(n):
                if r[i] < sampling_prob[1]:
                    cond_mask[i] = 0
        return cond_mask

    def sample_mask(self, seg_len):
        if self.sampling_strategy == "track-based":
            return self.track_based_sampling(seg_len)
        if self.sampling_strategy == "prob-based":
            return self.prob_based_sampling(seg_len, self.sampling_prob)

    def __len__(self):
        return self.f_len

    def __getitem__(self, idx):
        set_id, sid, sec_id = self.data_index[idx]
        data = self.load_data(set_id, sid)
        mix = data["mix"]
        chords = data["chords"]
        piano_roll = data["piano_roll"]
        drums = data["drums"]

        cfg = self.cfg
        st = sec_id
        ed = st + cfg.sample_sec
        frame_st = int(st * cfg.frame_res)
        frame_ed = int(ed * cfg.frame_res)

        mix = mix[:, frame_st: frame_ed]
        chords = chords[frame_st: frame_ed + 1]
        piano_roll = piano_roll[frame_st: frame_ed + 1]
        drums = drums[:, frame_st: frame_ed + 1]
        pad_chord = np.array(chords.sum(-1) == 0, dtype=np.float16)[:, None]
        chords = np.concatenate([chords, pad_chord], -1)

        seg_len = frame_ed - frame_st + 1
        cond_mask = self.sample_mask(seg_len)
        desc = self.get_prompt()
        return {
            "mix": mix,
            "chords": chords,
            "piano_roll": piano_roll,
            "drums": drums,
            "cond_mask": cond_mask,
            "desc": desc
        }

    def reset_random_seed(self, r, e):
        self.rng = np.random.RandomState(r + self.rid * 100)
        self.epoch = e
        self.rng.shuffle(self.data_index)


def collate_fn(batch):
    mix = torch.stack([torch.from_numpy(d["mix"]) for d in batch], 0)
    chords = torch.stack([torch.from_numpy(d["chords"]) for d in batch], 0)
    piano_roll = torch.stack([torch.from_numpy(d["piano_roll"]) for d in batch], 0)
    drums = torch.stack([torch.from_numpy(d["drums"]) for d in batch], 0)
    cond_mask = torch.stack([torch.from_numpy(d["cond_mask"]) for d in batch], 0)
    desc = [d["desc"] for d in batch]
    return {
        "mix": mix,
        "chords": chords,
        "piano_roll": piano_roll,
        "drums": drums,
        "cond_mask": cond_mask,
        "desc": desc,
    }
