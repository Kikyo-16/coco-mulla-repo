from .reverse_pianoroll import piano_roll_to_pretty_midi
import pretty_midi

ON = 375
OFF = ON + 1
TIE = OFF + 1

PITCH = TIE + 1
FLATTEN_PROG = PITCH + 128
END = FLATTEN_PROG + 1
PROG = END + 1
DRUM = PROG + 128

MIDI_DICT_PATH = "data/text/midi.lst"
DRUM_DICT_PATH = "data/text/drums.lst"


def load_midi_dict(path):
    with open(path, "r") as f:
        midi_dict = f.readlines()
    out_dict = {}
    parent_dict = {}
    for line in midi_dict:
        line = line.rstrip().split("\t")
        pg = line[2].split(" ")
        prog = []
        for p in pg:
            if str.startswith(p, "(") or (ord(p[0]) >= ord(' ') and ord(p[0]) <= ord('9')):
                break
            prog.append(p)

        prog = " ".join(prog)
        out_dict[line[0]] = prog
        parent_dict[prog] = line[1]
    # for p in out_dict:
    #    print(p, out_dict[p])
    return out_dict, parent_dict


def load_drums_dict(path):
    with open(path) as f:
        lines = f.readlines()
    out_dict = {}
    for line in lines:
        line = line.rstrip().split("\t")
        out_dict[line[1]] = line[2]
    # for k in out_dict:
    #    print(k, out_dict[k])
    return out_dict


MIDI_DICT, MIDI_PARENT_DICT = load_midi_dict(MIDI_DICT_PATH)
DRUMS_DICT = load_drums_dict(DRUM_DICT_PATH)


def encode_simul_events(events):
    index = {}
    outs = []
    for event in events:
        time = str(event["time"])
        if time not in index:
            index[time] = len(outs)
            outs.append([])
        ind = index[time]
        outs[ind].append(event)
    return outs, index


def read_events(midi, res, is_flatten):
    events = []
    max_time = -1
    if is_flatten:
        midi = flatten_midi(midi, res)
    tempo = midi.estimate_tempo()
    for instr in midi.instruments:

        prog = DRUM if instr.is_drum else instr.program + PROG
        instr_name = "drums" if instr.is_drum else MIDI_DICT[str(instr.program)]
        if instr_name == "drums":
            if str(instr.program) in DRUMS_DICT:
                instr_name = DRUMS_DICT[str(instr.program)]
        # print(instr_name, instr.program, instr.name)
        if prog == DRUM and is_flatten:
            continue
        if is_flatten:
            prog = FLATTEN_PROG
        for note in instr.notes:
            start = round(note.start * res)
            end = round(note.end * res)
            event = {
                "time": start,
                "on_off": ON,
                "prog": prog,
                "prog_name": instr_name,
                "pitch": note.pitch + PITCH,
                "end": end,
            }
            events.append(event)
            event = {
                "time": end,
                "on_off": OFF,
                "prog": prog,
                "prog_name": instr_name,
                "pitch": note.pitch + PITCH,
                "start": start,
            }
            events.append(event)
            if end > max_time:
                max_time = end

    events = sorted(events, key=lambda d: (d['time'], d['on_off'], d['prog'], d['pitch']))
    events, index = encode_simul_events(events)
    onsets = []
    st = 0
    for i, k in enumerate(index):
        ind = index[k]
        while events[ind][0]["time"] >= st:
            onsets.append(ind)
            st += 1
    offsets = []
    ed = max_time
    for i, k in enumerate(reversed(index)):
        ind = index[k]
        while events[ind][0]["time"] <= ed:
            offsets = [ind] + offsets
            ed -= 1
    while ed >= 0:
        offsets = [0] + offsets
        ed -= 1

    def sample_fn(st, ed):
        return sample_events(events, onsets, offsets, st, ed)

    return sample_fn, max_time, tempo


def sample_events(events, onsets, offsets, st, ed):
    onset = onsets[st]
    offset = offsets[ed - 1] + 1
    events = events[onset: offset]
    # if st < 10:
    #    print(st, ed, onset, offset, events)

    head = []
    for simul in events:
        for e in simul:
            if e["on_off"] == OFF and e["start"] < st:
                head.append(e)

    outs = []
    debug = []
    prog = -1
    for e in head:
        if not e["prog"] == prog:
            prog = e["prog"]
            outs.append(prog)

        outs.append(e["pitch"])

    outs.append(TIE)
    on_off = -1
    instr_names = []
    for simul in events:

        time_flag = True
        for e in simul:
            time = e["time"]
            offset = int(time) - st
            if offset > 0 and time_flag:
                outs.append(offset)
                time_flag = False

            if not e["prog"] == prog:
                prog = e["prog"]
                outs.append(prog)

            if not e["on_off"] == on_off:
                on_off = e["on_off"]
                outs.append(on_off)
            outs.append(e["pitch"])
            debug.append(e)
            instr_names.append(e["prog_name"])
    outs.append(END)
    return outs, list(set(instr_names))


def load_midi(path):
    midi = pretty_midi.PrettyMIDI(path)
    return midi


def flatten_midi(midi, res):
    piano_roll = midi.get_piano_roll(fs=res)
    print(piano_roll.shape[-1])
    new_midi = piano_roll_to_pretty_midi(piano_roll, fs=res)
    return new_midi


if __name__ == "__main__":
    path = "output/Track00186.mid"
    midi = load_midi(path)
    sample_fn, _ = read_events(midi, 100, is_flatten=True)
    st = 400
    ed = 764
    sample = sample_fn(st, ed)
    print(len(sample), sample)
