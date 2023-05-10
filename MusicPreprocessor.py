import glob
import numpy as np
import pretty_midi

class MusicPreprocessor:
    def __init__(self, midi_file_path, sequence_length=100, note_range=(21, 109)):
        self.midi_file_path = midi_file_path
        self.sequence_length = sequence_length
        self.note_range = note_range
        
    def get_data(self):
        data = []
        for file in glob.glob(self.midi_file_path + '/*.mid'):
            midi_data = pretty_midi.PrettyMIDI(file)
            note_seq = midi_data.get_piano_roll(fs=4)[self.note_range[0]:self.note_range[1]]
            note_seq[note_seq > 0] = 1
            if note_seq.shape[1] > self.sequence_length:
                for i in range(0, note_seq.shape[1]-self.sequence_length, self.sequence_length):
                    seq = note_seq[:, i:i+self.sequence_length]
                    data.append(seq)
        data = np.array(data)
        data = np.reshape(data, (-1, self.sequence_length, self.note_range[1]-self.note_range[0]))
        return data

    
