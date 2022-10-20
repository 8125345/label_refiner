import collections
import os
import math
import numpy as np
import pickle

# import constants as music_constants
# import midi_io
# from protobuf import music_pb2

import pkl_to_midi.constants as music_constants
import pkl_to_midi.midi_io as midi_io
from pkl_to_midi.protobuf import music_pb2

onset_threshold_list = [0.001, 0.01, 0.1, 0.3, 0.5, 0.8]

ChunkPrediction = collections.namedtuple(
    'ChunkPrediction',
    ('onset_predictions', 'velocity_values'))


def hparams_frames_per_second():
    """Compute frames per second"""
    return 16000 / 512


def pianoroll_to_note_sequence(chunk_pred,
                               frames_per_second,
                               velocity=70,
                               instrument=0,
                               program=0,
                               qpm=music_constants.DEFAULT_QUARTERS_PER_MINUTE,
                               min_midi_pitch=music_constants.MIN_MIDI_PITCH):
    frame_length_seconds = 1 / frames_per_second

    sequence = music_pb2.NoteSequence()
    sequence.tempos.add().qpm = qpm
    sequence.ticks_per_quarter = music_constants.STANDARD_PPQ

    note_duration = frame_length_seconds * 3  # to remove redundant same midi
    total_frames = 0  # left padding

    last_note = {}  # {'pitch': time}

    def unscale_velocity(velocity):
        unscaled = max(min(velocity, 1.), 0) * 80. + 10.
        if math.isnan(unscaled):
            return 0
        return int(unscaled)

    def process_chunk(chunk_prediction):
        nonlocal total_frames

        onset_predictions = chunk_prediction.onset_predictions
        velocity_values = chunk_prediction.velocity_values

        for i, onset in enumerate(onset_predictions):
            for pitch, active in enumerate(onset):
                if active:
                    time = (total_frames + i) * frame_length_seconds
                    pitch = pitch + min_midi_pitch
                    if time - last_note.get(pitch, -1) > note_duration:
                        note = sequence.notes.add()
                        note.start_time = time
                        note.end_time = time + note_duration
                        note.pitch = pitch
                        note.velocity = unscale_velocity(velocity_values[i, pitch] if velocity_values else velocity)
                        note.instrument = instrument
                        note.program = program

                        last_note[note.pitch] = note.start_time
                        # print('note:', note.pitch)

        total_frames += len(onset_predictions)

    # print('begin process chunk')
    process_chunk(chunk_pred)

    # print('end process chunk')
    sequence.total_time = total_frames * frame_length_seconds
    return sequence


def convert_to_midi_single(mel_data, data_path):
    """
    ground truth label转midi
    :param mel_data:
    :param data_path:
    :return:
    """
    cpy_concat_data = mel_data

    tmp_concat_data = cpy_concat_data > 0.5

    concat_pred = ChunkPrediction(onset_predictions=tmp_concat_data, velocity_values=None)

    concat_sequence = pianoroll_to_note_sequence(concat_pred, frames_per_second=hparams_frames_per_second(),
                                                 min_midi_pitch=21)

    midi_io.sequence_proto_to_midi_file(concat_sequence, data_path)


def convert_to_midi(concat_pkl_dir, online_pkl_dir, data_folder):
    # # 加载数据
    # # data_folder = "/data1/projects/BGMcloak/real_song_data/jm_20220718_1/"
    #
    # # data_folder = "/data1/projects/BGMcloak/real_song_data/jm_20220718_2/"
    #
    # # data_folder = "/data/projects/BGMcloak/real_song_data/ios_bgm_record_2022_0718_3"  # BGM声音大
    # data_folder = "/data/projects/BGMcloak/real_song_data/ios_bgm_record_2022_0718_4"  # BGM声音大
    #
    #
    # # concat_pkl_dir = "./data/tmp/forceconcat_20220708_1_predict.pkl"
    # # online_pkl_dir = './data/tmp/online_504_predict.pkl'
    #
    # concat_pkl_dir = os.path.join(data_folder, "forceconcat_20220708_1_predict.pkl")
    # online_pkl_dir = os.path.join(data_folder, "online_504_predict.pkl")

    with open(concat_pkl_dir, 'rb') as f:
        concat_data = pickle.load(f)

    with open(online_pkl_dir, 'rb') as f:
        online_data = pickle.load(f)

    cpy_concat_data = concat_data
    cpy_online_data = online_data

    cpy_concat_data = cpy_concat_data[:, 3:5, :]
    cpy_online_data = cpy_online_data[:, 3:5, :]

    cpy_concat_data = np.concatenate(cpy_concat_data, axis=0)
    cpy_online_data = np.concatenate(cpy_online_data, axis=0)

    for tmp_th in onset_threshold_list:
        tmp_concat_data = cpy_concat_data > tmp_th
        tmp_online_data = cpy_online_data > tmp_th

        concat_pred = ChunkPrediction(onset_predictions=tmp_concat_data, velocity_values=None)
        online_pred = ChunkPrediction(onset_predictions=tmp_online_data, velocity_values=None)

        concat_sequence = pianoroll_to_note_sequence(concat_pred, frames_per_second=hparams_frames_per_second(),
                                                     min_midi_pitch=21)
        online_sequence = pianoroll_to_note_sequence(online_pred, frames_per_second=hparams_frames_per_second(),
                                                     min_midi_pitch=21)

        dst_concat_file = os.path.join(data_folder, 'concat_%s.midi' % tmp_th)
        dst_online_file = os.path.join(data_folder, 'online_504_%s.midi' % tmp_th)
        midi_io.sequence_proto_to_midi_file(concat_sequence, dst_concat_file)
        midi_io.sequence_proto_to_midi_file(online_sequence, dst_online_file)
