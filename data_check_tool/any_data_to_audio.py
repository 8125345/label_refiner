"""
把mel频谱图转换为音频格式，便于校验数据
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import os.path
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, as_completed

import librosa
import glob
import json
import numpy as np
import soundfile as sf

import tensorflow as tf
from pkl_to_midi import pkl_to_mid

SR = samplerate = 16000


def spec2mel(data):
    spec = librosa.db_to_power(data)
    mel = spec.T

    return mel


def mel2audio(fn, sr=SR, hop_length=512, fmin=30.0, htk=True):
    mel = fn
    audio = librosa.feature.inverse.mel_to_audio(
        mel,
        sr=sr,
        hop_length=hop_length,
        fmin=fmin,
        # n_mels=n_mels,  # n_mels的特征librosa后续模块会从shape反向导出，无需填写
        htk=htk)
    return audio


def parse_np_data(path):
    """
    解析numpy格式文件
    :param path:
    :return:
    """
    data = np.load(path)
    chunk_spec = data[:, : 229]
    # bgm_chunk_spec = data[:, 229: 2 * 229]
    # onsets_label = data[:, 2 * 229:]
    onsets_label = data[:, 229:]
    # return bgm_chunk_spec, chunk_spec, onsets_label
    return chunk_spec, onsets_label


def parse_serialized_data(path):
    serialized = tf.io.read_file(path)
    data = tf.io.parse_tensor(serialized, tf.float32)

    feature = data[:, : 293120]
    midi = data[:, 293120:]

    feature = tf.reshape(feature, (640, 229, 2))
    midi = tf.reshape(midi, (640, 1))

    # concat = tf.concat((feature[:, :, 0], feature[:, :, 1], midi), axis=1)  # 640 * (229 + 229 + 88)
    bgm_chunk_spec = feature[:, :, 1]
    chunk_spec = feature[:, :, 1]
    return bgm_chunk_spec, chunk_spec, midi


def recover_seg(path, dst_folder_path, dst_name, mode, save_midi=True):
    """

    :param dst_name:
    :param path:
    :param dst_folder_path:
    :param mode:
    :return:
    """
    if mode == "np":
        # bgm, mix, midi = parse_np_data(path)
        mix, midi = parse_np_data(path)
    else:
        bgm, mix, midi = parse_serialized_data(path)

    mel = spec2mel(mix)
    audio = mel2audio(mel)
    # 保存mix路
    sf.write(os.path.join(dst_folder_path, dst_name + "_mix" + ".wav"), audio, samplerate)
    # todo 根据需要保存另一路

    if save_midi:
        pkl_to_mid.convert_to_midi_single(midi, os.path.join(dst_folder_path, dst_name) + ".midi")
    return audio


def recover_song(song_folder_path, dst_folder_path, mode="np", limit=None):
    """
    把一首歌的序列化文件（.np/.serialized）转化为音频，并保存

    :param song_folder_path: 歌曲所在路径，路径中存在多个序列化文件
    :param dst_folder_path: 生成音频保存路径
    :param mode: np: numpy格式，serialized: tf保存格式
    :return:
    """
    assert os.path.exists(song_folder_path)
    assert mode in ("np", "serialized")
    if mode == "np":
        file_paths = glob.glob(os.path.join(song_folder_path, "*.npy"))
    else:
        file_paths = glob.glob(os.path.join(song_folder_path, "*.serialized"))

    def sort_fun(path):
        f_name = os.path.split(path)[-1]
        return int(os.path.splitext(f_name)[0])

    file_paths = sorted(file_paths, key=sort_fun)

    # 此处用多进程控制
    for idx, file_path in enumerate(file_paths):
        f_name = os.path.split(file_path)[-1]
        # dst_path = os.path.join(dst_folder_path, os.path.splitext(f_name)[0] + ".wav")
        dst_name = os.path.splitext(f_name)[0]
        recover_seg(file_path, dst_folder_path, dst_name, mode=mode)
        if limit is not None:
            if idx == limit:
                break


def worker(map_dict):
    src_dir = map_dict["src_dir"]
    dst_dir = map_dict["dst_dir"]
    global_id = os.path.split(dst_dir)[-1]
    global_id = global_id.split("_")[-1]
    print(f"{global_id}_start")
    recover_song(src_dir, dst_dir, mode="np", limit=10)  # 每首歌最多抽检指定数量片段
    print(f"{global_id}_finish")


def run():
    # 把一个歌曲文件夹中若干片段转化为音频数据
    # song_folder_path = "/deepiano_data/zhaoliang/qingchen_data/npy_negbgm_record/std/bgm_20220906_200604/001"  # 歌曲路径
    # dst_folder_path = "/data1/projects/BGMcloak/tmp_files"  # 歌曲片段保存路径
    # recover_song(song_folder_path, dst_folder_path, mode="np")

    # song_folder_path = "/deepiano_data/zhaoliang/record_data/serialize_bgm_record_delay/std/bgm_record_20220822_20220905_114647/original/20220822export-001"
    # song_folder_path = "/deepiano_data/yuxiaofei/work/data_0718/serialize_changpu_delay_Peilian_SC55_metronome/std/Peilian_xml_SC55_20220811_172927/original/000"
    # song_folder_path = "/deepiano_data/yuxiaofei/work/data_0718/serialize_noise/noise_trans/noise_20220809_170445/1-16k"

    dst_base_root = "/deepiano_data/zhaoliang/data_for_check"
    dst_root = os.path.join(dst_base_root, "check_split_xml_20221010")
    if not os.path.exists(dst_root):
        os.mkdir(dst_root)

    worker_num = 1
    song_dir_list = [
        '/deepiano_data/zhaoliang/SC55_data/Alignment_data/split_320_npy/ipad6SC55录音版/total/xml_arachno_000'

    ]
    # 目标目录预生成，避免多进程创建目录IO冲突
    convert_map_list = list()  # 转换映射列表
    for dir_id, path in enumerate(song_dir_list):
        assert os.path.exists(path)
        dir_name = os.path.split(path)[-1]  # 源目录名称
        dir_name = dir_name + f"_{dir_id}"  # 增加全局id避免路径冲突
        dst_dir = os.path.join(dst_root, dir_name)  # 目标目录名称
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)
        convert_map_list.append({
            "src_dir": path,
            "dst_dir": dst_dir,
        })  # 记录源目录，目标目录(path, dst_dir)

    # 保存映射记录
    with open(os.path.join(dst_root, "dir_map.json"), "w") as f:
        json.dump(convert_map_list, f)

    pool = Pool(worker_num)
    ret = pool.map(worker, convert_map_list)
    pool.close()
    pool.join()


if __name__ == '__main__':
    run()
