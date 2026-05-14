import os
import traceback

import librosa
import numpy as np
import av
from io import BytesIO


def wav2(i, o, format):
    # 修复：直接传递路径字符串，不需要 "r" 参数
    inp = av.open(i)
    if format == "m4a":
        format = "mp4"
    out = av.open(o, "w", format=format)
    if format == "ogg":
        format = "libvorbis"
    if format == "mp4":
        format = "aac"

    ostream = out.add_stream(format)

    for frame in inp.decode(audio=0):
        for p in ostream.encode(frame):
            out.mux(p)

    for p in ostream.encode(None):
        out.mux(p)

    out.close()
    inp.close()


def audio2(i, o, format, sr):
    # 修复：直接传递路径字符串，不需要 "r" 参数
    inp = av.open(i)
    out = av.open(o, "w", format=format)
    if format == "ogg":
        format = "libvorbis"
    if format == "f32le":
        format = "pcm_f32le"

    ostream = out.add_stream(format, channels=1)
    ostream.sample_rate = sr

    for frame in inp.decode(audio=0):
        for p in ostream.encode(frame):
            out.mux(p)

    out.close()
    inp.close()


def load_audio(file, sr):
    file = (
        file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
    )  # 防止小白拷路径头尾带了空格和"和回车
    if os.path.exists(file) == False:
        raise RuntimeError(
            "You input a wrong audio path that does not exists, please fix it!"
        )
    try:
        # 修复：直接传递文件路径字符串，而不是以文本模式打开的文件对象
        with BytesIO() as out:
            audio2(file, out, "f32le", sr)
            return np.frombuffer(out.getvalue(), np.float32).flatten()

    except AttributeError:
        # 处理特殊情况：file 参数可能是元组 (sample_rate, audio_data)
        audio = file[1] / 32768.0
        if len(audio.shape) == 2:
            audio = np.mean(audio, -1)
        return librosa.resample(audio, orig_sr=file[0], target_sr=16000)

    except:
        raise RuntimeError(traceback.format_exc())


# 备选方案：如果你的 PyAV 版本仍需要指定模式，可以使用这个版本
def audio2_with_mode(i, o, format, sr):
    """
    备选函数：如果需要显式指定模式
    """
    # 如果 i 是文件对象，从中读取路径
    if hasattr(i, 'name'):
        filepath = i.name
    else:
        filepath = i
    
    # 确保路径是字符串
    if isinstance(filepath, bytes):
        filepath = filepath.decode('utf-8', errors='ignore')
    
    # 使用 mode='r' 参数（注意：不是 "rb"）
    inp = av.open(filepath, mode='r')
    out = av.open(o, "w", format=format)
    
    if format == "ogg":
        format = "libvorbis"
    if format == "f32le":
        format = "pcm_f32le"

    ostream = out.add_stream(format, channels=1)
    ostream.sample_rate = sr

    for frame in inp.decode(audio=0):
        for p in ostream.encode(frame):
            out.mux(p)

    out.close()
    inp.close()


# 另一个备选方案：使用 librosa 直接加载（更稳定）
def load_audio_librosa(file, sr):
    """
    使用 librosa 直接加载音频文件，避免 av 库的问题
    """
    file = (
        file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
    )
    if os.path.exists(file) == False:
        raise RuntimeError(
            "You input a wrong audio path that does not exists, please fix it!"
        )
    
    try:
        # 使用 librosa 直接加载
        audio, orig_sr = librosa.load(file, sr=sr, mono=True)
        return audio.astype(np.float32)
    
    except Exception as e:
        raise RuntimeError(f"Error loading audio {file}: {traceback.format_exc()}")