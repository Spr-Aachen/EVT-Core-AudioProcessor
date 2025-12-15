import os
import torch
import traceback
import librosa
import soundfile
from pathlib import Path

from .uvr5.mdxnet import MDXNetDereverb
from .uvr5.vr import AudioPre, AudioPreDeEcho


def uvr(
    audioData,
    sampleRate,
    modelPath,
    target,
    agg
):
    try:
        if "onnx_dereverb_by_foxjoy" in modelPath.lower() and Path(modelPath).suffix == ".onnx":
            pre_fun = MDXNetDereverb(
                model_path = modelPath,
                chunks = 15
            )
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            func = AudioPre if "DeEcho" not in Path(modelPath).stem else AudioPreDeEcho
            pre_fun = func(
                agg = int(agg),
                model_path = modelPath,
                device = device,
                is_half = False,
            )
        tmp_path = Path(os.getcwd()).joinpath("tmp.wav").as_posix()
        audioData = librosa.resample(audioData, orig_sr = sampleRate, target_sr = 44100)
        soundfile.write(tmp_path, audioData.T if len(audioData.shape) > 1 else audioData, 44100, subtype = 'PCM_16')
        data, samplerate = pre_fun._path_audio_(
            path = tmp_path,
            target = target,
            is_hp3 = "hp3" in Path(modelPath).stem.lower()
        )
    except:
        traceback.print_exc()
        data, sr = audioData, sampleRate
    else:
        data, sr = data.T if len(data.shape) > 1 else data, samplerate
    finally:
        os.remove(tmp_path)
        if modelPath == "onnx_dereverb_By_FoxJoy":
            del pre_fun.pred.model
            del pre_fun.pred.model_
        else:
            del pre_fun.model
            del pre_fun
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return data, sr


def denoiser(
    audioData,
    sampleRate,
    modelPath,
    target = "vocals", # 指定要保留人声还是背景音
    #agg = 10 # 人声提取激进程度(0-20, step=1)
):
    return uvr(
        audioData,
        sampleRate,
        modelPath,
        target = target,
        agg = 10
    )