import numpy
import librosa
from pydub import AudioSegment
from typing import Optional


def loader(
    path: str,
    sr: Optional[float] = 22050.,
    mono: bool = True
):
    '''
    Load a media file like using librosa.load()
    '''
    AudioFile = AudioSegment.from_file(path)

    AudioFile.set_channels(1) if mono and AudioFile.channels > 1 else None

    SamplesArray = [Channel.get_array_of_samples() for Channel in AudioFile.split_to_mono()]
    SamplesArray_ND = numpy.array(SamplesArray) if AudioFile.channels > 1 else numpy.array(*SamplesArray)

    AudioData = SamplesArray_ND.astype(numpy.float32) / numpy.iinfo(SamplesArray[0].typecode).max # Normalization (Librosa defaults to float32, soundfile defaults to float64)
    SampleRate = float(AudioFile.frame_rate)

    AudioData = librosa.core.resample(AudioData, orig_sr = SampleRate, target_sr = float(sr), res_type = 'soxr_vhq') if sr is not None else AudioData
    SampleRate = float(sr) if sr is not None else SampleRate

    return AudioData, SampleRate