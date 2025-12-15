import numpy as np


# This function is obtained from librosa.
def get_rms(
    y,
    *,
    frame_length=2048,
    hop_length=512,
    pad_mode="constant",
):
    padding = (int(frame_length // 2), int(frame_length // 2))
    y = np.pad(y, padding, mode=pad_mode)

    axis = -1
    # put our new within-frame axis at the end for now
    out_strides = y.strides + tuple([y.strides[axis]])
    # Reduce the shape on the framing axis
    x_shape_trimmed = list(y.shape)
    x_shape_trimmed[axis] -= frame_length - 1
    out_shape = tuple(x_shape_trimmed) + tuple([frame_length])
    xw = np.lib.stride_tricks.as_strided(
        y, shape=out_shape, strides=out_strides
    )
    if axis < 0:
        target_axis = axis - 1
    else:
        target_axis = axis + 1
    xw = np.moveaxis(xw, -1, target_axis)
    # Downsample along the target axis
    slices = [slice(None)] * xw.ndim
    slices[axis] = slice(0, None, hop_length)
    x = xw[tuple(slices)]

    # Calculate power
    power = np.mean(np.abs(x) ** 2, axis=-2, keepdims=True)

    return np.sqrt(power)


class Slicer:
    '''
    sr
        Sampling rate of the input audio.
    db_threshold
        The RMS Threshold presented in dB. Areas where all RMS values are below this Threshold will be regarded as silence. Increase this value if your audio is noisy. Defaults to -40.
    audioLength_min
        The minimum length required for each sliced audio clip, presented in milliseconds. Defaults to 5000.
    silentInterval_min
        The minimum length for a silence part to be sliced, presented in milliseconds. Set this value smaller if your audio contains only short breaks. The smaller this value is, the more sliced audio clips this script is likely to generate. Note that this value must be smaller than audioLength_min and larger than hopSize. Defaults to 300.
    hopSize
        Length of each RMS frame, presented in milliseconds. Increasing this value will increase the precision of slicing, but will slow down the process. Defaults to 10.
    max_silence_kept
        The maximum silence length kept around the sliced audio, presented in milliseconds. Adjust this value according to your needs. Note that setting this value does not mean that silence parts in the sliced audio have exactly the given length. The algorithm will search for the best position to slice, as described above. Defaults to 1000.
    '''
    def __init__(self,
        samplingRate: int,
        rmsThreshold: float = -40.,
        audioLength_min: int = 5000,
        silentInterval_min: int = 300,
        hopSize: int = 10,
        silenceKept_max: int = 1000
    ):
        if not audioLength_min >= silentInterval_min >= hopSize:
            raise ValueError('The following condition must be satisfied: audioLength_min >= silentInterval_min >= hopSize')
        if not silenceKept_max >= hopSize:
            raise ValueError('The following condition must be satisfied: silenceKept_max >= hopSize')
        silentInterval_min = samplingRate * silentInterval_min / 1000
        self.rmsThreshold = 10 ** (rmsThreshold / 20.)
        self.hopSize = round(samplingRate * hopSize / 1000)
        self.win_size = min(round(silentInterval_min), 4 * self.hopSize)
        self.audioLength_min = round(samplingRate * audioLength_min / 1000 / self.hopSize)
        self.silentInterval_min = round(silentInterval_min / self.hopSize)
        self.silenceKept_max = round(samplingRate * silenceKept_max / 1000 / self.hopSize)

    def _apply_slice(self, waveform, begin, end):
        if len(waveform.shape) > 1:
            return waveform[:, begin * self.hopSize: min(waveform.shape[1], end * self.hopSize)]
        else:
            return waveform[begin * self.hopSize: min(waveform.shape[0], end * self.hopSize)]

    # @timeit
    def slice(self, waveform):
        if len(waveform.shape) > 1:
            samples = waveform.mean(axis=0)
        else:
            samples = waveform
        if samples.shape[0] <= self.audioLength_min:
            return [waveform]
        rms_list = get_rms(y=samples, frame_length=self.win_size, hop_length=self.hopSize).squeeze(0)
        sil_tags = []
        silence_start = None
        clip_start = 0
        for i, rms in enumerate(rms_list):
            # Keep looping while frame is silent.
            if rms < self.rmsThreshold:
                # Record start of silent frames.
                if silence_start is None:
                    silence_start = i
                continue
            # Keep looping while frame is not silent and silence start has not been recorded.
            if silence_start is None:
                continue
            # Clear recorded silence start if interval is not enough or clip is too short
            is_leading_silence = silence_start == 0 and i > self.silenceKept_max
            need_slice_middle = i - silence_start >= self.silentInterval_min and i - clip_start >= self.audioLength_min
            if not is_leading_silence and not need_slice_middle:
                silence_start = None
                continue
            # Need slicing. Record the range of silent frames to be removed.
            if i - silence_start <= self.silenceKept_max:
                pos = rms_list[silence_start: i + 1].argmin() + silence_start
                if silence_start == 0:
                    sil_tags.append((0, pos))
                else:
                    sil_tags.append((pos, pos))
                clip_start = pos
            elif i - silence_start <= self.silenceKept_max * 2:
                pos = rms_list[i - self.silenceKept_max: silence_start + self.silenceKept_max + 1].argmin()
                pos += i - self.silenceKept_max
                pos_l = rms_list[silence_start: silence_start + self.silenceKept_max + 1].argmin() + silence_start
                pos_r = rms_list[i - self.silenceKept_max: i + 1].argmin() + i - self.silenceKept_max
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                    clip_start = pos_r
                else:
                    sil_tags.append((min(pos_l, pos), max(pos_r, pos)))
                    clip_start = max(pos_r, pos)
            else:
                pos_l = rms_list[silence_start: silence_start + self.silenceKept_max + 1].argmin() + silence_start
                pos_r = rms_list[i - self.silenceKept_max: i + 1].argmin() + i - self.silenceKept_max
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                else:
                    sil_tags.append((pos_l, pos_r))
                clip_start = pos_r
            silence_start = None
        # Deal with trailing silence.
        total_frames = rms_list.shape[0]
        if silence_start is not None and total_frames - silence_start >= self.silentInterval_min:
            silence_end = min(total_frames, silence_start + self.silenceKept_max)
            pos = rms_list[silence_start: silence_end + 1].argmin() + silence_start
            sil_tags.append((pos, total_frames + 1))
        # Apply and return slices.
        if len(sil_tags) == 0:
            IsSlicingNeeded = False
            return [waveform], IsSlicingNeeded
        else:
            IsSlicingNeeded = True
            chunks = []
            if sil_tags[0][0] > 0:
                chunks.append(self._apply_slice(waveform, 0, sil_tags[0][0]))
            for i in range(len(sil_tags) - 1):
                chunks.append(self._apply_slice(waveform, sil_tags[i][1], sil_tags[i + 1][0]))
            if sil_tags[-1][1] < total_frames:
                chunks.append(self._apply_slice(waveform, sil_tags[-1][1], total_frames))
            return chunks, IsSlicingNeeded
