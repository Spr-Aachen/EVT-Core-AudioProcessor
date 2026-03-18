import os
import sys
import glob
import soundfile
from typing import Union, Optional
from concurrent.futures import ThreadPoolExecutor

from pathlib import Path
current_dir = Path(__file__).absolute().parent.as_posix()
sys.path.insert(0, f"{current_dir}")
os.chdir(current_dir)

from utils.loadMedia import loader
from utils.denoiseAudio import denoiser
from utils.sliceAudio import Slicer


class Audio_Processing:
    '''
    0. Load the audio
    1. Denoise the audio
    2. Slice off the silent parts
    '''
    mediaExtensions = ['*.flac', '*.wav', '*.mp3', '*.aac', '*.m4a', '*.wma', '*.aiff', '*.au', '*.ogg', '*.mp4', '*.flv', '*.mkv', '*.avi']
    audioExtensions = ['*.flac', '*.wav', '*.mp3', '*.aac', '*.m4a', '*.wma', '*.aiff', '*.au', '*.ogg']

    subtypeDict = {
        '8':          'PCM_8',
        '16':         'PCM_16',
        '24':         'PCM_24',
        '32':         'PCM_32',
        '32 (Float)': 'FLOAT'
    }

    def __init__(self,
        inputMedia: str,
        outputFormat: Optional[str] = 'wav',
        sampleRate: Optional[Union[int, str]] = None,
        sampleWidth: Optional[Union[int, str]] = None,
        toMono: bool = False,
        denoiseAudio: bool = True,
        denoiseModelPath: str = "",
        denoiseTarget: str = '',
        sliceAudio: bool = True,
        rmsThreshold: float = -40.,
        audioLength: int = 5000,
        silentInterval: int = 300,
        hopSize: int = 10,
        silenceKept: int = 1000,
        outputRoot: str = "./",
        outputDirName: str = "",
    ):
        self.inputMedia = inputMedia
        self.outputFormat = outputFormat.lower() if outputFormat is not None else None
        self.denoiseAudio = denoiseAudio
        self.denoiseModelPath = denoiseModelPath
        self.denoiseTarget = denoiseTarget.lower().replace('人声', 'vocals').replace('背景声', 'instrument')
        self.sliceAudio = sliceAudio
        self.rmsThreshold = rmsThreshold
        self.audioLength = audioLength
        self.silentInterval = silentInterval
        self.hopSize = hopSize
        self.silenceKept = silenceKept
        self.sampleRate = eval(sampleRate) if sampleRate is not None else None
        self.sampleWidth = str(sampleWidth) if sampleWidth is not None else None
        self.toMono = toMono
        self.outputDir = Path(outputRoot).joinpath(outputDirName).as_posix()

        os.makedirs(self.outputDir, exist_ok = True)

    def getPatterns(self,
        directory: str,
        extensions: list
    ):
        patternList = []

        for extension in extensions:
            patternList.extend(glob.glob(Path(directory).joinpath(extension).as_posix()))

        return patternList

    def processMedia(self,
        filePath: str
    ):
        '''
        loader: Load audio from media files which supported by ffmpeg.
        denoiser: WIP
        Slicer: Once the valid (sound) part reached min length since last slice and a silent part longer than min interval are detected, the audio will be sliced apart from the frame(s) with the lowest RMS value within the silent area.
        Long silence parts may be deleted.
        '''
        if self.outputFormat is not None:
            if f'*.{self.outputFormat}'.lower() not in self.audioExtensions:
                raise Exception(f"Format '{self.outputFormat}' is currently not supported!")
        else:
            self.outputFormat = Path(filePath).suffix.strip('.')

        outputName_media = os.path.splitext(os.path.basename(filePath))[0] + '.' + self.outputFormat
        outputPath_media = os.path.join(self.outputDir, outputName_media)
        inputName_audio, inputPath_audio = outputName_media, outputPath_media
        audioData, sampleRate = loader(path = filePath, sr = self.sampleRate, mono = self.toMono)

        writeParamsList = [(inputPath_audio, audioData.T if len(audioData.shape) > 1 else audioData, int(sampleRate))] # .T: Swap axes if the audio is stereo

        if self.denoiseAudio:
            writeParamsList.clear()
            audioData, sampleRate = denoiser(
                audioData,
                sampleRate,
                modelPath = self.denoiseModelPath,
                target = self.denoiseTarget
            )
            outputName_audio = inputName_audio.rsplit('.', 1)[0] + '_Denoised_' + '.' + self.outputFormat
            outputPath_audio = os.path.normpath(os.path.join(self.outputDir, outputName_audio))
            writeParamsList.append((outputPath_audio, audioData.T if len(audioData.shape) > 1 else audioData, int(sampleRate)))

        if self.sliceAudio:
            slicer = Slicer(
                samplingRate = sampleRate,
                rmsThreshold = self.rmsThreshold,
                audioLength_min = self.audioLength,
                silentInterval_min = self.silentInterval,
                hopSize = self.hopSize,
                silenceKept_max = self.silenceKept
            )
            chunks, isSlicingNeeded = slicer.slice(audioData)
            if isSlicingNeeded == True:
                writeParamsList.clear()
                for i, chunk in enumerate(chunks):
                    outputName_audio = inputName_audio.rsplit('.', 1)[0] + f'_Sliced_{i}' + '.' + self.outputFormat
                    outputPath_audio = os.path.normpath(os.path.join(self.outputDir, outputName_audio))
                    writeParamsList.append((outputPath_audio, chunk.T if len(chunk.shape) > 1 else chunk, int(sampleRate)))
                try:
                    os.remove(inputPath_audio)
                except OSError:
                    pass
            else:
                pass

        for WriteParams in writeParamsList:
            soundfile.write(
                *WriteParams,
                subtype = str(self.subtypeDict.get(self.sampleWidth)) if self.sampleWidth is not None else None
            )

    def processAudio(self):
        print('Processing media...')

        with ThreadPoolExecutor(max_workers = os.cpu_count() if not self.denoiseAudio else 1) as Executor:
            Executor.map(
                self.processMedia,
                self.getPatterns(self.inputMedia, self.mediaExtensions) if Path(self.inputMedia).is_dir() else [self.inputMedia]
            )

        print('Finished processing.')