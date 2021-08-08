import pyaudio
import numpy as np

def get_audio(stream, all_audio_keypoints, CHUNK, RATE):
    data = np.fromstring(stream.read(CHUNK, exception_on_overflow=False),dtype=np.int16)
    data = data * np.hanning(len(data)) # smooth the FFT by windowing data
    fft = abs(np.fft.fft(data).real)
    fft = fft[:int(len(fft)/2)] # keep only first half
    freq = np.fft.fftfreq(CHUNK,1.0/RATE)
    freq = freq[:int(len(freq)/2)] # keep only first half
    freqPeak = freq[np.where(fft==np.max(fft))[0][0]]+1
    amp=np.average(np.abs(data))*2
    all_audio_keypoints['frequency'].append(freqPeak)
    all_audio_keypoints['amplitude'].append(amp)
    return

def process_audio(key, counter_head, counter_tail, all_audio_keypoints):
    arr = np.array(all_audio_keypoints[key][counter_head:counter_tail+1])
    mean = np.nanmean(arr)
    var = np.nanvar(arr)
    return mean, var