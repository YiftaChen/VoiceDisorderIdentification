from random import sample
import torch
from torch.nn import functional as F
import torch.nn as nn
import numpy as np
import torchaudio.transforms as ta_trans
from core.params import CommonParams, YAMNetParams
import random

class ToOneHot(nn.Module):
    def __call__(self, classification):
        if not isinstance(classification,bool):
            classification = torch.Tensor([classification]).to(torch.int64)
            classification = F.one_hot(classification,10).squeeze()
        return classification

class ToTensor(nn.Module):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        return torch.from_numpy(sample).float()

class RandomFlip(nn.Module):
    """Randomly flips the sample"""
    def __init__(self,probability=0.5):
        self.probability=probability

    def __call__(self,sample):
        if random.random() < self.probability:
            return np.array(sample[::-1])
        return sample
        
# class RandomFlip(nn.Module):
#     """Randomly flips the sample"""
#     def __init__(probability=0.5):
#         self.probability=probability

#     def __call__(self,sample):
#         if random.random() < self.probability:
#             return sample[::-1]
      

class PadWhiteNoise(nn.Module):
    """Pads white noise to short audio samples."""

    def __call__(self,sample,sr=50000):
        if len(sample)>60000:
            return sample
        
        mean = sample.mean()
        variance = sample.var()
        noise = np.random.normal(mean,variance,60000-len(sample))/10
        signal=np.concatenate((sample,noise))

        return signal

class Truncate(nn.Module):
    def __init__(self,N):
        self.N=int(N)
    def __call__(self, sample):
        return sample[:self.N].reshape(1,-1)

class WaveformToInput(torch.nn.Module):
    def __init__(self):
        super().__init__()
        audio_sample_rate = CommonParams.TARGET_SAMPLE_RATE
        window_length_samples = int(round(
            audio_sample_rate * CommonParams.STFT_WINDOW_LENGTH_SECONDS
        ))
        hop_length_samples = int(round(
            audio_sample_rate * CommonParams.STFT_HOP_LENGTH_SECONDS
        ))
        fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))
        assert window_length_samples == 400
        assert hop_length_samples == 160
        assert fft_length == 512
        self.mel_trans_ope = VGGishLogMelSpectrogram(
            CommonParams.TARGET_SAMPLE_RATE, n_fft=fft_length,
            win_length=window_length_samples, hop_length=hop_length_samples,
            f_min=CommonParams.MEL_MIN_HZ,
            f_max=CommonParams.MEL_MAX_HZ,
            n_mels=CommonParams.NUM_MEL_BANDS
        )
        # note that the STFT filtering logic is exactly the same as that of a
        # conv kernel. It is the center of the kernel, not the left edge of the
        # kernel that is aligned at the start of the signal.

    #TODO change hard coded number to configuration
    def __call__(self, waveform):
        res = self.wavform_to_log_mel(waveform=waveform,sample_rate=CommonParams.SVD_SAMPLE_RATE)[0]
        shape = res.shape
        return res[0].reshape(*shape[1:])
    #     '''
    #     Args:
    #         waveform: torch tsr [num_audio_channels, num_time_steps]
    #         sample_rate: per second sample rate
    #     Returns:
    #         batched torch tsr of shape [N, C, T]
    #     '''
    #     x = waveform.mean(axis=0, keepdims=True)  # average over channels
    #     resampler = ta_trans.Resample(sample_rate, CommonParams.TARGET_SAMPLE_RATE)
    #     x = resampler(x)
    #     x = self.mel_trans_ope(x)
    #     x = x.squeeze(dim=0).T  # # [1, C, T] -> [T, C]

    #     window_size_in_frames = int(round(
    #         CommonParams.PATCH_WINDOW_IN_SECONDS / CommonParams.STFT_HOP_LENGTH_SECONDS
    #     ))
    #     print(CommonParams.PATCH_WINDOW_IN_SECONDS)

    #     num_chunks = x.shape[0] // window_size_in_frames

    #     # reshape into chunks of non-overlapping sliding window
    #     num_frames_to_use = num_chunks * window_size_in_frames
    #     x = x[:num_frames_to_use]
    #     # [num_chunks, 1, window_size, num_freq]
    #     x = x.reshape(num_chunks, 1, window_size_in_frames, x.shape[-1])
    #     return x

    def wavform_to_log_mel(self, waveform, sample_rate):
        '''
        Args:
            waveform: torch tsr [num_audio_channels, num_time_steps]
            sample_rate: per second sample rate
        Returns:
            batched torch tsr of shape [N, C, T]
        '''
        x = waveform.mean(axis=0, keepdims=True)  # average over channels
        resampler = ta_trans.Resample(sample_rate, CommonParams.TARGET_SAMPLE_RATE)
        x = resampler(x)
        x = self.mel_trans_ope(x)
        x = x.squeeze(dim=0).T  # # [1, C, T] -> [T, C]
        spectrogram = x.cpu().numpy().copy()

        window_size_in_frames = int(round(
            CommonParams.PATCH_WINDOW_IN_SECONDS / CommonParams.STFT_HOP_LENGTH_SECONDS
        ))

        if YAMNetParams.PATCH_HOP_SECONDS == YAMNetParams.PATCH_WINDOW_SECONDS:
            num_chunks = x.shape[0] // window_size_in_frames

            # reshape into chunks of non-overlapping sliding window
            num_frames_to_use = num_chunks * window_size_in_frames
            x = x[:num_frames_to_use]
            # [num_chunks, 1, window_size, num_freq]
            x = x.reshape(num_chunks, 1, window_size_in_frames, x.shape[-1])
        else:  # generate chunks with custom sliding window length `patch_hop_seconds`
            patch_hop_in_frames = int(round(
                YAMNetParams.PATCH_HOP_SECONDS / CommonParams.STFT_HOP_LENGTH_SECONDS
            ))
            # TODO performance optimization with zero copy
            patch_hop_num_chunks = (x.shape[0] - window_size_in_frames) // patch_hop_in_frames + 1
            num_frames_to_use = window_size_in_frames + (patch_hop_num_chunks - 1) * patch_hop_in_frames
            x = x[:num_frames_to_use]
            x_in_frames = x.reshape(-1, x.shape[-1])
            x_output = np.empty((patch_hop_num_chunks, window_size_in_frames, x.shape[-1]))
            for i in range(patch_hop_num_chunks):
                start_frame = i * patch_hop_in_frames
                x_output[i] = x_in_frames[start_frame: start_frame + window_size_in_frames]
            x = x_output.reshape(patch_hop_num_chunks, 1, window_size_in_frames, x.shape[-1])
            x = torch.tensor(x, dtype=torch.float32)
        return x, spectrogram

class VGGishLogMelSpectrogram(ta_trans.MelSpectrogram):
    '''
    This is a _log_ mel-spectrogram transform that adheres to the transform
    used by Google's vggish model input processing pipeline
    '''

    def forward(self, waveform):
        r"""
        Args:
            waveform (torch.Tensor): Tensor of audio of dimension (..., time)

        Returns:
            torch.Tensor: Mel frequency spectrogram of size (..., ``n_mels``, time)
        """
        specgram = self.spectrogram(waveform)
        # NOTE at mel_features.py:98, googlers used np.abs on fft output and
        # as a result, the output is just the norm of spectrogram raised to power 1
        # For torchaudio.MelSpectrogram, however, the default
        # power for its spectrogram is 2.0. Hence we need to sqrt it.
        # I can change the power arg at the constructor level, but I don't
        # want to make the code too dirty
        specgram = specgram ** 0.5

        mel_specgram = self.mel_scale(specgram)
        mel_specgram = torch.log(mel_specgram + CommonParams.LOG_OFFSET)
        return mel_specgram
