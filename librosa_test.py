import librosa
from scipy.io import wavfile
from pydub import AudioSegment
import numpy as np

wav_list = [
    "/home/jun/workspace/sampling/wav_concat.wav",
    "/home/jun/workspace/sampling/sampled/VCTK/wav/p279/p279_180.wav",
    "/home/jun/workspace/sampling/sampled/AIHUB/wav/0023_G1A2E7_BGH/0023_G1A2E7_BGH_000004.wav",
    "/home/jun/workspace/sampling/sampled/VCTK/wav/p287/p287_372.wav",
    "/home/jun/workspace/sampling/sampled/VCTK/wav/p288/p288_379.wav",
    "/home/jun/workspace/sampling/sampled/VCTK/wav/p295/p295_102.wav",
    "/home/jun/workspace/sampling/sampled/VCTK/wav/p313/p313_085.wav",
    "/home/jun/workspace/sampling/sampled/VCTK/wav/p293/p293_087.wav",
    "/home/jun/workspace/sampling/sampled/VCTK/wav/p233/p233_365.wav",
    "/home/jun/workspace/sampling/sampled/VCTK/wav/p314/p314_128.wav",
]

warning_list = ["/home/jun/workspace/sampling/sampled/AIHUB/wav/7555_G1A2E7_KJH/7555_G1A2E7_KJH_011619.wav",
                "/home/jun/workspace/sampling/sampled/AIHUB/wav/7555_G1A2E7_KJH/7555_G1A2E7_KJH_007907.wav",
                "/home/jun/workspace/sampling/sampled/AIHUB/wav/7555_G1A2E7_KJH/7555_G1A2E7_KJH_006768.wav"
                ]


# wav_filepath = []
# with open("/data/hun/tts/filelists/multilingual_base_train.txt") as filenames:
#     for filename in filenames:
#         filename = filename.split("|")[0]
#         wav_filepath.append(
#             filename.replace("./dummies/wav/multilingual_base", "/home/jun/workspace/sampling/sampled/VCTK/wav")
#         )
# for wav in wav_filepath[50122:]:
for wav in warning_list:
    y, sr = librosa.load(wav, sr=16000, dtype=np.float32)
    yt, index = librosa.effects.trim(y, top_db=20)
    if index[0] - 8000 < 0:
        start_index = 0
    else:
        start_index = index[0] - 8000
    if index[1] + 8000 > y.shape[0]:
        end_index = y.shape[0]
    else:
        end_index = index[1] + 8000
    output_y = y[start_index:end_index]
    # print(librosa.get_duration(y, sr=16000), librosa.get_duration(yt, sr=16000))
# if librosa.get_duration(y=y, sr=16000) - librosa.get_duration(y=yt, sr=16000) > 2:
#     print(wav)
    wavfile.write("./librosa_test.wav", sr, output_y)
