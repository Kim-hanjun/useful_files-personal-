import logging
import speech_recognition as sr
from jamo import hangul_to_jamo
from nltk.translate import bleu_score
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import re
import os

r = sr.Recognizer()


def get_stt_result(fpath):
    korean_audio = sr.AudioFile(fpath)

    with korean_audio as source:
        audio = r.record(source)
    result = r.recognize_google(audio_data=audio, language="ko-KR")
    # result = r.recognize_google(audio_data=audio)
    return result


def main():
    LOG_PATH = "./logs/"
    THRESHOLD = 0.6
    logging.basicConfig(
        filename=os.path.join(LOG_PATH, "match.log"),
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s\t%(levelname)s\t%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger()
    LIMIT = 50123
    text_list = []
    filepath_list = []
    with open("/home/hun/data/tts/filelists/multilingual_base_train.txt") as filenames:
        for filename in filenames:
            print(filename)
            wav_filepath = filename.split("|")[0]
            filepath_list.append(wav_filepath.replace("./dummies/wav/multilingual_base", "/data/hun/tts/wav/librosa_load_trim_silence/AIHUB"))
            text = filename.split("|")[3].replace("\n", "")
            # for korean
            text = re.sub("[ .,?!'\"-~…]", "", text)
            # for english
            # text = re.sub("[\ -@\[-`\{-~…]", "", text)
            text_list.append(text)

    for i in tqdm(range(122, LIMIT)):
        try:
            result = get_stt_result(filepath_list[i])
            # for korean
            stt_result = re.sub("[ .,?!'\"-~…]", "", result)
            text_tokens = list(hangul_to_jamo(text_list[i]))
            stt_tokens = list(hangul_to_jamo(stt_result))
            # for english
            # text = re.sub("[\ -@\[-`\{-~…]", "", text)
            # text_tokens = list(set(text_list[i]))
            # stt_tokens = list(set(stt_result))         
            score = bleu_score.sentence_bleu([text_tokens], stt_tokens, weights=(0.5, 0.5))
            if score < THRESHOLD:
                logger.info(f"speech_path: '{filepath_list[i]}'")
                logger.info(f"bleu score: {score:.5f}")
                logger.info(f"origin_text:\t{text_list[i]}")
                logger.info(f"stt_result:\t\t{stt_result}")
                logger.info(f"result: \t\t{result}\n")
        except Exception:
            logger.info(f"{filepath_list[i]}")
            logger.info(f"!!!!!!! STT FAILED !!!!!!!")


if __name__ == "__main__":
    main()
