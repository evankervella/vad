import pandas as pd
import numpy as np
from scipy.io import wavfile
import pathlib
from pathlib import Path
import os
from tqdm import tqdm

def create_lists_files(list_dir: list) -> (list, list):
    list_dir.sort()
    list_wav = [f for f in list_dir if f.endswith('.wav')]
    list_json = [f for f in list_dir if f.endswith('.json')]
    return(list_wav, list_json)

def initiate_data(path_dir: pathlib.PosixPath, list_wav: list, list_json: list, index: int) -> (pd.DataFrame, str):
    fs, wav_data = wavfile.read(path_dir / list_wav[index])
    df = pd.DataFrame({'value': wav_data})
    df['timestamp'] = df.index / fs
    df['target'] = 0
    json_data = pd.read_json(path_dir / list_json[index])
    return(df, json_data)

def add_target(df: pd.DataFrame, json_data: str) -> pd.DataFrame:
    targets = np.zeros(df.shape[0])
    for idx, row in json_data.iterrows():
        targets[df[(df['timestamp'] >= row['speech_segments']['start_time']) & (df['timestamp'] <= row['speech_segments']['end_time'])]['target'].index] = 1
    df['target'] = targets.astype(int)
    df = df.set_index('timestamp')
    return(df)

if __name__ == "__main__":

    path_dir = Path('/Users/ekervella/Dropbox/GitHub/vad/vad_data').resolve(strict=True)
    list_dir = os.listdir(path_dir)
    list_wav, list_json = create_lists_files(list_dir)
    for index in tqdm(range(len(list_wav))):
        df, json_data = initiate_data(path_dir, list_wav, list_json, index)
        df = add_target(df, json_data)
        df_name = os.path.splitext(list_json[index])[0]+'.csv'
        df.to_csv(path_dir / 'csv/' / df_name)