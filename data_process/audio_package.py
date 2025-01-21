'''
FilePath: /Dodrio/data_process/audio_package.py
Descripttion: 
Author: chenyixiang
version: 
Date: 2025-01-15 20:23:36
LastEditors: chenyixiang
LastEditTime: 2025-01-21 14:19:32
'''

import os
from scipy.io import wavfile
import pandas as pd
import multiprocessing
import pyarrow.parquet as pq
import numpy as np
import librosa
import glob
import sys

from io import BytesIO
from tinytag import TinyTag # for read mp3 metainfo

from tqdm import tqdm

class DataPackage:
    def __init__(self, mid_name, wav_dir, parquet_dir, pack_dir, file_type='wav', num_processes=1, num_utts_per_parquet=1000, sample_rate=48000):
        self.mid_name = mid_name
        self.wav_dir = wav_dir
        self.parquet_dir = parquet_dir
        self.pack_dir = pack_dir
        self.file_type = file_type

        self.num_processes = num_processes
        self.num_utts_per_parquet = num_utts_per_parquet
        self.sample_rate = sample_rate

        self.audio_pos = {}

    def set_wav_dir(self, wav_dir):
        self.wav_dir = wav_dir
    
    def create_dir(self):
        os.makedirs(self.parquet_dir, exist_ok=True)
        os.makedirs(self.pack_dir, exist_ok=True)

    def set_wavlist(self, wav_dir):
        def get_file_list(inp_dir, suffix='.wav'):
            itm = []
            for home, dirs, files in os.walk(inp_dir):
                itm.append( map(lambda fname: home + '/' + fname,
                    list( filter( lambda filename: os.path.splitext(filename)[1] == suffix,
                    files) ) ) )
            file_list = [ele for ii in itm for ele in ii]
            return file_list
        suffix = '.'+self.file_type
        wavlist = get_file_list(wav_dir, suffix)
        wavdict = {}
        uttlist = []
        for wavpath in tqdm(wavlist, desc='SetList'):
            (path, filename) = os.path.split(wavpath)
            basename = filename.split(suffix)[0]
            wavdict[basename] = wavpath
            uttlist.append(basename)
        return wavdict, uttlist

    ############## Wav to Parquet ###############

    def save_parquet(self, wavinfo_dict, savelist, parquet_fn):
        sr_list = [wavinfo_dict[x][0] for x in savelist]
        dtype_list = [wavinfo_dict[x][1] for x in savelist]
        audio_list = [wavinfo_dict[x][2] for x in savelist]
        df = pd.DataFrame()
        df['utt'] = savelist
        df['sample_rate'] = sr_list
        df['dtype'] = dtype_list
        df['audio_data'] = audio_list
        df.to_parquet(parquet_fn)
        print(f"{parquet_fn} had be save")

    def get_mp3_metainfo(self,mp3file):
        tag = TinyTag.get(mp3file)
        return tag.samplerate

    def gen_parquet(self, wav_dir, parquet_dir):
        wavdict, uttlist = self.set_wavlist(wav_dir)

        wavinfo_dict ={}
        for utt in tqdm(uttlist, desc='LoadAudio'):
            wavpath = wavdict[utt]
            if self.file_type == 'wav':
                sr, wav = wavfile.read(wavpath)
                if len(wav)<1:
                    uttlist.remove(utt)
                    print(f"{utt} wavfile is None")
                    continue
                if len(wav.shape) > 1:
                    wav = wav[:,0]
                    print(f"{str(len(wav.shape))} channel is non-mono channel, just first channel will be save")
                dtype = str(wav.dtype)
                wavinfo_dict[utt] = [sr, dtype, wav]
            elif self.file_type == 'mp3':
                byte_mp3_data = open(wavpath, 'rb').read()
                sr = self.get_mp3_metainfo(wavpath)
                dtype = 'mp3'
                wavinfo_dict[utt] = [sr, dtype, byte_mp3_data]
            else:
                print("Now just accept mp3 and wav format")
                return

        # Using process pool to speedup
        prefix = self.file_type
        pool = multiprocessing.Pool(processes=self.num_processes)
        parquet_list = []
        parquet2utt = {}
        for i, j in enumerate(range(0, len(uttlist), self.num_utts_per_parquet)):
            pfile = prefix + '_' + self.mid_name + '_{:05d}.parquet'.format(i)
            parquet_file = os.path.join(parquet_dir, pfile)
            parquet_list.append(parquet_file)
            parquet2utt[pfile] = uttlist[j: j + self.num_utts_per_parquet] 
            #self.save_parquet(wavinfo_dict, uttlist[j: j + self.num_utts_per_parquet], parquet_file)
            pool.apply_async(self.save_parquet, (wavinfo_dict, uttlist[j: j + self.num_utts_per_parquet], parquet_file))
        pool.close()
        pool.join()

        utt2parquet_file = os.path.join(parquet_dir, 'utt2parquet.list') 
        u2pf = open(utt2parquet_file, 'w')
        for pak in parquet2utt.keys():
            for utt in parquet2utt[pak]:
                outline = utt+'|' + pak + '\n'
                u2pf.write(outline)
        u2pf.close()
    
    ############## Parquet to Wav ###############

    def parquet2wav(self, parquet_file, wav_dir): 
        ftype = os.path.split(parquet_file)[-1].split('_')[0]
        df = pq.read_table(parquet_file).to_pandas()
        basename = os.path.split(parquet_file)[-1].split('.parquet')[0]
        for idx in tqdm(range(len(df)), desc=f'{basename} Processing'):
            utt = df.iloc[idx]['utt']
            sr = df.iloc[idx]['sample_rate']
            dtype = df.iloc[idx]['dtype']
            audio = df.iloc[idx]['audio_data']
            if ftype == 'wav':
                wavpath = os.path.join(wav_dir, utt+'.wav')
                wavfile.write(wavpath, sr, audio)
            else:
                wavpath = os.path.join(wav_dir, utt+'.'+ftype)
                with open(wavpath, 'wb') as ww:
                    ww.write(audio)

    def dir_parquet2wav(self, parquet_dir, wav_dir):
        os.makedirs(wav_dir, exist_ok=True)
        pq_file_list = glob.glob(parquet_dir+'/*.parquet')
        for parquet_file in pq_file_list:
            self.parquet2wav(parquet_file, wav_dir)

    ############## Parquet to Package ###############   

    def audio_regular(self, audio, sr, dtype):
        if dtype == 'int16':
            dnum = 32768
        elif dtype == 'int32':
            dnum = 2147483648
        else:
            print(f"{dtype} is not normal data type")
        norm_audio = audio / dnum
        rs_audio = librosa.resample(norm_audio, orig_sr=sr, target_sr=self.sample_rate)
        int16_audio = (rs_audio*32768).astype(np.int16)
        return int16_audio

    def load_mp3(self, bio):
        audio, ori_sr = librosa.load(BytesIO(bio))
        rs_audio = librosa.resample(audio, orig_sr=ori_sr, target_sr=self.sample_rate)
        int16_audio = (rs_audio*32768).astype(np.int16)
        return int16_audio
    
    def parquet2package(self, parquet_file, package_file):
        ftype = os.path.split(parquet_file)[-1].split('_')[0]
        df = pq.read_table(parquet_file).to_pandas()
        basename = os.path.split(parquet_file)[-1].split('.parquet')[0]
        position = 0
        info_list = []
        outf = open(package_file, 'wb')
        for idx in tqdm(range(len(df)), desc=f'{basename} Processing'):
            utt = df.iloc[idx]['utt']
            sr = df.iloc[idx]['sample_rate']
            dtype = df.iloc[idx]['dtype']
            audio = df.iloc[idx]['audio_data']
            if ftype == 'wav':
                reg_audio = self.audio_regular(audio, sr, dtype)
            elif ftype == 'mp3':
                reg_audio = self.load_mp3(audio) 
            else:
                print("Now just accept mp3 and wav format")
                return

            byte_audio = bytes(reg_audio)
            outf.write(byte_audio)

            byte_num = len(reg_audio)* 2 
            end_position = position+byte_num 
            self.audio_pos[utt] = [position, end_position]
            info_list.append([utt, os.path.split(package_file)[-1], str(position), str(end_position)])

            position += byte_num
        return info_list

    def dir_parquet2package(self, parquet_dir, package_dir):
        pq_file_list = glob.glob(parquet_dir+'/*.parquet')
        pq_file_list.sort()
        all_info_list = []
        for one_pq in pq_file_list:
            basename = os.path.split(one_pq)[-1].split('.parquet')[0]
            ftype = os.path.split(one_pq)[-1].split('_')[0]
            pack_file = os.path.join(package_dir, basename+'.pack')
            info_list = self.parquet2package(one_pq, pack_file)
            all_info_list.append(info_list)
        info_outfile = os.path.join(package_dir, 'uttinfo.list')
        with open(info_outfile, 'w') as outf:
            for info_list in all_info_list:
                for info in info_list:
                    outline = '|'.join(info) + '\n'
                    outf.write(outline)
    
    ############## Package to Wav ############### 

    def load_data_from_package(self, pack_file, start, end):
        with open(pack_file, 'rb') as pf :
            pf.seek(start)
            data = pf.read(end-start)
        audio = np.frombuffer(data, dtype=np.int16)
        return audio
    def save_wav(self, audio, wavpath, sr):
        wavfile.write(wavpath, sr, audio)
        
    def package2wav(self, package_dir, wav_dir):
        os.makedirs(wav_dir, exist_ok=True)
        info_file = os.path.join(package_dir, 'uttinfo.list')
        with open(info_file, 'r') as iff:
            info_list = iff.readlines()
        for info in tqdm(info_list):
            utt, pf, start, end = info.split('|')
            start = int(start)
            end = int(end)
            wpf = os.path.join(package_dir, pf)
            audio = self.load_data_from_package(wpf, start, end)
            wavpath = os.path.join(wav_dir, utt+'.wav')
            self.save_wav(audio, wavpath, self.sample_rate)
    
    ################# other info save ####################

if __name__ == "__main__":
    wav_dir = sys.argv[1]
    parquet_dir = sys.argv[2]
    package_dir = sys.argv[3]

    dp = DataPackage(wav_dir, parquet_dir, package_dir) 
