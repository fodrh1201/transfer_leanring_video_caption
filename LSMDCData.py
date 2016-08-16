import util
import pandas as pd
import os
import hickle as hkl
import h5py
import numpy as np
import Error

# DataFrame Path

LSMDC_DATAFRAME_PATH = '/data/captiongaze/DataFrame'
TEST_DF_PATH = os.path.join(LSMDC_DATAFRAME_PATH, 'LSMDC_test.csv')
TRAIN_DF_PATH = os.path.join(LSMDC_DATAFRAME_PATH, 'LSMDC_train.csv')
VAL_DF_PATH = os.path.join(LSMDC_DATAFRAME_PATH, 'LSMDC_val.csv')

# Vocabulary Path

LSMDC_VOCABULARY_PATH = '/data/cationgaze/VOCABUALRY'
WORD_MATRIX_PATH = os.path.join(LSMDC_VOCABULARY_PATH, 'word_matrix.hkl')
WORD_TO_INDEX_PATH = os.path.join(LSMDC_VOCABULARY_PATH, 'word_to_index.hkl')
INDEX_TO_WORD_PATH = os.path.join(LSMDC_DATAFRAME_PATH, 'index_to_word.hkl')

# Video Feature Path

VIDEO_FEATURE_PATH = '/data/captiongaze/LSMDC_features'


class LSMDCData:
    """
    Loads and preprocessed data for the LSMDC dataset.
    """
    def __init__(self,
                 network='resnet',
                 layer='res5c',
                 padding=True,
                 clean_str=True,
                 max_length=30):

        self.network = network
        self.layer = layer
        self.padding = padding
        self.clean_str = clean_str
        self.max_length = max_length

        self.video_feature, self.keys = LSMDCData.read_video_features(network=self.network, layer=self.layer)
        self.train_df, self.val_df, self.test_df = LSMDCData.read_df_from_csvfile(self.keys)
        self.word_matrix, self.word_to_index, self.index_to_word = LSMDCData.read_vocabulary_from_hklfile()

        self.train_video, self.val_video, self.test_video = self.split_video_feature()

    def build_data(self):
        train_data = self.preprocessing(self.train_df, self.train_video)
        val_data = self.preprocessing(self.val_df, self.val_video)
        test_data = self.preprocessing(self.test_df, self.test_video)
        return train_data, val_data, test_data

    def split_video_feature(self):
        len_list = [len(self.train_df), len(self.val_df), len(self.test_df)]
        sum_list = [sum(len_list[:i+1]) for i in range(len(len_list))]

        video_features = []
        video_keys = []
        for start, end in zip(sum_list, sum_list[1:]):
            video_features.append(self.video_feature[start:end])
            video_keys.append(self.keys[start:end])
        # return : video_features = [[train_videos], [val_videos], [test_videos]]
        return video_features

    def preprocessing(self, dataframe, video_list):
        descriptions = dataframe.loc[:, 'description']
        roots = dataframe.loc[:, 'root']

        descriptions = [util.clean_str(sent).split() for sent in descriptions]
        descriptions = util.pad_sequences(descriptions, max_length=self.max_length)
        descriptions = [util.convert_sent_to_index(sent) for sent in descriptions]

        roots = [util.clean_root(root) for root in roots]
        return (video_list, descriptions, roots)

    @staticmethod
    def read_df_from_csvfile(keys):
        train_df = pd.read_csv(TRAIN_DF_PATH, sep='\t')
        test_df = pd.read_csv(TEST_DF_PATH, sep='\t')
        val_df = pd.read_csv(VAL_DF_PATH, sep='\t')

        train_df = train_df.set_index('key')
        test_df = test_df.set_index('key')
        val_df = val_df.set_index('key')

        train_keys = filter(lambda x: x in keys, train_df.index)
        test_keys = filter(lambda x: x in keys, test_df.index)
        val_keys = filter(lambda x: x in keys, val_df.index)

        extract_field = ['description', 'root']
        return [train_df.loc[train_keys, extract_field],
                val_df.loc[val_keys, extract_field],
                test_df.loc[test_keys, extract_field]]

    @staticmethod
    def read_vocabulary_from_hklfile():
        with open(WORD_MATRIX_PATH, 'r') as f:
            word_matrix = hkl.load(f)

        with open(WORD_TO_INDEX_PATH, 'r') as f:
            word_to_index = hkl.load(f)

        with open(INDEX_TO_WORD_PATH, 'r') as f:
            index_to_word = hkl.load(f)

        return [word_matrix, word_to_index, index_to_word]

    @staticmethod
    def read_video_features(network='resnet', layer='res5c'):
        feature_path = VIDEO_FEATURE_PATH
        if network.lower() == 'resnet':
            feature_path = os.path.join(VIDEO_FEATURE_PATH, network.upper()+'_'+layer.lower()+'hdf5')
        elif network.lower() == 'googlenet' or network.lower() == 'google':
            feature_path = os.path.join(VIDEO_FEATURE_PATH, 'GOOGLE.hdf5')
        elif network.lower() == 'c3d':
            feature_path = os.path.join(VIDEO_FEATURE_PATH, 'C3D.hdf5')
        else:
            raise Error("network and layer argument error. Cannot read features.")

        video_features = []
        with h5py.File(feature_path, 'r') as hf:
            video_keys = hf.keys()
            video_features = [np.array(hf.get(key)) for key in video_keys]
        return video_features, video_keys
