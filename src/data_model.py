from sklearn import preprocessing
import pandas as pd
import numpy as np

class DataModel:
    def __init__(self, data_paths):
        self.train_data_path = data_paths['train_data_path']
        self.test_data_path = data_paths['test_data_path']
        self.remaining_cycle_data_path = data_paths['remaining_cycle_data_path']
        self.train_df=[]
        self.test_df=[]
        self.truth_df=[]

    def load_data(self):

        # read training data - It is the aircraft engine run-to-failure data.
        self.train_df = pd.read_csv(self.train_data_path, sep=" ", header=None)
        self.train_df.drop(self.train_df.columns[[26, 27]], axis=1, inplace=True)
        self.train_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                            's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                            's15', 's16', 's17', 's18', 's19', 's20', 's21']

        self.train_df = self.train_df.sort_values(['id','cycle'])

        # read test data - It is the aircraft engine operating data without failure events recorded.
        self.test_df = pd.read_csv(self.test_data_path, sep=" ", header=None)
        self.test_df.drop(self.test_df.columns[[26, 27]], axis=1, inplace=True)
        self.test_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                            's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                            's15', 's16', 's17', 's18', 's19', 's20', 's21']

        # read ground truth data - It contains the information of true remaining cycles for each engine in the testing data.
        self.truth_df = pd.read_csv(self.remaining_cycle_data_path, sep=" ", header=None)
        self.truth_df.drop(self.truth_df.columns[[1]], axis=1, inplace=True)



    def label_data(self, w1, w0):
        # Data Labeling - generate column RUL(Remaining Usefull Life or Time to Failure)
        rul = pd.DataFrame(self.train_df.groupby('id')['cycle'].max()).reset_index()
        rul.columns = ['id', 'max']
        self.train_df = self.train_df.merge(rul, on=['id'], how='left')
        self.train_df['RUL'] = self.train_df['max'] - self.train_df['cycle']
        self.train_df.drop('max', axis=1, inplace=True)
        # generate label columns for training data
        # we will only make use of "label1" for binary classification, 
        # while trying to answer the question: is a specific engine going to fail within w1 cycles?

        self.train_df['label1'] = np.where(self.train_df['RUL'] <= w1, 1, 0 )
        self.train_df['label2'] = self.train_df['label1']
        self.train_df.loc[self.train_df['RUL'] <= w0, 'label2'] = 2

        # MinMax normalization (from 0 to 1)
        self.train_df['cycle_norm'] = self.train_df['cycle']
        cols_normalize = self.train_df.columns.difference(['id','cycle','RUL','label1','label2'])
        min_max_scaler = preprocessing.MinMaxScaler()
        norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(self.train_df[cols_normalize]), 
                                    columns=cols_normalize, 
                                    index=self.train_df.index)
        join_df = self.train_df[self.train_df.columns.difference(cols_normalize)].join(norm_train_df)
        self.train_df = join_df.reindex(columns = self.train_df.columns)

        # TEST
        ######
        # MinMax normalization (from 0 to 1)
        self.test_df['cycle_norm'] = self.test_df['cycle']
        norm_test_df = pd.DataFrame(min_max_scaler.transform(self.test_df[cols_normalize]), 
                                    columns=cols_normalize, 
                                    index=self.test_df.index)
        test_join_df = self.test_df[self.test_df.columns.difference(cols_normalize)].join(norm_test_df)
        self.test_df = test_join_df.reindex(columns = self.test_df.columns)
        self.test_df = self.test_df.reset_index(drop=True)


        # We use the ground truth dataset to generate labels for the test data.
        # generate column max for test data
        rul = pd.DataFrame(self.test_df.groupby('id')['cycle'].max()).reset_index()
        rul.columns = ['id', 'max']
        self.truth_df.columns = ['more']
        self.truth_df['id'] = self.truth_df.index + 1
        self.truth_df['max'] = rul['max'] + self.truth_df['more']
        self.truth_df.drop('more', axis=1, inplace=True)

        # generate RUL for test data
        self.test_df = self.test_df.merge(self.truth_df, on=['id'], how='left')
        self.test_df['RUL'] = self.test_df['max'] - self.test_df['cycle']
        self.test_df.drop('max', axis=1, inplace=True)

        # generate label columns w0 and w1 for test data
        self.test_df['label1'] = np.where(self.test_df['RUL'] <= w1, 1, 0 )
        self.test_df['label2'] = self.test_df['label1']
        self.test_df.loc[self.test_df['RUL'] <= w0, 'label2'] = 2

    def generate_sequence_array(self, sequence_length, sequence_cols):
        # generator for the sequences
        seq_gen = (list(self._gen_sequence(self.train_df[self.train_df['id']==id], sequence_length, sequence_cols)) 
                for id in self.train_df['id'].unique())

        # generate sequences and convert to numpy array
        seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
        
        # generate labels
        label_gen = [self._gen_labels(self.train_df[self.train_df['id']==id], sequence_length, ['label1']) 
                    for id in self.train_df['id'].unique()]
        label_array = np.concatenate(label_gen).astype(np.float32)

        return seq_array, label_array

    # function to reshape features into (samples, time steps, features) 
    def _gen_sequence(self, id_df, seq_length, seq_cols):
        """ Only sequences that meet the window-length are considered, no padding is used. This means for testing
        we need to drop those which are below the window-length. An alternative would be to pad sequences so that
        we can use shorter ones """
        # for one id I put all the rows in a single matrix
        data_matrix = id_df[seq_cols].values
        num_elements = data_matrix.shape[0]
        # Iterate over two lists in parallel.
        # For example id1 have 192 rows and sequence_length is equal to 50
        # so zip iterate over two following list of numbers (0,112),(50,192)
        # 0 50 -> from row 0 to row 50
        # 1 51 -> from row 1 to row 51
        # 2 52 -> from row 2 to row 52
        # ...
        # 111 191 -> from row 111 to 191
        for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
            yield data_matrix[start:stop, :]

    

    # function to generate labels
    def _gen_labels(self, id_df, seq_length, label):
        # For one id I put all the labels in a single matrix.
        # For example:
        # [[1]
        # [4]
        # [1]
        # [5]
        # [9]
        # ...
        # [200]] 
        data_matrix = id_df[label].values
        num_elements = data_matrix.shape[0]
        # I have to remove the first seq_length labels
        # because for one id the first sequence of seq_length size have as target
        # the last label (the previus ones are discarded).
        # All the next id's sequences will have associated step by step one label as target. 
        return data_matrix[seq_length:num_elements, :]
    
