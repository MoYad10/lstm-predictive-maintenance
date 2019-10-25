import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, LSTM
import os


class SeqModel:
    def __init__(self, sequence_length, seq_array, label_array):
        self.sequence_length = sequence_length
        self.seq_array = seq_array
        self.label_array = label_array
        
        # Next, we build a deep network. 
        self._build_lstm_network()
        self.history=[]
        
    def _build_lstm_network(self):
        # The first layer is an LSTM layer with 100 units followed by another LSTM layer with 50 units. 
        # Dropout is also applied after each LSTM layer to control overfitting. 
        # Final layer is a Dense output layer with single unit and sigmoid activation since this is a binary classification problem.
        # build the network
        nb_features = self.seq_array.shape[2]
        nb_out = self.label_array.shape[1]

        self.model = Sequential()

        self.model.add(LSTM(
                 input_shape=(self.sequence_length, nb_features),
                 units=100,
                 return_sequences=True))
        self.model.add(Dropout(0.2))

        self.model.add(LSTM(units=50, return_sequences=False))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(units=nb_out, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        print(self.model.summary())
        
    def fit(self, model_path, epochs, batch_size, validation_split):
        # fit the network
        self.history = self.model.fit(self.seq_array, self.label_array, epochs=epochs, batch_size=batch_size, validation_split=validation_split, 
                  callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min'),
                               keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=0)]
                  )

        # list all data in history
        print(self.history.history.keys())
        
        
    def plot_training_accuracy(self):
        # summarize history for Accuracy
        fig_acc = plt.figure(figsize=(10, 5))
        plt.plot(self.history.history['acc'])
        plt.plot(self.history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
#         fig_acc.savefig("model_accuracy.png")

        # summarize history for Loss
        fig_acc = plt.figure(figsize=(10, 5))
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
#         fig_acc.savefig("model_loss.png")

    def calc_training_confusion_matrix(self):

        # training metrics
        scores = self.model.evaluate(self.seq_array, self.label_array, verbose=1, batch_size=200)
        print('Accurracy: {}'.format(scores[1]))

        # make predictions and compute confusion matrix
        y_pred = self.model.predict_classes(self.seq_array,verbose=1, batch_size=200)
        y_true = self.label_array

        test_set = pd.DataFrame(y_pred)
    #     test_set.to_csv('/binary_submit_train.csv', index = None)

        print('Confusion matrix\n- x-axis is true labels.\n- y-axis is predicted labels')
        cm = confusion_matrix(y_true, y_pred)
        print(cm)

        # compute precision and recall
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        print( 'precision = ', precision, '\n', 'recall = ', recall)

    def calc_testing_confusion_matrix(self, test_df, model_path, sequence_cols):

        ##################################
        # EVALUATE ON TEST DATA
#         calc_testing_accuracy(test_df)
        ##################################

        # We pick the last sequence for each id in the test data

        seq_array_test_last = [test_df[test_df['id']==id][sequence_cols].values[-self.sequence_length:] 
                               for id in test_df['id'].unique() if len(test_df[test_df['id']==id]) >= self.sequence_length]

        seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)
#         print("seq_array_test_last")
#         print(seq_array_test_last)
#         print(seq_array_test_last.shape)

        # Similarly, we pick the labels

        #print("y_mask")
        # serve per prendere solo le label delle sequenze che sono almeno lunghe 50
        y_mask = [len(test_df[test_df['id']==id]) >= self.sequence_length for id in test_df['id'].unique()]
#         print("y_mask")
#         print(y_mask)
        label_array_test_last = test_df.groupby('id')['label1'].nth(-1)[y_mask].values
        label_array_test_last = label_array_test_last.reshape(label_array_test_last.shape[0],1).astype(np.float32)
        print(label_array_test_last.shape)
#         print("label_array_test_last")
#         print(label_array_test_last)

        # if best iteration's model was saved then load and use it
        if os.path.isfile(model_path):
            estimator = load_model(model_path)

        # test metrics
        scores_test = estimator.evaluate(seq_array_test_last, label_array_test_last, verbose=2)
        print('Accurracy: {}'.format(scores_test[1]))

        # make predictions and compute confusion matrix
        y_pred_test = estimator.predict_classes(seq_array_test_last)
        y_true_test = label_array_test_last

        test_set = pd.DataFrame(y_pred_test)
        test_set.to_csv('binary_submit_test.csv', index = None)

        print('Confusion matrix\n- x-axis is true labels.\n- y-axis is predicted labels')
        cm = confusion_matrix(y_true_test, y_pred_test)
        print(cm)

        # compute precision and recall
        precision_test = precision_score(y_true_test, y_pred_test)
        recall_test = recall_score(y_true_test, y_pred_test)
        f1_test = 2 * (precision_test * recall_test) / (precision_test + recall_test)
        print( 'Precision: ', precision_test, '\n', 'Recall: ', recall_test,'\n', 'F1-score:', f1_test )

        # Plot in blue color the predicted data and in green color the
        # actual data to verify visually the accuracy of the model.
        fig_verify = plt.figure(figsize=(20, 10))
        plt.plot(y_pred_test, color="blue")
        plt.plot(y_true_test, color="green")
        plt.title('prediction')
        plt.ylabel('value')
        plt.xlabel('row')
        plt.legend(['predicted', 'actual data'], loc='upper left')
        plt.show()
        fig_verify.savefig(".model_verify.png")
