import train
from tensorflow.keras.models import load_model
from preprocess_f import return_data
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report


feature_list = ['Wrpm', 'We', 'Tref', 'Test', 'TmpM', 'InvLam', 'idss_act', 'iqss_act', 'vdss_ref', 'vqss_ref', 'vdss_act', 'vqss_act',
                    'Roffset', 'Vdc', 'TmpInv', 'TmpIGBT', 'Tmea', 'Wmea', 'idsr_ref', 'iqsr_ref', 'idsr_act(LPF)', 'iqsr_act(LPF)',
                    'idsr_act', 'iqsr_act', 'vdsr_ref', 'vqsr_ref', 'vdsr_act(LPF)', 'vqsr_act(LPF)', 'theta_e', 'Rsin', 'Rcos', 'iu',
                    'iv', 'iw', 'InvSF']


def predict(test_data, model_best, test_y, train_data):
    best_model = load_model(model_best)
    train_pred = best_model.predict(train_data)
    print("=="*70)
    print(">>>> np.abs(train_pred - train_data) = ", np.abs(train_pred - train_data).shape)#23520, 35
    print("=="*70)
    #diff = np.abs(train_pred - train_data)
    max_predicted_all_label, mean_predicted_all_label = [], []
    
    for i in range(0, 35):
        #ab = np.abs(train_pred[:, i] - train_data[:, i])
        #print("ab.shape = ", ab.shape)
        train_mse_loss = (train_pred[:, i] - train_data[:, i])**2
        #train_mae_loss = np.mean(np.abs(train_pred[:, i] - train_data[:, i]), axis=0)
        print("i = ", i)
        print("train_mse_loss.shape = ", train_mse_loss)
        #print("=="*70)
        #print(">>>> train_data.shape = ", train_data.shape)#23520, 35
        #print("=="*70)
        #print(">>>> train_mae_loss")
        #print(train_mae_loss)
        #print(len(train_mae_loss))#23520
        #print("=="*70)
        plt.title(feature_list[i])
        plt.hist(train_mse_loss, bins=50)
        plt.xlabel('Train MSE Loss')
        plt.ylabel("Number of samples")
        #plt.show()

        max_train_threshold = np.max(train_mse_loss)
        threshold = np.mean(train_mse_loss)
        print("=="*70)
        print("Reconstruction train error threshold(mean) = ", threshold, 2)
        print("=="*70)
        print("Reconstruction train error threshold(max) = ", max_train_threshold)
        print("=="*70)
        test_pred = best_model.predict(test_data)
        print("=="*70)
        print(">>>> test_pred")
        print("test_pred.shape = ", test_pred.shape)#2520, 35
        #print(test_pred)
        test_mse_loss = (test_pred[:, i] - test_data[:, i])**2
        print("=="*70)
        print(">>>> test_mse_loss")
        #print(test_mse_loss)
        print(len(test_mse_loss))#2520
        print("=="*70)

        plt.hist(test_mse_loss, bins=50)
        plt.xlabel('Test MSE Loss')
        plt.ylabel("Number of samples")
        #plt.show()

        test_threshold = np.max(test_mse_loss)
        mean_test_threshold = np.mean(test_mse_loss)
        print("=="*70)
        print("Reconstruction error about Test Data(mean) = ", mean_test_threshold)
        print("=="*70)
        print("Reconstruction error about Test Data(max) = ", test_threshold)
        print("=="*70)
        #print("T H R E S H O L D = T R A I N - M A X")
        #print("=="*70)
        anomalies = test_mse_loss > max_train_threshold
        max_predicted_label = list(map(int, anomalies))

        #print("=="*70)
        #print("T H R E S H O L D = T R A I N - M E A N")
        anomalies = test_mse_loss > threshold
        #print("=="*70)
        #print("Number = ", np.sum(anomalies))
        #print("=="*70)
        #print("Indices = ", np.where(anomalies.flatten()))
        #print("=="*70)
        #print("anomalies = ", anomalies.flatten())
        #print("=="*70)
        mean_predicted_label = list(map(int, anomalies))
        #print(predicted_label)
        
        max_predicted_all_label.append(max_predicted_label)
        mean_predicted_all_label.append(mean_predicted_label)
        
    return test_y, max_predicted_all_label, mean_predicted_all_label
    #return test_y, max_predicted_label, mean_predicted_label

def score(true_label, predicted_label, file_name, mode):
    f = open("test.txt", 'w')
    for i in range(0, 35):
        R_score = 'ROC AUC score: {:.2f}'.format(roc_auc_score(true_label, predicted_label[i])*100)
        print('ROC AUC score: {:.2f}'.format(roc_auc_score(true_label, predicted_label[i])*100))
        print("confusion matrix = ")
        print(confusion_matrix(true_label, predicted_label[i]))
        print("classification report = ")
        print(classification_report(true_label, predicted_label[i]))
        f.write('\n')
        f.write(file_name)
        f.write('\n')
        f.write(feature_list[i])
        f.write('\n')
        if mode == 'mean':
            f.write(">>>>>>>> MEAN <<<<<<<<")
        elif mode == 'max':
            f.write(">>>>>>> MAX <<<<<<<")
        f.write('\n')
        f.write(R_score)
        f.write('\n')
        f.write(str(confusion_matrix(true_label, predicted_label[i])))
        f.write('\n')
        f.write(str(classification_report(true_label, predicted_label[i])))
        
    f.close()


if __name__ == "__main__":
    #f = open("result", 'w')
    data_op = 'dc_v_scale'
    # dc_v_scale, temper, resolver_offset, uvw_i_offset, inverter_i_scale
    model_op = 'mlp'#'mlp'
    feature_ex = False
    scaler_mode = True
    lda_mode = False
    #d_list = ['Wrpm', 'We', 'Tref', 'Test', 'TmpM', 'InvLam', 'idss_act', 'iqss_act', 'vdss_ref', 'vqss_ref',
    #                  'vdss_act', 'vqss_act', 'Roffset', 'Vdc', 'TmpInv', 'TmpIGBT', 'Tmea', 'Wmea', 'idsr_ref',
    #                  'iqsr_ref', 'idsr_act(LPF)', 'iqsr_act(LPF)', 'idsr_act', 'iqsr_act', 'vdsr_ref', 'vqsr_ref',
    #                  'vdsr_act(LPF)', 'vqsr_act(LPF)', 'theta_e', 'Rsin', 'Rcos', 'iu', 'iv', 'iw', 'InvSF']

    file_name = './best_model/0902/' + 'dc_v_scale_mlp_scal__True_lda_False_Fri Sep  2 21:37:18 2022' + '.h5'
    result_file = file_name.split('.')[1]
    result_file = result_file.split('/')[1] + ' ' + result_file.split('/')[2] + ' ' +result_file.split('/')[3]
    #f.write(result_file)
    
    nx_tr, ny_tr, nx_valid, ny_valid, all_te, test_y = return_data(data_op=data_op, 
                                                                    feature_ex=feature_ex, 
                                                                    scaler_mode=scaler_mode, 
                                                                    lda_mode=lda_mode,
                                                                    model_op=model_op)
    print("all_te.shape = ", all_te.shape)#2520,35
    true, max_pred, mean_pred = predict(all_te, file_name, test_y, nx_tr)
    print("=="*70)
    print("T H R E S H O L D = T R A I N - M E A N")
    print("=="*70)
    score(true, mean_pred, result_file, 'mean')
    print("=="*70)
    print("T H R E S H O L D = T R A I N - M A X")
    print("=="*70)
    score(true, max_pred, result_file, 'max')
    print("##" * 100)
    
    print("result_file = ", result_file)
    #f.close()
        

