from scipy.io import loadmat
import scipy.io as scio
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
import time

# import setting
alpha_list = 0.01
bootstrap_num = 1000
list_path = r'...\Brain_environment_association\Control_association\early_development\time_all\New_data\Resample_bootstrap\bootstrap_list.mat'
list_data = loadmat(list_path)
list_data = list_data['bootstrap_list']
list_data = np.array(list_data)
# import cognition
y_path = r'...\Brain_environment_association\Control_association\early_development\time_all\New_data\new_mat.mat'
y_data = loadmat(y_path)
y_data = y_data['new_mat']
y_data_all = np.array(y_data)

start_time = time.time()
X_path = r'...\Brain_environment_association\Control_association\early_development\time_all\New_data\input_data.mat'
X_data = loadmat(X_path)
X_data = X_data['input_data']
X_data_all = np.array(X_data)
if y_data_all.ndim == 1:
    n_iter = 1
else:
    n_iter = y_data_all.shape[1]
for beh_ind in range(n_iter):
    All_para = []
    All_r2 = []
    # bootstrap
    for i in range(bootstrap_num):
        first_index = list_data[:, i]
        X_data = X_data_all[first_index, :]
        indices = np.random.choice(range(len(X_data)), size=len(X_data), replace=True)
        X_data = X_data[indices, :]
        y_bootstrap = y_data_all[first_index, :]
        y_bootstrap = y_bootstrap[indices, :]
        y_data = y_bootstrap[:, beh_ind]
        # start
        model = Lasso(alpha=alpha_list, precompute=True, max_iter=1000, tol=1e-5, selection='random')
        model.fit(X_data, y_data)
        # parameter
        coefficients = model.coef_
        All_para.append(coefficients)
        # r2
        y_pred = model.predict(X_data)
        r2 = r2_score(y_data, y_pred)
        All_r2.append(r2)
    # save
    All_para = np.array(All_para)
    save_path = r'...\Brain_environment_association\Control_association\early_development\time_all\New_data\result\All_para_%s.mat' % (beh_ind+1)
    scio.savemat(save_path, {'All_para': All_para})
    All_r2 = np.array(All_r2)
    save_path = r'...\Brain_environment_association\Control_association\early_development\time_all\New_data\result\All_r2_%s.mat' % (beh_ind+1)
    scio.savemat(save_path, {'All_r2': All_r2})
end_time = time.time()
elapsed_time = end_time - start_time
print(elapsed_time)
