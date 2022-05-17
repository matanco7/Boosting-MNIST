
import numpy as np
import matplotlib.pyplot as plt 

X_train = np.loadtxt(fname='MNIST_train_images.csv', delimiter=',')
y_train = np.loadtxt(fname='MNIST_train_labels.csv', delimiter=',')

X_test = np.loadtxt(fname='MNIST_test_images.csv', delimiter=',')
y_test = np.loadtxt(fname='MNIST_test_labels.csv', delimiter=',')

# plt.imshow(np.asarray(np.reshape(X_train[np.random.randint(0,X_train.shape[0]), :], (28, 28))), cmap='gray', vmin=0, vmax=255)

def theta_array(data):
    arr = []
    for i in range(len(data[0])):
        x = np.unique(data[:, i])
        arr.append(x)
    return arr

def min_learner_new_data(data, label, theta, idx, mode):
    hypo_vec = ((data[:, idx] > theta) * 2 - 1)
    res_vec = (hypo_vec != label) * 1
    return res_vec

def create_error_dict(dict1, data, label, theta_array):
    m = len(data)
    for pixel in range(len(data[0])):
        list1 = []
        for theta in theta_array[pixel]:
            list1.append(min_learner_new_data(data, label, theta, pixel, "high"))
        dict1[pixel] = np.array(list1)

def find_min_hypo(dict_hypo1, theta_array, data, p_vec):
    min_error = 10
    for pixel in range(len(data[0])): # go over all pixels
        err_mat = np.asarray([dict_hypo1[pixel], 1-dict_hypo1[pixel]])
        hypo_mat = err_mat @ p_vec
        prev_error = min_error
        min_error = min(min_error, hypo_mat.min())
        if min_error < prev_error:
            min_loc = np.argwhere(hypo_mat == np.min(hypo_mat))
            min_idx = pixel
            min_theta = theta_array[pixel][min_loc[0][1]]
            min_mode = "high" if min_loc[0][0] == 0 else "low"
    return min_error, min_idx, min_theta, min_mode

def update_p_vec(data, label, min_error, min_idx, min_theta, min_mode, p_vec):
    e_t = min_error
    res = (1-e_t)/e_t
    alpha_t = 0.5 * np.log(res)
    label_array = np.asarray(label)
    hypo_vec = ((data[:, min_idx] > min_theta) * 2) - 1
    if min_mode == "high":
        hypo_array = np.array(hypo_vec)
    else:
        hypo_array = np.asarray(hypo_vec * -1)
    p_element_val = p_vec * np.exp(-alpha_t * label_array * hypo_array)
    new_p_vec = p_element_val / p_element_val.sum()
    return new_p_vec, alpha_t, hypo_array

def adaboost(dict_hypo1, data, label, theta_array, p_vec, T):
    num_of_errors = []
    h_vec = []
    alpha_vec = []
    for i in range(T):
        min_err, min_idx, min_theta, min_mode = find_min_hypo(dict_hypo1, theta_array, data, p_vec)
        p_vec, alpha, h_t = update_p_vec(data, label, min_err, min_idx, min_theta, min_mode, p_vec)
        alpha_vec.append(alpha)
        h_vec.append(h_t)
        h_mat = np.array(h_vec).T
        alpha_mat = np.array(alpha_vec)
        y_hat = np.sign(h_mat @ alpha_mat)
        num_of_errors.append(sum(y_hat != label))
        print("finished iter ", i)
    return num_of_errors


T = 30
#Training implementation 
        
print("Training data using Adaboost ")
set_size = len(X_train)
set_theta_array = theta_array(X_train)
set_dict = {}
create_error_dict(set_dict, X_train, y_train, set_theta_array)
print("finished learning all pixel errors for different thetas in set data")
print("start looking for min hypo")
p_vec = np.ones(set_size)/set_size
train_error = adaboost(set_dict, X_train, y_train, set_theta_array, p_vec, T)
       

#Testing implementation

print("Testing data using Adaboost")
set_size = len(X_test)
set_theta_array = theta_array(X_test)
set_dict = {}
create_error_dict(set_dict, X_test, y_test, set_theta_array)
print("finished learning all pixel errors for different thetas in set data")
print("start looking for min hypo")
p_vec = np.ones(set_size)/set_size
test_error = adaboost(set_dict, X_test, y_test, set_theta_array, p_vec, T)

print('Train errors:', train_error[-1], 'Test errors', test_error[-1])
plt.figure()
plt.plot(range(T), train_error, color='blue', marker='o', markerfacecolor='black', label="Train")
plt.plot(range(T), test_error, color='red', marker='o', markerfacecolor='black', label="Test")
plt.title(" Number of errors in each AdaBoost iteration")
plt.xlabel(" T - Iteratons")
plt.ylabel("Errors")
plt.legend(['Train', 'Test'])
plt.show()







