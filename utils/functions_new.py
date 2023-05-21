import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from keras.regularizers import L1L2
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split

class SimpleMLP:
    @staticmethod
    def build(shape, rate):
        model = Sequential()
        model.add(Dense(int(200*rate), input_shape=(shape,)))
        model.add(Activation("relu"))
        model.add(Dense(int(100*rate), input_shape=(int(200*rate),)))
        model.add(Activation("relu"))
        model.add(Dense(10))
        model.add(Activation("softmax"))
        return model
    def build_100(shape, classes):
        model = Sequential()
        model.add(Dense(20, input_shape=(shape,)))
        model.add(Activation("relu"))
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model

def save_file(file_name,data):
    open_file = open(file_name, "wb")
    pickle.dump(data, open_file)
    open_file.close()

def open_file(file_name):
    open_file = open(file_name, "rb")
    Dataset = pickle.load(open_file)
    open_file.close()
    return Dataset


def batch_data_non_iid(client_data, client_label, bs=32):
    '''Takes in a clients data shard and create a tfds object off it
    args:
        shard: a data, label constituting a client's data shard
        bs:batch size
    return:
        tfds object'''
    #seperate shard into data and labels lists
    data, label = client_data,client_label
    label=tf.keras.utils.to_categorical(label, 10)
    dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))
    return dataset.shuffle(len(label)).batch(bs)

def batch_data_non_iid_new(client_data, client_label, bs=32):
    '''Takes in a clients data shard and create a tfds object off it
    args:
        shard: a data, label constituting a client's data shard
        bs:batch size
    return:
        tfds object'''
    # separate shard into data and labels lists
    id_test = int(len(client_data) * 0.8)
    data, label = client_data[:id_test], client_label[:id_test]
    test_data, test_label = client_data[id_test:-1], client_label[id_test:-1]

    label = tf.keras.utils.to_categorical(label, 10)
    label_test = tf.keras.utils.to_categorical(test_label, 10)  # one-hot encode test labels

    dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))
    test_dataset = tf.data.Dataset.from_tensor_slices((list(test_data), list(label_test)))

    return dataset.shuffle(len(label)).batch(bs), test_dataset.shuffle(len(label_test)).batch(bs)

def batch_data(data_shard, bs=32):
    '''Takes in a clients data shard and create a tfds object off it
    args:
        shard: a data, label constituting a client's data shard
        bs:batch size
    return:
        tfds object'''
    #seperate shard into data and labels lists
    data, label = zip(*data_shard)
    label=tf.keras.utils.to_categorical(label, 10)
    dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))
    return dataset.shuffle(len(label)).batch(bs)

def batch_data_new(data_shard, bs=32):
    '''Takes in a clients data shard and create a tfds object off it
    args:
        shard: a data, label constituting a client's data shard
        bs:batch size
    return:
        tfds object'''
    #seperate shard into data and labels lists
    client_data, client_label = zip(*data_shard)
    id_test = int(len(client_data) * 0.8)
    data, label = client_data[:id_test], client_label[:id_test]
    test_data, test_label = client_data[id_test:-1], client_label[id_test:-1]

    label = tf.keras.utils.to_categorical(label, 10)
    label_test = tf.keras.utils.to_categorical(test_label, 10)  # one-hot encode test labels

    dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))
    test_dataset = tf.data.Dataset.from_tensor_slices((list(test_data), list(label_test)))

    return dataset.shuffle(len(label)).batch(bs), test_dataset.shuffle(len(label_test)).batch(bs)

def batch_data_femnist(data_shard, bs=32):
    data, label = zip(*data_shard)
    dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))
    return dataset.shuffle(len(label)).batch(bs)

class SimpleMLP3:
    @staticmethod
    def build(shape,rate):
        model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_dim=shape),
        tf.keras.layers.Dense(int(64*rate), activation='relu'),
        tf.keras.layers.Dense(10)
        ])
        return model

class SimpleMLP4:
    @staticmethod
    def build(rate):
        model = Sequential()
        model.add(tf.keras.layers.Conv2D(int(32*rate), (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                         input_shape=(32, 32, 3)))
        model.add(tf.keras.layers.Conv2D(int(32*rate), (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Dropout(0.2))

        model.add(tf.keras.layers.Flatten())
        model.add(Dense(int(128*rate), activation='relu', kernel_initializer='he_uniform'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(Dense(10, activation='softmax'))
        return model


def get_submodel(model, percentage_prune):
    r_prune = (1 - percentage_prune)
    m = model
    n = np.array(m) * 0
    for i in range(len(m) - 2):
        if i % 2 == 0:
            x, y = m[i].shape
            y = round(y * r_prune)
            aa = np.transpose(np.array(m[i]))
            bb = np.transpose(n[i])
            bb[:y] = aa[:y]
            cc = np.transpose(bb)
            m[i] = cc
            n[i + 1]
            n[i + 1][:y] = m[i + 1][:y]
            m[i + 1] = n[i + 1]
            n[i + 2]
            n[i + 2][:y] = m[i + 2][:y]
            m[i + 2] = n[i + 2]

    return m

def get_submodel_new(model, node):
    # r_prune = (1 - percentage_prune)
    m = model
    n = np.array(m) * 0
    for i in range(len(m) - 2):
        if i % 2 == 0:
            x, y = m[i].shape
            y = node
            aa = np.transpose(np.array(m[i]))
            bb = np.transpose(n[i])
            bb[:y] = aa[:y]
            cc = np.transpose(bb)
            m[i] = cc
            n[i + 1]
            n[i + 1][:y] = m[i + 1][:y]
            m[i + 1] = n[i + 1]
            n[i + 2]
            n[i + 2][:y] = m[i + 2][:y]
            m[i + 2] = n[i + 2]

    return m

def group_gradient(gradlist, loss_list, group_num, q,qm):
    gm=1/group_num
    nm=len(gradlist)
    pm=1/nm

    bla=(gm/(qm+1))
    sum_val=0
    for i in range(len(loss_list)):
        sum_val+=pm*(loss_list[i]**(qm+1))
    sla= sum_val**((q-qm)/(qm+1))
    coefm= sla*bla

    gradm=gradlist[0]*0
    for i in range(len(gradlist)):
        gradm= gradm+(pm*(qm+1)*loss_list[i]**qm)*gradlist[i]

    gradm=coefm*gradm
    return gradm

def group_hessian_new(initial_weight, gradlist, loss_list, group_num, q,qm,lr):
    gm=1/group_num
    nm=len(gradlist)
    pm=1/nm
    L=1/lr

    bla= (gm/(qm+1))*((q-qm)/(qm+1))
    sum_val=0
    for i in range(len(loss_list)):
        sum_val+=pm*(loss_list[i]**(qm+1))
    sla= sum_val**((q-2*qm-1)/(qm+1))
    coefm1= sla*bla

    gradm=gradlist[0]*0
    for i in range(len(loss_list)):
        gradm= gradm+(pm*(qm+1)*loss_list[i]**qm)*gradlist[i]
    grad_val= norm_grad(gradm)/norm_grad(initial_weight)

    bla2= (gm/(qm+1))
    sum_val=0
    for i in range(len(loss_list)):
        sum_val+=pm*(loss_list[i]**(qm+1))
    sla2= sum_val**((q-qm)/(qm+1))
    coefm2= sla2*bla2

    val=0
    for i in range(len(loss_list)):
        val+= pm*(qm+1)*qm*(loss_list[i]**(qm-1))* (norm_grad(gradlist[i])/norm_grad(initial_weight)) + pm*(qm+1)*(loss_list[i]**qm)*L
    hessian= coefm1*grad_val+coefm2*val
    return hessian

def norm_grad(gradm):
    total_grad= []
    for i in range(len(gradm)):
        client_grads = gradm[i].reshape(-1).tolist()
        total_grad+=client_grads
    total_grad= np.array(total_grad)
    return np.sum(np.square(total_grad))



def group_hessian(gradlist, loss_list, group_num, q,qm,lr):
    gm=1/group_num
    nm=len(gradlist)
    pm=1/nm
    L=1/lr

    bla= (gm/(qm+1))*((q-qm)/(qm+1))
    sum_val=0
    for i in range(len(loss_list)):
        sum_val+=pm*(loss_list[i]**(qm+1))
    sla= sum_val**((q-2*qm-1)/(qm+1))
    coefm1= sla*bla

    gradm=gradlist[0]*0
    for i in range(len(loss_list)):
        gradm= gradm+(pm*(qm+1)*loss_list[i]**qm)*gradlist[i]
    grad_val= norm_grad(gradm)

    bla2= (gm/(qm+1))
    sum_val=0
    for i in range(len(loss_list)):
        sum_val+=pm*(loss_list[i]**(qm+1))
    sla2= sum_val**((q-qm)/(qm+1))
    coefm2= sla2*bla2

    val=0
    for i in range(len(loss_list)):
        val+= pm*(qm+1)*qm*(loss_list[i]**(qm-1))* norm_grad(gradlist[i])+pm*(qm+1) + pm*(qm+1)*(loss_list[i]**qm)*L
    hessian= coefm1*grad_val+coefm2*val
    return hessian

def fed_avg(grad_list):
    gradm= grad_list[0]*0
    for i in range(len(grad_list)):
        gradm= gradm + grad_list[i]
    return gradm/len(grad_list)


def get_submodel_real(large_model,smaller_model):
    m1= large_model
    m2= smaller_model

    for i in range(len(m2)):
        if len(m1[i].shape)==2:
            x1,y1 = m1[i].shape
            x2,y2= m2[i].shape
            if x1==x2 and y1!=y2:
                aa= np.transpose(np.array(m1[i]))
                bb= aa[:y2]

                bb= np.transpose(bb)
                m2[i]=bb
            else:
                bb = m1[i][:x2]
                m2[i]=bb
        else:
            x2= m2[i].shape
            bb= m1[i][:x2[0]]
            m2[i]=bb
    return m2


def get_masked_model(smaller_model, bigger_model):
    m1 = bigger_model
    m2 = smaller_model
    m1 = np.array(m1) * 0
    for i in range(len(m2)):
        if len(m1[i].shape) == 2:
            x1, y1 = m1[i].shape
            x2, y2 = m2[i].shape
            if x1 == x2 and y1 != y2:
                aa = np.transpose(np.array(m1[i]))
                bb = np.transpose(np.array(m2[i]))
                aa[:y2] = bb
                aa = np.transpose(aa)
                m1[i] = aa
            else:
                m1[i][:x2] = m2[i]
        else:
            x2 = m2[i].shape
            m1[i][:x2[0]] = m2[i]
    return m1

def pad_list(small_list, big_list):
    if small_list.shape != big_list.shape:
        pad_widths = [(0, big_list.shape[i] - small_list.shape[i]) for i in range(len(small_list.shape))]
        padded_list = np.pad(small_list, pad_widths, mode='constant')
    else:
        padded_list = small_list
    return padded_list

def group_gradient_chatgpt(gradlist, loss_list, group_num, q,qm):
    gm=1/group_num
    nm=len(gradlist)
    pm=1/nm

    bla=(gm/(qm+1))
    sum_val = np.sum(np.power(loss_list, qm+1)) * pm
    sla= np.power(sum_val, (q-qm)/(qm+1))
    coefm= sla*bla

    gradm = np.zeros_like(gradlist[0])
    for i in range(len(gradlist)):
        gradm += (pm*(qm+1)*np.power(loss_list[i],qm))*gradlist[i]
    gradm = coefm*gradm
    return gradm

def set_model_weights(model, value):
    weights=[]
    for i in range(len(model)):
        weights.append( np.ones_like(model[i])*value)
    return weights


def get_masked_model_chatgpt(small_model, large_model):
    new_model = []
    for i in range(len(small_model)):
        new_model.append(pad_list(small_model[i], large_model[i]))

    return new_model

def group_hessian_new_chatgpt(initial_weight, gradlist, loss_list, group_num, q,qm ):
    gm=1/group_num
    nm=len(gradlist)
    pm=1/nm
    lr = 0.01
    L=1/lr

    sum_val = np.sum(pm*(np.power(loss_list,qm+1)))
    sla = np.power(sum_val, (q-2*qm-1)/(qm+1))
    coefm1 = (gm/(qm+1))*((q-qm)/(qm+1))*sla

    gradm = np.sum(pm*(qm+1)*np.power(np.array(loss_list).reshape(-1,1),qm)*gradlist, axis=0)
    grad_val = np.linalg.norm(gradm)/np.linalg.norm(initial_weight)

    sum_val = np.sum(pm*(np.power(loss_list,qm+1)))
    sla2 = np.power(sum_val, (q-qm)/(qm+1))
    coefm2 = (gm/(qm+1))*sla2
    val = np.sum(pm*(qm+1)*qm*np.power(loss_list,qm-1)*(np.linalg.norm(gradlist, axis=1)/np.linalg.norm(initial_weight)) + pm*(qm+1)*np.power(loss_list,qm)*L)
    hessian = coefm1*grad_val + coefm2*val

    return hessian

def crop_weight(bigger_weight, smaller_weight):
    """
    Crops the bigger weight to the shape of the smaller weight.
    """
    # Get the shape of the smaller weight
    bigger_weight= np.array(bigger_weight)
    smaller_weight = np.array(smaller_weight)
    smaller_shape = smaller_weight.shape
    # Get the slice indices for each dimension
    slice_indices = [slice(0, sh) for sh in smaller_shape]
    # Crop the bigger weight to the shape of the smaller weight
    cropped_weight = bigger_weight[tuple(slice_indices)]
    return cropped_weight


def get_cropped_model_chatgpt(large_model, small_model):
    new_model = []
    for i in range(len(small_model)):
        new_model.append(crop_weight(large_model[i], small_model[i]))

    return new_model