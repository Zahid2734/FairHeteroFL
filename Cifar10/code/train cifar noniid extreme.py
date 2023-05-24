from utils.functions_new import *
import tensorflow as tf
import numpy as np
import sys
import random
import os
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(37)
random.seed(5)
tf.random.set_seed(8)


def train_cifar_noniid_extreme(q, q1, q2, q3, q4, q5):
    print(q,q1,q2,q3,q4,q5)
    file_name = "../non_iid_cifar_extreme.pkl"
    clients = open_file(file_name)

    client_data = []
    client_labels = []
    for client in clients:
        data = []
        labels = []
        for point in client:
            data.append(point[0])
            labels.append(point[1])
        client_data.append(data)
        client_labels.append(labels)

    # process and batch the training data for each client

    clients_batched = dict()
    clients_batched_test = dict()
    for i in range(100):
        clients_batched[i], clients_batched_test[i] = batch_data_non_iid_new(client_data[i], client_labels[i], bs=32)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_test = x_test.astype('float32')
    x_test /= 255
    nb_classes = 10
    y_test = tf.keras.utils.to_categorical(y_test, nb_classes)

    test_batched = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(len(y_test))
    client_names = list(clients_batched.keys())

    q = q
    group_num = 5

    qm1 = q1
    qm2 = q2
    qm3 = q3
    qm4 = q4
    qm5 = q5


    print(q, qm1, qm2, qm3, qm4, qm5)
    loss = 'categorical_crossentropy'
    metrics = ['accuracy']

    epochs = 2001
    lr = .00005
    L = 1 / lr
    bla = SimpleMLP4
    model = bla.build(1)
    model.compile(loss=loss,
                  optimizer=tf.keras.optimizers.Adam(
                      learning_rate=lr),
                  metrics=metrics)
    global_weight = model.get_weights()
    initial_weight = model.get_weights()

    model1 = bla.build(1)
    model1.compile(loss=loss,
                   optimizer=tf.keras.optimizers.Adam(
                       learning_rate=lr),

                   metrics=metrics)
    model1_weights = model1.get_weights()

    model2 = bla.build(.75)
    model2.compile(loss=loss,
                   optimizer=tf.keras.optimizers.Adam(
                       learning_rate=lr),

                   metrics=metrics)
    model2_weights = model2.get_weights()

    model3 = bla.build(.5)
    model3.compile(loss=loss,
                   optimizer=tf.keras.optimizers.Adam(
                       learning_rate=lr),

                   metrics=metrics)
    model3_weights = model3.get_weights()

    model4 = bla.build(.35)
    model4.compile(loss=loss,
                   optimizer=tf.keras.optimizers.Adam(
                       learning_rate=lr),

                   metrics=metrics)
    model4_weights = model4.get_weights()

    model5 = bla.build(.25)
    model5.compile(loss=loss,
                   optimizer=tf.keras.optimizers.Adam(
                       learning_rate=lr),
                   metrics=metrics)
    model5_weights = model5.get_weights()

    group1_accuracy = []
    group1_loss = []

    group2_accuracy = []
    group2_loss = []

    group3_accuracy = []
    group3_loss = []

    group4_accuracy = []
    group4_loss = []

    group5_accuracy = []
    group5_loss = []

    group1_train_accuracy = []
    group1_train_loss = []
    group2_train_accuracy = []
    group2_train_loss = []
    group3_train_accuracy = []
    group3_train_loss = []
    group4_train_accuracy = []
    group4_train_loss = []
    group5_train_accuracy = []
    group5_train_loss = []
    global_accuracy = []
    global_loss = []
    global_weight_list = []
    for i in range(epochs):
        print("group 1 training")
        model1_accuracy = []
        model1_loss = []
        model1_train_accuracy = []
        model1_train_loss = []
        model1_grad = []
        for a in range(20):
            model1.set_weights(global_weight)
            if i % 10 == 0 and i > 0:
                score1 = model1.evaluate(clients_batched_test[client_names[a]], verbose=1)
                model1_accuracy.append(score1[1])
                model1_loss.append(score1[0])
            hist1 = model1.fit(clients_batched[client_names[a]], epochs=1, verbose=1)
            grad1 = np.array(global_weight) - np.array(model1.get_weights())
            model1_train_accuracy.append(hist1.history['accuracy'][-1])
            model1_train_loss.append(hist1.history['loss'][-1])
            model1_grad.append(grad1 * L)
        group1_accuracy.append(model1_accuracy)
        group1_loss.append(model1_loss)
        group1_train_accuracy.append(model1_train_accuracy)
        group1_train_loss.append(model1_train_loss)

        group_1_grad = group_gradient(model1_grad, model1_train_loss, group_num, q, qm1)
        group_1_hess_new = group_hessian_new(global_weight, model1_grad, model1_train_loss, group_num, q, qm1, lr)
        coef1 = set_model_weights(group_1_grad, group_1_hess_new)
        coef1 = get_masked_model_chatgpt(coef1, global_weight)
        group_1_grad = get_masked_model_chatgpt(group_1_grad, global_weight)

        print("group 2 training")
        model2_accuracy = []
        model2_loss = []
        model2_grad = []
        model2_train_accuracy = []
        model2_train_loss = []
        group2_weight = get_cropped_model_chatgpt(global_weight, model2_weights)
        for a in range(20, 40):
            model2.set_weights(group2_weight)
            if i % 10 == 0 and i > 0:
                score2 = model2.evaluate(clients_batched_test[client_names[a]], verbose=1)
                model2_accuracy.append(score2[1])
                model2_loss.append(score2[0])
            hist2 = model2.fit(clients_batched[client_names[a]], epochs=1, verbose=1)
            grad2 = np.array(group2_weight) - np.array(model2.get_weights())
            model2_train_accuracy.append(hist2.history['accuracy'][-1])
            model2_train_loss.append(hist2.history['loss'][-1])
            model2_grad.append(grad2 * L)
        group2_accuracy.append(model2_accuracy)
        group2_loss.append(model2_loss)
        group2_train_accuracy.append(model2_train_accuracy)
        group2_train_loss.append(model2_train_loss)

        group_2_grad = group_gradient(model2_grad, model2_train_loss, group_num, q, qm2)
        group_2_hess_new = group_hessian_new(global_weight, model2_grad, model2_train_loss, group_num, q, qm2, lr)
        coef2 = set_model_weights(group_2_grad, group_2_hess_new)
        coef2 = get_masked_model_chatgpt(coef2, global_weight)
        group_2_grad = get_masked_model_chatgpt(group_2_grad, global_weight)

        print("group 3 training")
        model3_accuracy = []
        model3_loss = []
        model3_grad = []
        model3_train_accuracy = []
        model3_train_loss = []
        group3_weight = get_cropped_model_chatgpt(global_weight, model3_weights)
        for a in range(40, 60):
            model3.set_weights(group3_weight)
            if i % 10 == 0 and i > 0:
                score3 = model3.evaluate(clients_batched_test[client_names[a]], verbose=1)
                model3_accuracy.append(score3[1])
                model3_loss.append(score3[0])
            hist3 = model3.fit(clients_batched[client_names[a]], epochs=1, verbose=1)
            grad3 = np.array(group3_weight) - np.array(model3.get_weights())
            model3_train_accuracy.append(hist3.history['accuracy'][-1])
            model3_train_loss.append(hist3.history['loss'][-1])
            model3_grad.append(grad3 * L)
        group3_accuracy.append(model3_accuracy)
        group3_loss.append(model3_loss)
        group3_train_accuracy.append(model3_train_accuracy)
        group3_train_loss.append(model3_train_loss)

        group_3_grad = group_gradient(model3_grad, model3_train_loss, group_num, q, qm3)
        group_3_hess_new = group_hessian_new(global_weight, model3_grad, model3_train_loss, group_num, q, qm3, lr)
        coef3 = set_model_weights(group_3_grad, group_3_hess_new)
        coef3 = get_masked_model_chatgpt(coef3, global_weight)
        group_3_grad = get_masked_model_chatgpt(group_3_grad, global_weight)

        print("group 4 training")
        model4_accuracy = []
        model4_loss = []
        model4_grad = []
        model4_train_accuracy = []
        model4_train_loss = []
        group4_weight = get_cropped_model_chatgpt(global_weight, model4_weights)
        for a in range(60, 80):
            model4.set_weights(group4_weight)
            if i % 10 == 0 and i > 0:
                score4 = model4.evaluate(clients_batched_test[client_names[a]], verbose=1)
                model4_accuracy.append(score4[1])
                model4_loss.append(score4[0])
            hist4 = model4.fit(clients_batched[client_names[a]], epochs=1, verbose=1)
            grad4 = np.array(group4_weight) - np.array(model4.get_weights())
            model4_train_accuracy.append(hist4.history['accuracy'][-1])
            model4_train_loss.append(hist4.history['loss'][-1])
            model4_grad.append(grad4 * L)
        group4_accuracy.append(model4_accuracy)
        group4_loss.append(model4_loss)
        group4_train_accuracy.append(model4_train_accuracy)
        group4_train_loss.append(model4_train_loss)

        group_4_grad = group_gradient(model4_grad, model4_train_loss, group_num, q, qm4)
        group_4_hess_new = group_hessian_new(global_weight, model4_grad, model4_train_loss, group_num, q, qm4, lr)
        coef4 = set_model_weights(group_4_grad, group_4_hess_new)
        coef4 = get_masked_model_chatgpt(coef4, global_weight)
        group_4_grad = get_masked_model_chatgpt(group_4_grad, global_weight)

        print("group 5 training")
        model5_accuracy = []
        model5_loss = []
        model5_grad = []
        model5_train_accuracy = []
        model5_train_loss = []
        group5_weight = get_cropped_model_chatgpt(global_weight, model5_weights)
        for a in range(80, 100):
            model5.set_weights(group5_weight)
            if i % 10 == 0 and i > 0:
                score5 = model5.evaluate(clients_batched_test[client_names[a]], verbose=1)
                model5_accuracy.append(score5[1])
                model5_loss.append(score5[0])
            hist5 = model5.fit(clients_batched[client_names[a]], epochs=1, verbose=1)
            grad5 = np.array(group5_weight) - np.array(model5.get_weights())
            model5_train_accuracy.append(hist5.history['accuracy'][-1])
            model5_train_loss.append(hist5.history['loss'][-1])
            model5_grad.append(grad5 * L)
        group5_accuracy.append(model5_accuracy)
        group5_loss.append(model5_loss)
        group5_train_accuracy.append(model5_train_accuracy)
        group5_train_loss.append(model5_train_loss)

        group_5_grad = group_gradient(model5_grad, model5_train_loss, group_num, q, qm5)
        group_5_hess_new = group_hessian_new(global_weight, model5_grad, model5_train_loss, group_num, q, qm5, lr)
        coef5 = set_model_weights(group_5_grad, group_5_hess_new)
        coef5 = get_masked_model_chatgpt(coef5, global_weight)
        group_5_grad = get_masked_model_chatgpt(group_5_grad, global_weight)

        sum_grad = (np.array(group_1_grad) + np.array(group_2_grad) + np.array(group_3_grad) + np.array(
            group_4_grad) + np.array(group_5_grad))

        sum_coef = np.array(coef1) + np.array(coef2) + np.array(coef3) + np.array(coef4) + np.array(coef5)

        grad = np.divide(sum_grad, sum_coef)

        global_weight = np.array(global_weight) - grad
        global_weight.tolist()
        model.set_weights(global_weight)
        model.evaluate(x_test, y_test)
        score = model.evaluate(x_test, y_test, verbose=0)
        print('communication round:', i)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])
        global_accuracy.append(score[1])
        global_loss.append(score[0])

        if i % 10 == 0 and i > 0:
            global_weight_list.append(global_weight)
            sample_list = [global_accuracy, global_loss, group1_accuracy, group1_loss, group2_accuracy, group2_loss,
                           group3_accuracy, group3_loss, group4_accuracy, group4_loss, group5_accuracy, group5_loss,
                           group1_train_accuracy, group1_train_loss, group2_train_accuracy, group2_train_loss,
                           group3_train_accuracy, group3_train_loss, group4_train_accuracy, group4_train_loss,
                           group5_train_accuracy, group5_train_loss,
                           global_weight_list]
            save_file_name = f'../data/Fair_Cifar10_noniid_extreme20_zone_{q}_{qm1}_{qm2}_{qm3}_{qm4}_{qm5}_{epochs}_{lr}.pkl'
            save_file(save_file_name, sample_list)


if __name__ == "__main__":
 # Access the values of q, q1, q2, q3, q4, q5
    q = sys.argv[1]
    q1 = sys.argv[2]
    q2 = sys.argv[3]
    q3 = sys.argv[4]
    q4 = sys.argv[5]
    q5 = sys.argv[6]
    train_cifar_noniid_extreme(float(q), float(q1), float(q2), float(q3), float(q4), float(q5))
