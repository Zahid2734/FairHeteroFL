from utils.functions_new import *
import tensorflow as tf
import numpy as np
import sys
import random
import os
import tensorflow_federated as tff
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(37)
random.seed(5)
tf.random.set_seed(8)


def train_shakespeare(q, q1, q2, q3, q4, q5):
    print(q,q1,q2,q3,q4,q5)
    # Load the Shakespeare dataset
    train_data, test_data = tff.simulation.datasets.shakespeare.load_data()

    # A fixed vocabularly of ASCII chars that occur in the works of Shakespeare and Dickens:
    vocab = list('dhlptx@DHLPTX $(,048cgkoswCGKOSW[_#\'/37;?bfjnrvzBFJNRVZ"&*.26:\naeimquyAEIMQUY]!%)-159\r')

    # Creating a mapping from unique characters to indices
    char2idx = {u: i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)

    # Input pre-processing parameters
    SEQ_LENGTH = 80
    BATCH_SIZE = 8
    BUFFER_SIZE = 100  # For dataset shuffling

    # Construct a lookup table to map string chars to indexes,
    # using the vocab loaded above:
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=vocab, values=tf.constant(list(range(len(vocab))),
                                           dtype=tf.int64)),
        default_value=0)

    def to_ids(x):
        s = tf.reshape(x['snippets'], shape=[1])
        chars = tf.strings.bytes_split(s).values
        ids = table.lookup(chars)
        return ids

    def split_input_target(chunk):
        input_text = tf.map_fn(lambda x: x[:-1], chunk)
        target_text = tf.map_fn(lambda x: x[1:], chunk)
        return (input_text, target_text)

    def preprocess(dataset):
        return (
            # Map ASCII chars to int64 indexes using the vocab
            dataset.map(to_ids)
            # Split into individual chars
            .unbatch()
            # Form example sequences of SEQ_LENGTH +1
            .batch(SEQ_LENGTH + 1, drop_remainder=True)
            # Shuffle and form minibatches
            .shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
            # And finally split into (input, target) tuples,
            # each of length SEQ_LENGTH.
            .map(split_input_target))

    class FlattenedCategoricalAccuracy(tf.keras.metrics.SparseCategoricalAccuracy):

        def __init__(self, name='accuracy', dtype=tf.float32):
            super().__init__(name, dtype=dtype)

        def update_state(self, y_true, y_pred, sample_weight=None):
            y_true = tf.reshape(y_true, [-1, 1])
            y_pred = tf.reshape(y_pred, [-1, len(vocab), 1])
            return super().update_state(y_true, y_pred, sample_weight)

    clients = get_clients()

    def data(client, source):
        return preprocess(source.create_tf_dataset_for_client(client))

    train_datasets = [data(client, source=train_data) for client in clients]
    # val_datasets= [data(client, source= test_data) for client in clients]

    # We concatenate the test datasets for evaluation with Keras by creating a
    # Dataset of Datasets, and then identity flat mapping across all the examples.
    test_dataset = tf.data.Dataset.from_tensor_slices(
        [data(client, test_data) for client in clients]).flat_map(lambda x: x)
    # Define the learning rate
    lr = 0.0001

    epochs = 301
    q = q
    group_num = 5

    qm1 = q1
    qm2 = q2
    qm3 = q3
    qm4 = q4
    qm5 = q5

    L = 1 / lr
    bla = SimpleMLP5
    model = bla.build(1)
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[FlattenedCategoricalAccuracy()],
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=lr),
        run_eagerly=True,
        experimental_run_tf_function=False
    )
    global_weight = model.get_weights()
    initial_weight = model.get_weights()

    model1 = bla.build(1)
    model1.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[FlattenedCategoricalAccuracy()],
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=lr),
        run_eagerly=True,
        experimental_run_tf_function=False
    )
    model1_weights = model1.get_weights()

    model2 = bla.build(.75)
    model2.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[FlattenedCategoricalAccuracy()],
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=lr),
        experimental_run_tf_function=False
    )
    model2_weights = model2.get_weights()

    model3 = bla.build(.5)
    model3.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[FlattenedCategoricalAccuracy()],
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=lr),
        run_eagerly=True,
        experimental_run_tf_function=False
    )
    model3_weights = model3.get_weights()

    model4 = bla.build(.35)
    model4.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[FlattenedCategoricalAccuracy()],
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=lr),
        run_eagerly=True,
        experimental_run_tf_function=False
    )
    model4_weights = model4.get_weights()

    model5 = bla.build(.25)
    model5.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[FlattenedCategoricalAccuracy()],
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=lr),
        run_eagerly=True,
        experimental_run_tf_function=False
    )
    model5_weights = model5.get_weights()

    group1_accuracy = []
    group1_loss = []
    group1_hessian = []
    group2_accuracy = []
    group2_loss = []
    group2_hessian = []
    group3_accuracy = []
    group3_loss = []
    group3_hessian = []
    group4_accuracy = []
    group4_loss = []
    group4_hessian = []
    group5_accuracy = []
    group5_loss = []
    group5_hessian = []
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

    for i in range(epochs):
        print("group 1 training")
        model1_accuracy = []
        model1_loss = []
        model1_train_accuracy = []
        model1_train_loss = []
        model1_grad = []
        if i % 10 == 0 and i > 0:
            model1.set_weights(global_weight)
            for a in range(30):
                score1 = model1.evaluate(train_datasets[a], verbose=1)
                model1_accuracy.append(score1[1])
                model1_loss.append(score1[0])
        randomlist1 = random.sample(range(0, 30), 10)
        for a in range(len(randomlist1)):
            model1.set_weights(global_weight)
            hist1 = model1.fit(train_datasets[randomlist1[a]], epochs=1, verbose=1)
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
        group1_hessian.append(group_1_hess_new)
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
        if i % 10 == 0 and i > 0:
            model2.set_weights(group2_weight)
            for a in range(30, 60):
                score2 = model2.evaluate(train_datasets[a], verbose=1)
                model2_accuracy.append(score2[1])
                model2_loss.append(score2[0])
        randomlist2 = random.sample(range(30, 60), 10)

        for a in range(len(randomlist2)):
            model2.set_weights(group2_weight)
            hist2 = model2.fit(train_datasets[randomlist2[a]], epochs=1, verbose=1)
            grad2 = np.array(group2_weight) - np.array(model2.get_weights())
            # model2_accuracy.append(hist2.history['val_accuracy'][-1])
            # model2_loss.append(hist2.history['val_loss'][-1])
            model2_train_accuracy.append(hist2.history['accuracy'][-1])
            model2_train_loss.append(hist2.history['loss'][-1])
            model2_grad.append(grad2 * L)
        group2_accuracy.append(model2_accuracy)
        group2_loss.append(model2_loss)
        group2_train_accuracy.append(model2_train_accuracy)
        group2_train_loss.append(model2_train_loss)

        group_2_grad = group_gradient(model2_grad, model2_train_loss, group_num, q, qm2)
        group_2_hess_new = group_hessian_new(global_weight, model2_grad, model2_train_loss, group_num, q, qm2, lr)
        group2_hessian.append(group_2_hess_new)
        coef2 = set_model_weights(group_2_grad, group_2_hess_new)
        coef2 = get_masked_model_chatgpt(coef2, global_weight)
        group_2_grad = get_masked_model_chatgpt(group_2_grad, global_weight)

        print("group 3 training")
        model3_accuracy = []
        model3_loss = []
        model3_grad = []
        model3_train_accuracy = []
        model3_train_loss = []
        # sub_weight = global_weight
        # sub_weight = get_submodel_new(sub_weight, 12)
        group3_weight = get_cropped_model_chatgpt(global_weight, model3_weights)
        if i % 10 == 0 and i > 0:
            model3.set_weights(group3_weight)
            for a in range(60, 90):
                score3 = model3.evaluate(train_datasets[a], verbose=1)
                model3_accuracy.append(score3[1])
                model3_loss.append(score3[0])
        randomlist3 = random.sample(range(60, 90), 10)
        for a in range(len(randomlist3)):
            model3.set_weights(group3_weight)
            hist3 = model3.fit(train_datasets[randomlist3[a]], epochs=1, verbose=1)
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
        group3_hessian.append(group_3_hess_new)
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
        if i % 10 == 0 and i > 0:
            model4.set_weights(group4_weight)
            for a in range(90, 120):
                score4 = model4.evaluate(train_datasets[a], verbose=1)
                model4_accuracy.append(score4[1])
                model4_loss.append(score4[0])
        randomlist4 = random.sample(range(90, 120), 10)
        for a in range(len(randomlist4)):
            model4.set_weights(group4_weight)
            hist4 = model4.fit(train_datasets[randomlist4[a]], epochs=1, verbose=1)
            grad4 = np.array(group4_weight) - np.array(model4.get_weights())
            # model4_accuracy.append(hist4.history['val_accuracy'][-1])
            # model4_loss.append(hist4.history['val_loss'][-1])
            model4_train_accuracy.append(hist4.history['accuracy'][-1])
            model4_train_loss.append(hist4.history['loss'][-1])
            model4_grad.append(grad4 * L)
        group4_accuracy.append(model4_accuracy)
        group4_loss.append(model4_loss)
        group4_train_accuracy.append(model4_train_accuracy)
        group4_train_loss.append(model4_train_loss)

        group_4_grad = group_gradient(model4_grad, model4_train_loss, group_num, q, qm4)
        group_4_hess_new = group_hessian_new(global_weight, model4_grad, model4_train_loss, group_num, q, qm4, lr)
        group4_hessian.append(group_4_hess_new)
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
        if i % 10 == 0 and i > 0:
            model5.set_weights(group5_weight)
            for a in range(120, 150):
                score5 = model5.evaluate(train_datasets[a], verbose=1)
                model5_accuracy.append(score5[1])
                model5_loss.append(score5[0])
        randomlist5 = random.sample(range(120, 150), 10)
        for a in range(len(randomlist5)):
            model5.set_weights(group5_weight)
            hist5 = model5.fit(train_datasets[randomlist5[a]], epochs=1, verbose=1)
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
        group5_hessian.append(group_5_hess_new)
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
        score = model.evaluate(test_dataset, verbose=0)
        print('communication round:', i)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])
        global_accuracy.append(score[1])
        global_loss.append(score[0])

        if i % 5 == 0 and i > 0:
            sample_list = [global_accuracy, global_loss, group1_accuracy, group1_loss, group2_accuracy, group2_loss,
                           group3_accuracy, group3_loss, group4_accuracy, group4_loss, group5_accuracy, group5_loss,
                           group1_train_accuracy, group1_train_loss, group2_train_accuracy, group2_train_loss,
                           group3_train_accuracy, group3_train_loss, group4_train_accuracy, group4_train_loss,
                           group5_train_accuracy, group5_train_loss, global_weight]
            save_file_name = f'../data/Fair_shakespeare_{q}_{qm1}_{qm2}_{qm3}_{qm4}_{qm5}_{epochs}_{lr}.pkl'
            save_file(save_file_name, sample_list)





if __name__ == "__main__":
 # Access the values of q, q1, q2, q3, q4, q5
    q = sys.argv[1]
    q1 = sys.argv[2]
    q2 = sys.argv[3]
    q3 = sys.argv[4]
    q4 = sys.argv[5]
    q5 = sys.argv[6]
    train_shakespeare(float(q), float(q1), float(q2), float(q3), float(q4), float(q5))
