import warnings

warnings.filterwarnings("ignore")
import numpy as np
import tensorflow as tf
import pickle

with open("../training.pickle", "rb") as ftrain:
    dataset_train = pickle.load(ftrain)
    X, Y = dataset_train
with open("../testing.pickle", "rb") as ftest:
    dataset_test = pickle.load(ftest)
    X_test, Y_test = dataset_test
with open("../nf.pickle", "rb") as factual:
    dataset_actual = pickle.load(factual)
    X_actual, Y_actual = dataset_actual
with open("../source.pickle", "rb") as fsource:
    source_text_to_int = pickle.load(fsource)
with open("../target.pickle", "rb") as ftarget:
    target_text_to_int = pickle.load(ftarget)

# parameters
tf.reset_default_graph()
HIDDEN_SIZE_0 = 128
HIDDEN_SIZE = 32
LEARNING_RATE = 0.001
EPOCHES = 30
BATCH_SIZE = 256
SENTENCE_LIMIT_SIZE = 70
EMBEDDING_SIZE = 100
source_vocab_size = 125
encoder_embedding_size = 100

with tf.name_scope("dnn"):

    with tf.name_scope("placeholders"):
        inputs = tf.placeholder(dtype=tf.int32, shape=(None, SENTENCE_LIMIT_SIZE), name="inputs")
        targets = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="targets")

    # embeddings
    with tf.name_scope("embeddings"):

        # embedding_matrix = tf.Variable(initial_value=static_embeddings, trainable=False, name="embedding_matrix")
        # embed = tf.nn.embedding_lookup(embedding_matrix, inputs, name="embed")
        encoder_embed = tf.contrib.layers.embed_sequence(inputs, source_vocab_size, encoder_embedding_size)

        sum_embed = tf.reduce_sum(encoder_embed, axis=1, name="sum_embed")

    # model
    with tf.name_scope("model"):

        W0 = tf.Variable(tf.random_normal(shape=(EMBEDDING_SIZE, HIDDEN_SIZE_0), stddev=0.1), name="W0")
        b0 = tf.Variable(tf.zeros(shape=(HIDDEN_SIZE_0), name="b0"))

        W1 = tf.Variable(tf.random_normal(shape=(HIDDEN_SIZE_0, HIDDEN_SIZE), stddev=0.1), name="W1")
        b1 = tf.Variable(tf.zeros(shape=(HIDDEN_SIZE), name="b1"))

        W2 = tf.Variable(tf.random_normal(shape=(HIDDEN_SIZE, 1), stddev=0.1), name="W2")
        b2 = tf.Variable(tf.zeros(shape=(1), name="b2"))

        z0 = tf.add(tf.matmul(sum_embed, W0), b0)
        a0 = tf.nn.relu(z0)

        z1 = tf.add(tf.matmul(a0, W1), b1)
        a1 = tf.nn.relu(z1)

        logits = tf.add(tf.matmul(a1, W2), b2)
        outputs = tf.nn.sigmoid(logits, name="outputs")

    # loss
    with tf.name_scope("loss"):
        # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits))
        loss = tf.losses.mean_squared_error(targets, outputs)
    # optimizer
    with tf.name_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    # evaluation
    with tf.name_scope("evaluation"):
        correct_preds = tf.equal(tf.cast(tf.greater(outputs, 0.5), tf.float32), targets)
        accuracy = tf.reduce_sum(tf.reduce_sum(tf.cast(correct_preds, tf.float32), axis=1))


def get_batch(x, y, batch_size=BATCH_SIZE, shuffle=True):
    assert x.shape[0] == y.shape[0], print("error shape!")
    # shuffle
    if shuffle:
        shuffled_index = np.random.permutation(range(x.shape[0]))

        x = x[shuffled_index]
        y = y[shuffled_index]

    n_batches = int(x.shape[0] / batch_size)

    for i in range(n_batches - 1):
        x_batch = x[i * batch_size : (i + 1) * batch_size]
        y_batch = y[i * batch_size : (i + 1) * batch_size]

        yield x_batch, y_batch


saver = tf.train.Saver()

import time

with tf.Session() as sess:
    # sess.run(tf.global_variables_initializer())
    saver.restore(sess, "./models/dnn_final")
    # writer = tf.summary.FileWriter("./graphs/dnn_final", tf.get_default_graph())
    n_batches = int(X.shape[0] / BATCH_SIZE)
    print("n_batches: ", n_batches)
    total_ind = 0
    end_flag = 0
    test_sum = 0
    t_batches = int(X_test.shape[0] / BATCH_SIZE)
    for x_batch, y_batch in get_batch(X_test, Y_test):
        answer = sess.run(outputs, feed_dict={inputs: x_batch, targets: y_batch})
        for index in range(len(answer)):
            test_sum += (abs(answer[index] * 64 - y_batch[index] * 64)) / (y_batch[index] * 64)
    print("Test loss: {}".format(test_sum / (256 * (t_batches - 1))))
    answer = sess.run(outputs, feed_dict={inputs: X_test[-1:], targets: Y_test[-1:]})
    # print(answer, Y_test[-1])
    # lstm_test_accuracy.append(test_sum/(256*(t_batches-1)))
    real_sum = 0
    r_batches = int(X.shape[0] / BATCH_SIZE)
    for x_batch, y_batch in get_batch(X, Y):
        answer = sess.run(outputs, feed_dict={inputs: x_batch, targets: y_batch})
        for index in range(len(answer)):
            real_sum += (abs(answer[index] * 64 - y_batch[index] * 64)) / (y_batch[index] * 64)
    print("Train loss: {}".format(real_sum / (256 * (r_batches - 1))))
    # lstm_real_accuracy.append(real_sum/(256*(r_batches-1)))

    answer = sess.run(outputs, feed_dict={inputs: X_actual, targets: Y_actual})
    summation = 0
    jndex = 0
    pos = 0
    nfs = ["aggcounter", "anonipaddr", "forcetcp", "tcp_gen", "tcpack", "tcpresp", "timefilter", "udpipencap"]
    len_nfs = [15, 5, 17, 15, 2, 19, 12, 4]
    nn = a = b = c = 0
    temp_list = []
    for index in range(89):
        a += answer[index]
        b += Y_actual[index]
        c += abs(answer[index] - Y_actual[index])
        summation += abs(answer[index] - Y_actual[index]) / Y_actual[index]
        nn += abs(answer[index] - Y_actual[index]) / Y_actual[index]
        if len_nfs[pos] > 1:
            len_nfs[pos] -= 1
        else:
            temp_var = c / a
            temp_list.append(temp_var[0])
            pos += 1
            a = b = c = nn = 0
    print("Performance on real Click elements: ")
    for index, item in enumerate(temp_list):
        print("WMAPE of:", nfs[index], item)
    time_start = time.time()

    # writer.close()
