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
HIDDEN_SIZE = 512
SENTENCE_LIMIT_SIZE = 70
EMBEDDING_SIZE = 100
source_vocab_size = 125
encoder_embedding_size = 100
filters_size = [3, 5]
num_filters = 50
BATCH_SIZE = 256
EPOCHES = 50
LEARNING_RATE = 0.001
L2_LAMBDA = 10
KEEP_PROB = 0.8

with tf.name_scope("cnn"):

    with tf.name_scope("placeholders"):
        inputs = tf.placeholder(dtype=tf.int32, shape=(None, SENTENCE_LIMIT_SIZE), name="inputs")
        targets = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="targets")

    # embeddings
    with tf.name_scope("embeddings"):

        # embedding_matrix = tf.Variable(initial_value=static_embeddings, trainable=False, name="embedding_matrix")
        # embed = tf.nn.embedding_lookup(embedding_matrix, inputs, name="embed")
        encoder_embed = tf.contrib.layers.embed_sequence(inputs, source_vocab_size, encoder_embedding_size)

        embed_expanded = tf.expand_dims(encoder_embed, -1, name="embed_expand")

    # max-pooling results
    pooled_outputs = []

    # iterate multiple filter
    for i, filter_size in enumerate(filters_size):
        with tf.name_scope("conv_maxpool_%s" % filter_size):
            filter_shape = [filter_size, EMBEDDING_SIZE, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, mean=0.0, stddev=0.1), name="W")
            b = tf.Variable(tf.zeros(num_filters), name="b")

            conv = tf.nn.conv2d(input=embed_expanded, filter=W, strides=[1, 1, 1, 1], padding="VALID", name="conv")

            # activation
            a = tf.nn.relu(tf.nn.bias_add(conv, b), name="activations")
            # pooling
            max_pooling = tf.nn.max_pool(
                value=a,
                ksize=[1, SENTENCE_LIMIT_SIZE - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="max_pooling",
            )
            pooled_outputs.append(max_pooling)

    # filter information
    total_filters = num_filters * len(filters_size)
    total_pool = tf.concat(pooled_outputs, 3)
    flattend_pool = tf.reshape(total_pool, (-1, total_filters))

    # dropout
    # with tf.name_scope("dropout"):
    # dropout = tf.nn.dropout(flattend_pool, KEEP_PROB)

    # output
    with tf.name_scope("output"):
        W = tf.get_variable("W", shape=(total_filters, 1), initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.zeros(1), name="b")

        logits = tf.add(tf.matmul(flattend_pool, W), b)
        predictions = tf.nn.sigmoid(logits, name="predictions")

    # loss
    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits))
        loss = loss + L2_LAMBDA * tf.nn.l2_loss(W)
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    # evaluation
    with tf.name_scope("evaluation"):
        correct_preds = tf.equal(tf.cast(tf.greater(predictions, 0.5), tf.float32), targets)
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
    saver.restore(sess, "./models/cnn_final")
    # writer = tf.summary.FileWriter("./graphs/cnn_final", tf.get_default_graph())
    n_batches = int(X.shape[0] / BATCH_SIZE)
    print("n_batches: ", n_batches)
    total_ind = 0
    end_flag = 0
    test_sum = 0
    t_batches = int(X_test.shape[0] / BATCH_SIZE)
    for x_batch, y_batch in get_batch(X_test, Y_test):
        answer = sess.run(predictions, feed_dict={inputs: x_batch, targets: y_batch})
        for index in range(len(answer)):
            test_sum += (abs(answer[index] * 64 - y_batch[index] * 64)) / (y_batch[index] * 64)
    print("Test loss: {}".format(test_sum / (256 * (t_batches - 1))))
    answer = sess.run(predictions, feed_dict={inputs: X_test[-1:], targets: Y_test[-1:]})
    # print(answer, Y_test[-1])
    # lstm_test_accuracy.append(test_sum/(256*(t_batches-1)))
    real_sum = 0
    r_batches = int(X.shape[0] / BATCH_SIZE)
    for x_batch, y_batch in get_batch(X, Y):
        answer = sess.run(predictions, feed_dict={inputs: x_batch, targets: y_batch})
        for index in range(len(answer)):
            real_sum += (abs(answer[index] * 64 - y_batch[index] * 64)) / (y_batch[index] * 64)
    print("Train loss: {}".format(real_sum / (256 * (r_batches - 1))))
    # lstm_real_accuracy.append(real_sum/(256*(r_batches-1)))

    answer = sess.run(predictions, feed_dict={inputs: X_actual, targets: Y_actual})
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
