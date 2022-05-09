import load_trace
import a3c
import env
import tensorflow as tf
import os
import numpy as np

from primo.distill import DistillEngine


os.environ["CUDA_VISIBLE_DEVICES"] = ""

# bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_INFO = 6
S_LEN = 8  # take how many frames in the past
A_DIM = 6
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
SUMMARY_DIR = "./results"
TEST_TRACES = "./cooked_traces/"
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward
NN_MODEL = "./baselines/models/pretrain_linear_reward.ckpt"


def main():

    np.random.seed(RANDOM_SEED)
    assert len(VIDEO_BIT_RATE) == A_DIM

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(TEST_TRACES)

    net_env = env.Environment(all_cooked_time=all_cooked_time, all_cooked_bw=all_cooked_bw)

    with tf.Session() as sess:

        actor = a3c.ActorNetwork(sess, state_dim=[S_INFO, S_LEN], action_dim=A_DIM, learning_rate=ACTOR_LR_RATE)

        actor_parameter = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        print(actor_parameter)

        critic = a3c.CriticNetwork(sess, state_dim=[S_INFO, S_LEN], learning_rate=CRITIC_LR_RATE)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()  # save neural net parameters

        # restore neural net parameters
        if NN_MODEL is not None:  # NN_MODEL is the path to file
            saver.restore(sess, NN_MODEL)
            print("Testing model restored.")

        def predict_wrapper(state):
            action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
            action_cumsum = np.cumsum(action_prob)
            bit_rate = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
            return bit_rate

        def rl_test_wrapper(rl_predict):
            time_stamp = 0

            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY

            action_vec = np.zeros(A_DIM)
            action_vec[bit_rate] = 1

            s_batch = [np.zeros((S_INFO, S_LEN))]
            a_batch = [action_vec]
            r_batch = []
            entropy_record = []

            video_count = 0

            while True:  # serve video forever
                # the action is from the last decision
                # this is to make the framework similar to the real
                (
                    delay,
                    sleep_time,
                    buffer_size,
                    rebuf,
                    video_chunk_size,
                    next_video_chunk_sizes,
                    end_of_video,
                    video_chunk_remain,
                ) = net_env.get_video_chunk(bit_rate)

                time_stamp += delay  # in ms
                time_stamp += sleep_time  # in ms

                # reward is video quality - rebuffer penalty - smoothness
                reward = (
                    VIDEO_BIT_RATE[bit_rate] / M_IN_K
                    - REBUF_PENALTY * rebuf
                    - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] - VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K
                )

                r_batch.append(reward)

                last_bit_rate = bit_rate

                # retrieve previous state
                if len(s_batch) == 0:
                    state = [np.zeros((S_INFO, S_LEN))]
                else:
                    state = np.array(s_batch[-1], copy=True)

                # dequeue history record
                state = np.roll(state, -1, axis=1)

                # this should be S_INFO number of terms
                state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
                state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
                state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
                state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
                state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
                state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

                bit_rate = rl_predict(state)

                s_batch.append(state)

                if end_of_video:

                    last_bit_rate = DEFAULT_QUALITY
                    bit_rate = DEFAULT_QUALITY  # use the default action here

                    del s_batch[:]
                    del a_batch[:]
                    del r_batch[:]

                    action_vec = np.zeros(A_DIM)
                    action_vec[bit_rate] = 1

                    s_batch.append(np.zeros((S_INFO, S_LEN)))
                    a_batch.append(action_vec)

                    video_count += 1

                    if video_count >= len(all_file_names):
                        break

                    # """metis seems not use all video"""
                    # break

                if not "trajectory" in locals():
                    trajectory = []
                trajectory.append((state, bit_rate, reward))
            return trajectory

        """
        Primo
        """
        # 1. Define feature name
        feature_names = ["last_qoe", "last_buffer"]
        for i in range(S_LEN):
            feature_names.append("throughput_" + str(i))
        for i in range(S_LEN):
            feature_names.append("download_" + str(i))
        for i in range(A_DIM):
            feature_names.append("next_size_" + str(i))
        feature_names.append("remain")

        def state_transformer(rl_state):
            decision_tree_state = []
            decision_tree_state.append(rl_state[0, -1])
            decision_tree_state.append(rl_state[1, -1])
            for i in range(S_LEN):
                decision_tree_state.append(rl_state[2, i])
            for i in range(S_LEN):
                decision_tree_state.append(rl_state[3, i])
            for i in range(A_DIM):
                decision_tree_state.append(rl_state[4, i])
            decision_tree_state.append(rl_state[5, -1])
            return decision_tree_state

        distiller = DistillEngine(
            environment=net_env,
            actor=actor,
            session=sess,
            feature_names=feature_names,
            prune_factor=0.001,
            # search_method="grid",
            rl_test_wrapper=rl_test_wrapper,
            predict_wrapper=predict_wrapper,
            state_transformer=state_transformer,
        )
        distiller.distill(epochs=10)

        # epochs: int = 100,
        # train_frac: float = 0.8,
        # max_samples: int = 1000,
        # n_batch_rollouts: int = 10,


if __name__ == "__main__":
    main()
