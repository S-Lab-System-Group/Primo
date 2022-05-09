import logging
import numpy as np

from primo.distill.decision_tree import DecisionTree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def get_dt_rollouts(rl_test_wrapper, student, n_batch_rollouts):
    rollouts = []
    for _ in range(n_batch_rollouts):
        rollouts.extend(rl_test_wrapper(student.dt_predict_wrapper))
    return rollouts


def _sample(obss, acts, dt_obss, max_samples):
    idx = np.random.choice(len(obss), size=max_samples)
    return obss[idx], acts[idx], dt_obss[idx]


def evaluate_student(rl_test_wrapper, student, n_batch_rollouts):
    eval_batch = 5 * n_batch_rollouts
    rollouts = get_dt_rollouts(rl_test_wrapper, student, eval_batch)

    return sum((rew for _, _, rew in rollouts)) / eval_batch


def identify_best_policy(rl_test_wrapper, policies, n_batch_rollouts):
    logger.info("=" * 68)
    logger.info(f"Start identify best policy from initial {len(policies)} policies.")
    # cut policies by half on each iteration
    while len(policies) > 1:
        # Step 1: Sort policies by current estimated reward
        policies = sorted(policies, key=lambda entry: -entry[1])

        # Step 2: Prune second half of policies
        n_policies = int((len(policies) + 1) / 2)
        logger.info(f"Current policy count: {n_policies}")

        # Step 3: build new policies
        new_policies = []
        for i in range(n_policies):
            policy, rew = policies[i]
            new_rew = evaluate_student(rl_test_wrapper, policy, n_batch_rollouts)
            new_policies.append((policy, new_rew))
            logger.info(f"Reward update: {rew:.4f} -> {new_rew:.4f}")

        policies = new_policies

    assert len(policies) == 1, "Unexpected candidate policy numbers."
    best_policy = policies[0][0]
    final_reward = evaluate_student(rl_test_wrapper, best_policy, n_batch_rollouts)
    logger.info(f"Best policy configuration = {best_policy.tree.get_params()}")
    logger.info(
        f"Best policy: final reward = {final_reward:.4f}, leaves number = {best_policy.tree.get_n_leaves()}, nodes number = {best_policy.tree.tree_.node_count}, depth = {best_policy.tree.get_depth()}"
    )

    return best_policy


def parse_trajectory(trajectories, state_transformer):
    obss, acts, dt_obss = [], [], []

    obss.extend((obs for obs, _, _ in trajectories))
    acts.extend((act for _, act, _ in trajectories))
    if state_transformer == None:
        dt_obss.extend((np.array(obs).reshape(1, -1) for obs in obss))
    else:
        dt_obss.extend((state_transformer(obs) for obs in obss))

    return obss, acts, dt_obss


def train_dagger(
    trajectories=None,
    state_transformer=None,
    rl_test_wrapper=None,
    predict_wrapper=None,
    max_depth=None,
    max_leaves=None,
    prune_factor=0.001,
    epochs=100,
    train_frac=0.8,
    max_samples=20000,
    n_batch_rollouts=10,
    seed=1,
):
    np.random.seed(seed)
    students = []

    obss, acts, dt_obss = parse_trajectory(trajectories, state_transformer)

    student = DecisionTree(
        state_transformer=state_transformer, max_depth=max_depth, max_leaf_nodes=max_leaves, prune_factor=prune_factor
    )

    logger.info(f"Training student with {max_samples} points")
    # Dagger outer loop
    for i in range(epochs):
        logger.info(f"Epoch {i+1}/{epochs}:")

        # Step a: Train from a random subset of aggregated data
        cur_obss, cur_acts, cur_dt_obss = _sample(np.array(obss), np.array(acts), np.array(dt_obss), max_samples)
        # logger.info(f"Training student with {len(cur_obss)} points")
        student.train(cur_dt_obss, cur_acts, train_frac)

        # Step b: Generate trace using student
        student_trace = get_dt_rollouts(rl_test_wrapper, student, n_batch_rollouts)
        student_obss = [obs for obs, _, _ in student_trace]
        student_dt_obss = [state_transformer(obs) for obs in student_obss]

        # Step c: Query the oracle for supervision
        # teacher_acts = teacher.predict(student_obss)
        teacher_acts = []
        for obs in student_obss:
            teacher_acts.append(predict_wrapper(obs))

        # Step d: Add the augmented state-action pairs back to aggregate
        obss.extend(student_obss)
        acts.extend(teacher_acts)
        dt_obss.extend(student_dt_obss)

        # Step e: Estimate the reward
        cur_rew = sum((rew for _, _, rew in student_trace)) / n_batch_rollouts
        logger.info(f"Student reward: {cur_rew:.4f}")

        students.append((student.clone(), cur_rew))
        # print(students[0][0].state_transformer)

    best_student = identify_best_policy(rl_test_wrapper, students, n_batch_rollouts)

    return best_student


def automl_train_dagger(
    trajectories=None,
    state_transformer=None,
    rl_test_wrapper=None,
    predict_wrapper=None,
    max_depth=None,
    max_leaves=None,
    searcher=None,
    epochs=100,
    train_frac=0.8,
    max_samples=20000,
    n_batch_rollouts=10,
    seed=1,
):
    np.random.seed(seed)
    students = []

    obss, acts, dt_obss = parse_trajectory(trajectories, state_transformer)
    student = DecisionTree(
        state_transformer=state_transformer, max_depth=max_depth, max_leaf_nodes=max_leaves, prune_factor=None
    )

    logger.info(f"Training student with {max_samples} points")
    # Dagger outer loop
    for i in range(epochs):
        logger.info(f"Epoch {i+1}/{epochs}:")

        # Step a: Train from a random subset of aggregated data
        cur_obss, cur_acts, cur_dt_obss = _sample(np.array(obss), np.array(acts), np.array(dt_obss), max_samples)
        obss_train, obss_test, acts_train, acts_test = train_test_split(
            cur_dt_obss, cur_acts, train_size=train_frac, shuffle=True
        )

        # logger.info(f"Training student with {len(cur_obss)} points")
        student, prune, train_score = searcher.search(obss_train, acts_train)
        train_acc = np.mean(acts_train == student.predict(obss_train))
        test_acc = np.mean(acts_test == student.predict(obss_test))
        logger.info(
            f"Train score: {train_score:5.3f}  | Train accuracy: {train_acc:5.3f} | Test accuracy: {test_acc:5.3f} | Leaves number : {student.tree.get_n_leaves():5d} | Configuration: {prune}  "
        )

        # Step b: Generate trace using student
        student_trace = get_dt_rollouts(rl_test_wrapper, student, n_batch_rollouts)
        student_obss = [obs for obs, _, _ in student_trace]
        student_dt_obss = [state_transformer(obs) for obs in student_obss]

        # Step c: Query the oracle for supervision
        # teacher_acts = teacher.predict(student_obss)
        teacher_acts = []
        for obs in student_obss:
            teacher_acts.append(predict_wrapper(obs))

        # Step d: Add the augmented state-action pairs back to aggregate
        obss.extend(student_obss)
        acts.extend(teacher_acts)
        dt_obss.extend(student_dt_obss)

        # Step e: Estimate the reward
        cur_rew = sum((rew for _, _, rew in student_trace)) / n_batch_rollouts
        logger.info(f"Student reward: {cur_rew:.4f}")

        students.append((student.clone(), cur_rew))

    best_student = identify_best_policy(rl_test_wrapper, students, n_batch_rollouts)

    return best_student
