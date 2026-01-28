"""Generate offline datasets for the Zermelo point maze environment.

Adapted from generate_locomaze.py. Uses the same BFS-based oracle policy
since the point actor is rule-based (follow subgoal direction).

Usage:
    python data_gen_scripts/generate_zermelo.py \
        --env_name zermelo-pointmaze-medium-v0 \
        --dataset_type navigate \
        --num_episodes 1000 \
        --save_path datasets/zermelo_pointmaze_medium_navigate.npz
"""
import pathlib
from collections import defaultdict

import gymnasium
import numpy as np
from absl import app, flags
from tqdm import trange

import ogbench.locomaze  # noqa

FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'zermelo-pointmaze-medium-v0', 'Environment name.')
flags.DEFINE_string('dataset_type', 'navigate', 'Dataset type.')
flags.DEFINE_string('save_path', None, 'Save path.')
flags.DEFINE_float('noise', 0.2, 'Gaussian action noise level.')
flags.DEFINE_integer('num_episodes', 1000, 'Number of episodes.')
flags.DEFINE_integer('max_episode_steps', 1001, 'Maximum number of steps in an episode.')
flags.DEFINE_string('flow_field_path', None, 'Path to a .npy flow field file (default: built-in field).')


def main(_):
    assert FLAGS.dataset_type in ['path', 'navigate', 'stitch', 'explore']

    env_kwargs = dict(
        terminate_at_goal=False,
        max_episode_steps=FLAGS.max_episode_steps,
    )
    if FLAGS.flow_field_path is not None:
        env_kwargs['flow_field_path'] = FLAGS.flow_field_path

    env = gymnasium.make(FLAGS.env_name, **env_kwargs)
    ob_dim = env.observation_space.shape[0]

    # Point oracle: last 2 dims of agent_ob are the subgoal direction.
    def actor_fn(ob, temperature):
        return ob[-2:]

    # Store all empty cells and vertex cells.
    all_cells = []
    vertex_cells = []
    maze_map = env.unwrapped.maze_map
    for i in range(maze_map.shape[0]):
        for j in range(maze_map.shape[1]):
            if maze_map[i, j] == 0:
                all_cells.append((i, j))
                if (
                    maze_map[i - 1, j] == 0
                    and maze_map[i + 1, j] == 0
                    and maze_map[i, j - 1] == 1
                    and maze_map[i, j + 1] == 1
                ):
                    continue
                if (
                    maze_map[i, j - 1] == 0
                    and maze_map[i, j + 1] == 0
                    and maze_map[i - 1, j] == 1
                    and maze_map[i + 1, j] == 1
                ):
                    continue
                vertex_cells.append((i, j))

    dataset = defaultdict(list)
    total_steps = 0
    total_train_steps = 0
    num_train_episodes = FLAGS.num_episodes
    num_val_episodes = FLAGS.num_episodes // 10

    for ep_idx in trange(num_train_episodes + num_val_episodes):
        if FLAGS.dataset_type in ['path', 'navigate', 'explore']:
            init_ij = all_cells[np.random.randint(len(all_cells))]
            goal_ij = vertex_cells[np.random.randint(len(vertex_cells))]
        elif FLAGS.dataset_type == 'stitch':
            init_ij = all_cells[np.random.randint(len(all_cells))]
            adj_cells = []
            adj_steps = 4
            bfs_map = maze_map.copy()
            for i in range(bfs_map.shape[0]):
                for j in range(bfs_map.shape[1]):
                    bfs_map[i][j] = -1
            bfs_map[init_ij[0], init_ij[1]] = 0
            queue = [init_ij]
            while len(queue) > 0:
                i, j = queue.pop(0)
                for di, dj in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if (
                        0 <= ni < bfs_map.shape[0]
                        and 0 <= nj < bfs_map.shape[1]
                        and maze_map[ni, nj] == 0
                        and bfs_map[ni, nj] == -1
                    ):
                        bfs_map[ni][nj] = bfs_map[i][j] + 1
                        queue.append((ni, nj))
                        if bfs_map[ni][nj] == adj_steps:
                            adj_cells.append((ni, nj))
            goal_ij = adj_cells[np.random.randint(len(adj_cells))] if len(adj_cells) > 0 else init_ij
        else:
            raise ValueError(f'Unsupported dataset_type: {FLAGS.dataset_type}')

        ob, _ = env.reset(options=dict(task_info=dict(init_ij=init_ij, goal_ij=goal_ij)))

        done = False
        step = 0
        cur_subgoal_dir = None

        while not done:
            if FLAGS.dataset_type == 'explore':
                if step % 10 == 0:
                    cur_subgoal_dir = np.random.randn(2)
                    cur_subgoal_dir = cur_subgoal_dir / (np.linalg.norm(cur_subgoal_dir) + 1e-6)
                subgoal_dir = cur_subgoal_dir
            else:
                subgoal_xy, _ = env.unwrapped.get_oracle_subgoal(env.unwrapped.get_xy(), env.unwrapped.cur_goal_xy)
                subgoal_dir = subgoal_xy - env.unwrapped.get_xy()
                subgoal_dir = subgoal_dir / (np.linalg.norm(subgoal_dir) + 1e-6)

            agent_ob = env.unwrapped.get_ob(ob_type='states')
            # Exclude the agent's position and add the subgoal direction.
            agent_ob = np.concatenate([agent_ob[2:], subgoal_dir])
            action = actor_fn(agent_ob, temperature=0)
            action = action + np.random.normal(0, FLAGS.noise, action.shape)
            action = np.clip(action, -1, 1)

            next_ob, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            success = info['success']

            if success and FLAGS.dataset_type == 'navigate':
                goal_ij = vertex_cells[np.random.randint(len(vertex_cells))]
                env.unwrapped.set_goal(goal_ij)

            dataset['observations'].append(ob)
            dataset['actions'].append(action)
            dataset['terminals'].append(done)
            dataset['qpos'].append(info['prev_qpos'])
            dataset['qvel'].append(info['prev_qvel'])

            ob = next_ob
            step += 1

        total_steps += step
        if ep_idx < num_train_episodes:
            total_train_steps += step

    print('Total steps:', total_steps)

    train_path = FLAGS.save_path
    val_path = FLAGS.save_path.replace('.npz', '-val.npz')
    pathlib.Path(train_path).parent.mkdir(parents=True, exist_ok=True)

    train_dataset = {}
    val_dataset = {}
    for k, v in dataset.items():
        if 'observations' in k and v[0].dtype == np.uint8:
            dtype = np.uint8
        elif k == 'terminals':
            dtype = bool
        else:
            dtype = np.float32
        train_dataset[k] = np.array(v[:total_train_steps], dtype=dtype)
        val_dataset[k] = np.array(v[total_train_steps:], dtype=dtype)

    for path, ds in [(train_path, train_dataset), (val_path, val_dataset)]:
        np.savez_compressed(path, **ds)


if __name__ == '__main__':
    app.run(main)
