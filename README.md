PPO for Discrete OpenAI Gym Environments
========================================
Training Proximal Policy Optimization (PPO) with PyTorch on discrete-action Gym tasks


1. Overview
-----------

This project implements the Proximal Policy Optimization (PPO) algorithm in PyTorch
for OpenAI Gym environments with **discrete action spaces**, such as:

- MountainCar-v0
- CartPole-v1

The implementation follows an **Actor–Critic** architecture and uses the
**PPO-Clip** objective. TensorBoard is integrated for monitoring training
progress (losses, rewards, episode lengths, etc.).


2. Main Features
----------------

- Actor–Critic structure
  - Actor: outputs a probability distribution over actions π(a|s)
  - Critic: estimates the state value function V(s)

- PPO-Clip objective
  - Uses the probability ratio `ratio = π_new(a|s) / π_old(a|s)`
  - Clamps the ratio to stabilize policy updates
  - Uses advantage estimates (Gt − V(s)) in the update

- TensorBoard logging
  - Logs losses (actor / critic)
  - Logs per-episode reward and number of steps


3. Requirements
---------------

3.1 Recommended Python Version
------------------------------

Python 3.7 – 3.10 is recommended.

3.2 Required Packages
---------------------

Install the dependencies with:

    pip install torch torchvision
    pip install numpy
    pip install gym
    pip install tensorboardX

Note:
- If you use a newer version of Gym (e.g., `gym>=0.26` or `gymnasium`),
  the code already includes basic handling for the updated `reset()` and
  `step()` return formats.


4. Suggested Project Structure
------------------------------

A typical project layout might look like this:

    project_root/
    ├─ ppo_main.py          # Main PPO training script
    ├─ README_PPO.txt       # This README (or README.md)
    ├─ exp/                 # TensorBoard logs (created automatically)
    └─ param/
       ├─ net_param/        # Saved Actor / Critic parameters
       └─ img/              # Reserved for plots (not strictly required)

The script will automatically create:

- ./param/net_param
- ./param/img
- ./exp

if they do not already exist.


5. How to Run
-------------

5.1 Train on MountainCar-v0 (default)
-------------------------------------

Simply run:

    python ppo_main.py

or explicitly specify the environment:

    python ppo_main.py --env-name MountainCar-v0


5.2 Train on CartPole-v1
------------------------

    python ppo_main.py --env-name CartPole-v1


5.3 Enable Rendering
--------------------

To visualize the environment while training:

    python ppo_main.py --env-name CartPole-v1 --render

Warning:
- Enabling `--render` can significantly slow down training.
  A common workflow is:
  1) Train without rendering.
  2) Load the saved model later and run a short demo with rendering.


5.4 Show All Command-Line Options
---------------------------------

    python ppo_main.py --help


6. Important Hyperparameters
----------------------------

The script uses `argparse` to define hyperparameters. Common options
(typical defaults shown as examples) include:

- `--env-name` (str, default: `"MountainCar-v0"`)
  Name of the Gym environment.

- `--gamma` (float, default: `0.99`)
  Discount factor γ.

- `--seed` (int, default: `1`)
  Random seed.

- `--render` (bool, default: `False`)
  Whether to render the environment.

- `--log-dir` (str, default: `"./exp"`)
  Directory for TensorBoard logs.

- `--param-dir` (str, default: `"./param"`)
  Root directory for saving model parameters.

- `--max-episode` (int, e.g. `1000`)
  Maximum number of training episodes.

- `--ppo-update-time` (int, e.g. `10`)
  Number of PPO optimization epochs per update.

- `--buffer-capacity` (int, e.g. `8000`)
  Buffer capacity (used conceptually; implementation may not require filling it fully).

- `--batch-size` (int, e.g. `32`)
  Mini-batch size for PPO updates.

- `--actor-lr` (float, e.g. `1e-3`)
  Learning rate of the Actor network.

- `--critic-lr` (float, e.g. `3e-3`)
  Learning rate of the Critic network.

- `--clip-param` (float, e.g. `0.2`)
  PPO clipping parameter ε. The ratio is typically constrained within [1−ε, 1+ε].

- `--max-grad-norm` (float, e.g. `0.5`)
  Maximum gradient norm for gradient clipping.


7. Algorithm and Code Structure
-------------------------------

7.1 Actor–Critic Networks
-------------------------

Actor (policy network):

- Input: state vector
- Output: probability distribution over actions π(a|s)

Typical structure:

- Fully connected layer (state → hidden size, e.g. 128 units)
- ReLU activation
- Fully connected output layer (hidden size → number of actions)
- Softmax over the output logits to obtain the probabilities

Critic (value network):

- Input: state vector
- Output: scalar value V(s)

Typical structure:

- Fully connected layer (state → hidden size, e.g. 128 units)
- ReLU activation
- Fully connected output layer (hidden size → 1)


7.2 Transition Storage
----------------------

During interaction with the environment, transitions are stored in a buffer.

A transition typically contains:

- `state`
- `action`
- `a_prob` (the probability of the selected action under the **old** policy)
- `reward`
- `next_state`

After enough samples are collected (at least one mini-batch), the agent
performs a PPO update using these stored transitions.


7.3 PPO Update Overview
-----------------------

The PPO update (e.g., `PPO.update()`) usually involves:

1. **Compute discounted returns Gt**

   For each time step t in a trajectory, the discounted return is:

       Gt = r_t + γ r_{t+1} + γ^2 r_{t+2} + ...

   In code this is often implemented by iterating backwards through the
   rewards and accumulating:

       R = 0
       Gt_list = []
       for r in reversed(reward_list):
           R = r + gamma * R
           Gt_list.insert(0, R)

2. **Multiple PPO epochs**

   For `ppo_update_time` iterations:

   - Use a sampler (e.g., `BatchSampler` + `SubsetRandomSampler`)
     to generate mini-batches of data.

   - **Critic update:**
     - Compute the predicted values V(s) for the batch.
     - Minimize the mean squared error between Gt and V(s):

           value_loss = MSE(Gt_batch, V_batch)

   - **Actor update:**
     - Compute the new action probabilities π_new(a|s) for the batch.
     - Form the probability ratio:

           ratio = π_new(a|s) / (π_old(a|s) + 1e-8)

     - Compute the advantage:

           advantage = Gt_batch - V_batch.detach()

     - Compute PPO-Clip surrogate objectives:

           surr1 = ratio * advantage
           surr2 = clamp(ratio, 1 − clip_param, 1 + clip_param) * advantage

       The policy loss is:

           action_loss = −mean(min(surr1, surr2))

     - Backpropagate `action_loss` and clip gradients by `max_grad_norm` if needed.


8. TensorBoard Logging
----------------------

This project uses `tensorboardX.SummaryWriter` to log:

- Per PPO update:
  - `loss/action_loss`
  - `loss/value_loss`

- Per episode:
  - `Episode/step`   (number of steps taken in the episode)
  - `Episode/reward` (total reward of the episode)

To launch TensorBoard:

    tensorboard --logdir=./exp

Then open the printed URL in your browser (often `http://localhost:6006`).


9. Model Saving
---------------

Within the main training loop, the script periodically saves the model
parameters (for example, every 100 episodes):

    if i_ep % 100 == 0:
        agent.save_param(prefix=f"{env_name}_ep{i_ep}")

The default save paths are under:

    ./param/net_param/{env_name}_ep{episode}_actor_{timestamp}.pth
    ./param/net_param/{env_name}_ep{episode}_critic_{timestamp}.pth

You can change the root directory using the `--param-dir` argument, e.g.:

    python ppo_main.py --param-dir ./models


10. Summary
-----------

- This project provides a clean, practical implementation of PPO for
  **discrete-action** OpenAI Gym environments using PyTorch.

- It combines:
  - Actor–Critic architecture
  - PPO-Clip objective
  - TensorBoard logging
  - Automatic parameter saving

- It is suitable both as:
  - A learning resource for understanding PPO + Gym + PyTorch, and
  - A base code to extend to new environments or network architectures.
