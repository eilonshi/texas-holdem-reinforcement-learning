autoplay = True  # play automatically if played against keras-rl

window_length = 1
nb_max_start_steps = 1  # random action
train_interval = 100  # train every 100 steps
nb_steps_warmup = 50  # before training starts, should be higher than start steps
nb_steps = 100000
memory_limit = int(nb_steps / 2)
batch_size = 500  # items sampled from memory to train
enable_double_dqn = False

# TODO: change to capslock
