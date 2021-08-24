from train_utils import WRN, generate_init_sample, prepare_state, train_epoch, eval_model, save_model, load_model
import time
import argparse
import input_pipeline
import jax
import jax.numpy as jnp

parser = argparse.ArgumentParser("Energy Based Models")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--replay_buffer_size", type=int, default=10000)
parser.add_argument("--reinit_freq", type=float, default=0.05) # probability p of initialising starting point for sgld from Uniform[-1;1], with probability 1 - p the point is taken from replay buffer
parser.add_argument("--lr", type=float, default=1e-4) # learning rate
parser.add_argument("--warmup_iters", type=int, default=1)
parser.add_argument("--xent_only", type=int, default=0) # if set to 1 the model will be trained only for cross-entropy loss
parser.add_argument("--sgld_steps", type=int, default=20)
parser.add_argument("--sgld_lr", type=float, default=1.0) 
parser.add_argument("--sgld_std", type=float, default=1e-2)
parser.add_argument("--p_x_weight", type=float, default=1.0) # weight of loss -LogSumExp(f(x)) + LogSumExp(f(x_hat))
parser.add_argument("--weight_decay", type=float, default=0.0) 
parser.add_argument("--img_std", type=float, default=0.03) 
parser.add_argument("--print_every", type=int, default=10)
parser.add_argument("--save_dir", type=str, default='./saved_models')
parser.add_argument("--ckpt_every", type=int, default=100000)
parser.add_argument("--save_img_every", type=int, default=200)
parser.add_argument("--load_file", type=str)
parser.add_argument("--seed", type=int, default=54)

args = parser.parse_args()
args.eval_batch_size = 10000

print("data init started")
data_source = input_pipeline.CIFAR10DataSource(args)
train_ds = data_source.train_ds
train_iter = iter(train_ds)
steps_per_epoch = data_source.TRAIN_IMAGES // args.batch_size

test_ds = input_pipeline.to_jax_batch(next(iter(data_source.eval_ds)))
print("data init finished")

print("model init started")
state, key = prepare_state(args)
state_dict = None
if args.load_file:
    state_dict = load_model(state, args, args.load_file)
    state = state_dict["state"]
print("model init finished")

save_model(state.params["params"], jnp.ones((1, 32, 32, 3)), args, 'start_params_only.pt')
save_model(state, jnp.ones((args.replay_buffer_size, 32, 32, 3)), args, 'start_state.pt')

num_epochs = 150

key, subkey = jax.random.split(key)
replay_buffer = generate_init_sample(subkey, (args.replay_buffer_size, 32, 32, 3), args)
if state_dict is not None:
    replay_buffer = state_dict["replay_buffer"]

print("training started")

for epoch in range(1, num_epochs + 1):
    key, subkey = jax.random.split(key)
    state, train_metrics, replay_buffer = train_epoch(state, train_iter, epoch, steps_per_epoch, replay_buffer, subkey, args)
    test_loss, test_accuracy = eval_model(state.params, test_ds)
    print('Testing - epoch: %d, loss: %.2f, accuracy: %.2f' % (epoch, test_loss, test_accuracy * 100))
    if epoch % args.ckpt_every == 0:
        save_model(state, replay_buffer, args, f'{WRN.func.__name__}_ckpt_{epoch}.pt')
