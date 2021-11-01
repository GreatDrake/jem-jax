from train_utils import WRN, log_prob_grad, generate_init_sample, load_model, prepare_state, train_epoch, eval_model, save_model
import argparse
import input_pipeline
import jax
import jax.numpy as jnp
import models
import flax
import jax.lax as lax
from image_utils import save_image
import os
import sys

parser = argparse.ArgumentParser("Energy Based Models")
parser.add_argument("--filename", type=str)
parser.add_argument("--img_dir", type=str)
parser.add_argument("--sgld_steps", type=int, default=40)
parser.add_argument("--n_sample_steps", type=int, default=1000)
parser.add_argument("--reinit_freq", type=float, default=0.05)
parser.add_argument("--sgld_lr", type=float, default=1.0)
parser.add_argument("--sgld_std", type=float, default=1e-2)
parser.add_argument("--replay_buffer_size", type=int, default=10000)
parser.add_argument("--batch_size", type=int, default=64)

args = parser.parse_args()
args.seed = 42
args.eval_batch_size = 100
args.img_std = 0.0
eval_size = 10000

params_dict = WRN().init(jax.random.PRNGKey(args.seed), jax.numpy.ones((1, 32, 32, 3)))

data_source = input_pipeline.CIFAR10DataSource(args)
eval_iter = iter(data_source.eval_ds)

with open(args.filename, "rb") as f:
    params = flax.serialization.from_bytes(params_dict, f.read())

print("model loading finished")
sys.stdout.flush()

test_loss, test_accuracy = eval_model(params, eval_iter, eval_size // args.eval_batch_size)

print('loss: %.2f, accuracy: %.2f' % (test_loss, test_accuracy * 100))

key = jax.random.PRNGKey(args.seed)
key, subkey = jax.random.split(key)
replay_buffer = generate_init_sample(subkey, (args.replay_buffer_size, 32, 32, 3), args)

print("generating images")


@jax.jit
def langevin(rng_keys, x_t, params):
    return lax.fori_loop(0, rng_keys.shape[0], lambda t, x_t: x_t + args.sgld_lr * log_prob_grad(x_t, params) + args.sgld_std * jax.random.normal(rng_keys[t], shape=x_t.shape), x_t)

for i in range(args.n_sample_steps):
    if i % 2 == 0:
        print(i)
        sys.stdout.flush()
    key, *train_keys = jax.random.split(key, num=args.sgld_steps + 10)
 
    inds = jax.random.randint(train_keys[0], shape=(args.batch_size,), minval=0, maxval=replay_buffer.shape[0])
    buffer_samples = replay_buffer[inds]
    random_samples = generate_init_sample(train_keys[1], (args.batch_size, 32, 32, 3), args)
    choose_random = (jax.random.uniform(train_keys[2], shape=(args.batch_size, ), minval=0, maxval=1) < args.reinit_freq)[:, None, None, None]
    x_t = choose_random * random_samples + (1 - choose_random) * buffer_samples

    rng_keys = jnp.array(train_keys[-args.sgld_steps:])

    #x_t = lax.fori_loop(0, rng_keys.shape[0], lambda t, x_t: x_t + args.sgld_lr * log_prob_grad(x_t, params) + args.sgld_std * jax.random.normal(rng_keys[t], shape=x_t.shape), x_t)
    x_t = langevin(rng_keys, x_t, params)

    replay_buffer = jax.ops.index_update(replay_buffer, inds, x_t)

    if i % 50 == 0:
        for idx, img in enumerate(replay_buffer): 
           save_image(os.path.join(args.img_dir, f'{idx}.png'), img)
             

print("saving images")
for idx, img in enumerate(replay_buffer):
    save_image(os.path.join(args.img_dir, f'{idx}.png'), img)
    if idx % 1000 == 0:
        print(idx)
