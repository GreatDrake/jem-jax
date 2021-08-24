import jax
import jax.numpy as jnp               # JAX NumPy
from jax import grad

import flax
from flax import linen as nn          # The Linen API
from flax.training import train_state
import optax                          # The Optax gradient processing and optimization library

import numpy as np                    # Ordinary NumPy

from input_pipeline import to_jax_batch
from image_utils import save_image_grid

import jax_resnet

import functools
import os.path

class CNN(nn.Module):
  @nn.compact
  # Provide a constructor to register a new parameter 
  # and return its initial value
  def __call__(self, x):
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    
    x = nn.Dropout(0.1)(x, deterministic=True)

    x = nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    
    x = nn.Dropout(0.1)(x, deterministic=True)

    x = x.reshape((x.shape[0], -1)) # Flatten
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)

    x = nn.Dropout(0.2)(x, deterministic=True)    

    x = nn.Dense(features=10)(x)    # There are 10 classes in MNIST
    return x

#WRN = functools.partial(jax_resnet.WideResNet50, norm_cls=None, n_classes=10)
#WRN = functools.partial(jax_resnet.WideResNet50, n_classes=10)
#WRN = functools.partial(jax_resnet.ResNet50, norm_cls=None, n_classes=10)
#WRN = functools.partial(jax_resnet.ResNet50, n_classes=10)
#WRN = functools.partial(jax_resnet.ResNet18, norm_cls=None, n_classes=10)
WRN = functools.partial(CNN)


def generate_init_sample(key, shape, args):
    return jax.random.uniform(key, shape=shape, minval=-1, maxval=1)

def compute_metrics(logits, labels):
    loss = jnp.mean(optax.softmax_cross_entropy(logits, jax.nn.one_hot(labels, num_classes=10)))
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {
        'loss': loss,
        'accuracy': accuracy
    }
    return metrics

@jax.jit
def eval_step(params, batch):
    logits = WRN().apply(params, batch['image'])
    return compute_metrics(logits, batch['label'])

def eval_model(model, test_ds):
    metrics = eval_step(model, test_ds)
    metrics = jax.device_get(metrics)
    eval_summary = jax.tree_map(lambda x: x.item(), metrics)
    return eval_summary['loss'], eval_summary['accuracy']

def clip_grad_norm(grad, max_norm):
    norm = jnp.linalg.norm(jax.tree_util.tree_leaves(jax.tree_map(jnp.linalg.norm, grad)))
    factor = jnp.minimum(1, max_norm / (norm + 1e-6))
    return jax.tree_map((lambda x: x * factor), grad)

@functools.partial(jax.jit, static_argnums=(4, 5, 6))
def train_step_dumb(state, batch, start_x, rng_keys, dummy_1, dummy_2, dummy_3):
    def loss_fn(params):
        logits = WRN().apply(params, batch['image'])
        loss = jnp.mean(optax.softmax_cross_entropy(
            logits=logits, 
            labels=jax.nn.one_hot(batch['label'], num_classes=10)))
        return loss, logits
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits, batch['label'])
    metrics["grad_norm"] = jnp.linalg.norm(jax.tree_util.tree_leaves(jax.tree_map(jnp.linalg.norm, grads)))
    return state, metrics, start_x

@functools.partial(jax.jit, static_argnums=(4, 5, 6))
def train_step(state, batch, start_x, rng_keys, sgld_lr, sgld_std, p_x_weight):
    def log_prob(images):
        logits = WRN().apply(state.params, images)
        return jax.scipy.special.logsumexp(logits, axis=1).sum()
    log_prob_grad = jax.grad(log_prob)

    x_t = start_x
    for t in range(len(rng_keys)):
        d = sgld_lr * log_prob_grad(x_t) + sgld_std * jax.random.normal(rng_keys[t], shape=x_t.shape)
        x_t += d

    def loss_fn(params, x, x_hat):
        logits = WRN().apply(params, x)
        logits_hat = WRN().apply(params, x_hat)

        clf_loss = jnp.mean(optax.softmax_cross_entropy(
            logits=logits, 
            labels=jax.nn.one_hot(batch['label'], num_classes=10)))
        
        lse_x_hat = jnp.mean(jax.scipy.special.logsumexp(logits_hat, axis=1))
        lse_x = jnp.mean(jax.scipy.special.logsumexp(logits, axis=1))
        
        gen_loss = lse_x_hat - lse_x

        return clf_loss + p_x_weight * gen_loss, (logits, lse_x_hat, lse_x)
        
        #exp_xy = jnp.mean((logits * jax.nn.one_hot(batch['label'], num_classes=10)).sum(axis=1))
        #return -exp_xy + lse_x_hat, (logits, lse_x_hat, lse_x)

 
    grad_fn = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)
    (_, (logits, lse_x_hat, lse_x)), grads = grad_fn(state.params, batch['image'], x_t)
    
    state = state.apply_gradients(grads=clip_grad_norm(grads, 15))

    metrics = compute_metrics(logits, batch['label'])
    metrics["lse_x_hat"] = lse_x_hat
    metrics["lse_x"] = lse_x
    metrics["grad_norm"] = jnp.linalg.norm(jax.tree_util.tree_leaves(jax.tree_map(jnp.linalg.norm, grads)))

    return state, metrics, x_t

def print_metrics(metrics, args):
    mean_metrics = {
        k: np.mean([m[k] for m in metrics])
        for k in metrics[0]}

    print('Mean, loss: %.4f, accuracy: %.2f' % (mean_metrics['loss'], mean_metrics['accuracy'] * 100))
    if args.xent_only == 0:
        print('lse_x_hat: %.4f, lse_x: %.2f' % (metrics[-1]['lse_x_hat'], metrics[-1]['lse_x']))
        print('diff: %.4f' % (metrics[-1]['lse_x_hat'] - metrics[-1]['lse_x'],))
    print('gradient norm: %.4f' % (metrics[-1]['grad_norm']))

def train_epoch(state, train_iter, epoch, steps_per_epoch, replay_buffer, key, args):
    batch_metrics = []
    step_fn = train_step_dumb if args.xent_only == 1 else train_step

    for step, batch in zip(range(steps_per_epoch), train_iter):
        if step > 0 and step % args.print_every == 0:
            print(step * args.batch_size)
            print_metrics(jax.device_get(batch_metrics), args)

        batch = to_jax_batch(batch)

        key, *train_keys = jax.random.split(key, num=args.sgld_steps + 10)

        start_x = generate_init_sample(train_keys[0], batch['image'].shape, args)
        cnt_old = jax.random.bernoulli(train_keys[1], shape=(args.batch_size,), p=1 - args.reinit_freq).sum()
        old_idx = jax.random.randint(train_keys[2], shape=(args.batch_size,), minval=0, maxval=replay_buffer.shape[0])

        start_x = jax.ops.index_update(start_x, jnp.arange(cnt_old), replay_buffer[old_idx[:cnt_old]])

        state, metrics, samples = step_fn(state, batch, start_x, train_keys[-args.sgld_steps:], args.sgld_lr, args.sgld_std, args.p_x_weight)
        
        batch_metrics.append(metrics)

        replay_buffer = jax.ops.index_update(replay_buffer, old_idx, samples)

        if step % args.save_img_every == 0:
            img_name = "imgs/sample_%d_%d.png" % (epoch, step)
            save_image_grid(os.path.join(args.save_dir, img_name), samples[:16], 4, 4)

    training_batch_metrics = jax.device_get(batch_metrics)
    training_epoch_metrics = {
        k: np.mean([metrics[k] for metrics in training_batch_metrics])
        for k in training_batch_metrics[0]}

    print('Training - epoch: %d, loss: %.4f, accuracy: %.2f' % (epoch, training_epoch_metrics['loss'], training_epoch_metrics['accuracy'] * 100))

    return state, training_epoch_metrics, replay_buffer

def prepare_state(args):
    rng = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(rng)

    cnn = WRN()
    params = cnn.init(init_key, jnp.ones((1, 32, 32, 3)))

    def warmup_and_staircase(value, warmup_iters):
        def schedule(count):
            rate = jnp.where(count <= warmup_iters, value * count / warmup_iters, value)
            return rate

        return schedule

    schedule_fn = warmup_and_staircase(args.lr, args.warmup_iters)
    tx = optax.adamw(learning_rate=schedule_fn, weight_decay=args.weight_decay)

    state = train_state.TrainState.create(apply_fn=cnn.apply, params=params, tx=tx)

    return state, key

def save_model(state, replay_buffer, args, filename):
    ckpt_dict = {
        "state": state,
        "replay_buffer": replay_buffer
    }
    with open(os.path.join(args.save_dir, filename), "wb") as f:
        f.write(flax.serialization.to_bytes(ckpt_dict))

def load_model(state, args, filename):
    ckpt_dict = {
         "state": state,
         "replay_buffer": jnp.ones((args.replay_buffer_size, 32, 32, 3))
     }
    with open(os.path.join(args.save_dir, filename), "rb") as f:
        new_dict = flax.serialization.from_bytes(ckpt_dict, f.read())
    return new_dict
