import jax
import jax.numpy as jnp               # JAX NumPy
from jax import grad
import jax.lax as lax

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
from sys import stdout

import models


from jax.config import config
#config.update("jax_enable_x64", True)

WRN = functools.partial(models.WideResNet, num_classes=10, depth=28, widen_factor=10)

#WRN = functools.partial(models.CNN)

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
    logits, _ = WRN().apply(params, batch['image'], mutable="batch_stats")
    return compute_metrics(logits, batch['label'])

def eval_model(model, eval_iter, eval_steps):
    batch_metrics = []
    for step, batch in zip(range(eval_steps), eval_iter):
        metrics = eval_step(model, to_jax_batch(batch))
        metrics = jax.device_get(metrics)
        batch_metrics.append(metrics)
    eval_batch_metrics = jax.device_get(batch_metrics)
    eval_batch_metrics = {
        k: np.mean([metrics[k] for metrics in eval_batch_metrics])
        for k in eval_batch_metrics[0]}
    return eval_batch_metrics['loss'], eval_batch_metrics['accuracy']

def clip_grad_norm(grad, max_norm):
    norm = jnp.linalg.norm(jax.tree_util.tree_leaves(jax.tree_map(jnp.linalg.norm, grad)))
    factor = jnp.minimum(1, max_norm / (norm + 1e-6))
    return jax.tree_map((lambda x: x * factor), grad)

@functools.partial(jax.jit, static_argnums=(4, 5, 6))
def train_step_dumb(state, batch, start_x, rng_keys, dummy_1, dummy_2, dummy_3):
    def loss_fn(params):
        logits, _ = WRN().apply(params, batch['image'], mutable="batch_stats")
        loss = jnp.mean(optax.softmax_cross_entropy(
            logits=logits, 
            labels=jax.nn.one_hot(batch['label'], num_classes=10)))
        return loss, logits
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits, batch['label'])
    metrics["grad_norm"] = jnp.linalg.norm(jax.tree_util.tree_leaves(jax.tree_map(jnp.linalg.norm, grads)))
    metrics["weights_norm"] = jnp.linalg.norm(jax.tree_util.tree_leaves(jax.tree_map(jnp.linalg.norm, state.params['params'])))
    return state, metrics, start_x



def log_prob(images, params):
    logits, _ = WRN().apply(params, images, mutable="batch_stats")
    return jax.scipy.special.logsumexp(logits, axis=1).sum()
log_prob_grad = jax.grad(log_prob, argnums=0)

def loss_fn(params, x_lab, y_lab, x, x_hat):
    logits_clf, _ = WRN().apply(params, x_lab, mutable="batch_stats")
    clf_loss = jnp.mean(optax.softmax_cross_entropy(
        logits=logits_clf, 
        labels=jax.nn.one_hot(y_lab, num_classes=10)))
        
    logits, _ = WRN().apply(params, x, mutable="batch_stats")
    logits_hat, _ = WRN().apply(params, x_hat, mutable="batch_stats")
    
    lse_x_hat = jnp.mean(jax.scipy.special.logsumexp(logits_hat, axis=1))
    lse_x = jnp.mean(jax.scipy.special.logsumexp(logits, axis=1))
        
    gen_loss = lse_x_hat - lse_x
    
    return clf_loss + gen_loss, (logits_clf, lse_x_hat, lse_x)
grad_fn = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)
 
@functools.partial(jax.jit, static_argnums=(5, 6, 7))
def train_step(state, batch_gen, batch_cls, x_t, rng_keys, sgld_lr, sgld_std, p_x_weight):
    x_t = lax.fori_loop(0, rng_keys.shape[0], lambda t, x_t: x_t + sgld_lr * log_prob_grad(x_t, state.params) + sgld_std * jax.random.normal(rng_keys[t], shape=x_t.shape), x_t) 

    (_, (logits, lse_x_hat, lse_x)), grads = grad_fn(state.params, batch_cls['image'], batch_cls['label'], batch_gen['image'], x_t)
   
    state = state.apply_gradients(grads=grads)

    metrics = compute_metrics(logits, batch_cls['label'])
    metrics["lse_x_hat"] = lse_x_hat
    metrics["lse_x"] = lse_x
    metrics["grad_norm"] = jnp.linalg.norm(jax.tree_util.tree_leaves(jax.tree_map(jnp.linalg.norm, grads))) 
    metrics["weights_norm"] = jnp.linalg.norm(jax.tree_util.tree_leaves(jax.tree_map(jnp.linalg.norm, state.params['params'])))

    return state, metrics, x_t

def print_metrics(metrics, args):
    mean_metrics = {
        k: np.mean([m[k] for m in metrics])
        for k in metrics[0]}
    
    print('Mean, loss: %.4f, accuracy: %.2f' % (mean_metrics['loss'], mean_metrics['accuracy'] * 100))
    if args.xent_only == 0:
        print('lse_x_hat: %.4f, lse_x: %.2f' % (metrics[-1]['lse_x_hat'], metrics[-1]['lse_x']))
        print('diff: %.4f' % (metrics[-1]['lse_x_hat'] - metrics[-1]['lse_x'],))
        
        if metrics[-1]['lse_x_hat'] < -1e4:
            raise Exception('Model diverged （>﹏<）')
    print('weights norm: %.4f' % (metrics[-1]['weights_norm']))
    print('gradient norm: %.4f' % (metrics[-1]['grad_norm']))
    stdout.flush()

def train_epoch(state, train_iter, train_iter_labeled, epoch, steps_per_epoch, replay_buffer, key, args):
    batch_metrics = []
    step_fn = train_step_dumb if args.xent_only == 1 else train_step

    for step in range(steps_per_epoch):  
        if step > 0 and step % args.print_every == 0:
            print(step * args.batch_size)
            print_metrics(jax.device_get(batch_metrics), args)

        batch_gen = to_jax_batch(next(train_iter)) 
        batch_cls = to_jax_batch(next(train_iter_labeled))

        key, *train_keys = jax.random.split(key, num=args.sgld_steps + 10)

        inds = jax.random.randint(train_keys[0], shape=(args.batch_size,), minval=0, maxval=replay_buffer.shape[0])
        buffer_samples = replay_buffer[inds]
        random_samples = generate_init_sample(train_keys[1], batch_gen['image'].shape, args)
        choose_random = (jax.random.uniform(train_keys[2], shape=(batch_gen['image'].shape[0], ), minval=0, maxval=1) < args.reinit_freq)[:, None, None, None]
        start_x = choose_random * random_samples + (1 - choose_random) * buffer_samples

        state, metrics, samples = step_fn(state, batch_gen, batch_cls, start_x, jnp.array(train_keys[-args.sgld_steps:]), args.sgld_lr, args.sgld_std, args.p_x_weight)
        
        batch_metrics.append(metrics)

        replay_buffer = jax.ops.index_update(replay_buffer, inds, samples)

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
    #tx = optax.adamw(learning_rate=schedule_fn, weight_decay=args.weight_decay)
    tx = optax.adam(learning_rate=args.lr)
  
    state = train_state.TrainState.create(apply_fn=cnn.apply, params=params, tx=tx)

    return state, key

def prepare_dummy_state():
    rng = jax.random.PRNGKey(54)
    key, init_key = jax.random.split(rng)

    cnn = WRN()
    params = cnn.init(init_key, jnp.ones((1, 32, 32, 3)))

    tx = optax.adamw(learning_rate=0.001)
  
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
