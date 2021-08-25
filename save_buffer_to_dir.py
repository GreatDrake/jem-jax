from train_utils import generate_init_sample, load_model, prepare_state, train_epoch, eval_model, save_model
import argparse
import input_pipeline
import jax
import os.path
from image_utils import save_image

parser = argparse.ArgumentParser("Save buffer")
parser.add_argument("--ckpt_file", type=str)
parser.add_argument("--dir", type=str)

args = parser.parse_args()
args.eval_batch_size = 10000
args.seed = 54
args.replay_buffer_size = 10000
args.save_dir = "./saved_models"
args.img_std = 0.03
args.weight_decay = 0.0
args.warmup_iters = 1
args.lr = 1e-4
args.batch_size = 10000

data_source = input_pipeline.CIFAR10DataSource(args)
test_ds = input_pipeline.to_jax_batch(next(iter(data_source.eval_ds)))

init_state, key = prepare_state(args)
state_dict = load_model(init_state, args, args.ckpt_file)
state = state_dict["state"]
replay_buffer = state_dict["replay_buffer"]

for idx, img in enumerate(replay_buffer):
    save_image(os.path.join(args.dir, f'{idx}.png'), img)
    if idx % 1000 == 0:
        print(idx)

