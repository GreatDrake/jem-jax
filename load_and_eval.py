from train_utils import generate_init_sample, load_model, prepare_state, train_epoch, eval_model, save_model
import argparse
import input_pipeline
import jax

parser = argparse.ArgumentParser("Energy Based Models")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-4) # learning rate
parser.add_argument("--warmup_iters", type=int, default=1)
parser.add_argument("--weight_decay", type=float, default=0.0) 
parser.add_argument("--img_std", type=float, default=0.03) 
parser.add_argument("--save_dir", type=str, default='./saved_models')
parser.add_argument("--filename", type=str)

args = parser.parse_args()
args.eval_batch_size = 10000
args.seed = 54
args.replay_buffer_size = 10000

data_source = input_pipeline.CIFAR10DataSource(args)
test_ds = input_pipeline.to_jax_batch(next(iter(data_source.eval_ds)))

init_state, key = prepare_state(args)
state_dict = load_model(init_state, args, args.filename)
state = state_dict["state"]

print("model loading finished")

test_loss, test_accuracy = eval_model(state.params, test_ds)
print('loss: %.2f, accuracy: %.2f' % (test_loss, test_accuracy * 100))


