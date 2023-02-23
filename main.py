import argparse

# import models
import importlib
from trainer import BaseLearner, SeqDataLoader
from utils import Config, make_dirs, seed_everything
from model import PRINT
parser = argparse.ArgumentParser()
parser.add_argument('--commit', type=str,default="debug")
parser.add_argument('--sample', type=float, default=0.1)
parser.add_argument('--model', type=str, default="PRINT")
parser.add_argument('--task', type=str, default="KDD2012")
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--seed', type=int, default=2022)
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=1)


args = parser.parse_args()

seed = args.seed
if __name__ == "__main__":

    args = Config(args)
    config = args.easy_use()
    seed_everything(seed, config.device)
    print(config.__dict__)

    make_dirs()
    model = PRINT(config)
    loader = SeqDataLoader(config,model)
    print ("sample lens:", len(loader.train_dataset))

    learner = BaseLearner(config, model, loader)
    args.tab_printer(learner.logger)

    learner.train()
    learner.test()
