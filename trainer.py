import datetime
import glob
import re
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.tensorboard import SummaryWriter
from utils import Logger, dict2str, savepath
import numpy as np
from collections import defaultdict
from functools import partial
from torch.utils.data import DataLoader, Dataset

class BaseLearner(object):
    def __init__(self, config, model, loader):
        self.loader = loader
        self.config = config

        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.device)
        # self.model = model_cls(config,)
        self.model = model

        if config.device != -1:
            self.model.cuda()

        self.epochs = config.epochs
        self.early_stop_rounds = config.early_stop
        self.learning_rate = config.learning_rate

        self.cur_epoch = 0
        self.cur_step = 0
        self.best_epoch = 0
        self.best_result = -np.inf

        self.optimizer = getattr(torch.optim, config.optimizer)(self.model.parameters(), lr=self.learning_rate)

        self.save_floder = os.path.join(savepath, config.task, config.model, config.commit)
        os.makedirs(self.save_floder, exist_ok=True)

        date = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        self.writer = SummaryWriter(self.save_floder, comment=date)
        self.logger = Logger(os.path.join(self.save_floder, 'run.log'))

    def train_one_epoch(self):
        loader = self.loader.train_dataloader()
        total_loss = []
        with tqdm.tqdm(loader, ncols=100) as pbar:
            for input in pbar:
                if (self.cur_step + 1) % self.config.eval_steps == 0:
                    self.test()
                train_loss, summary_tensor_dic = self.train_one_step(input)
                total_loss.append(train_loss)
                train_loss = self.smooth_loss(total_loss)
                pbar.set_description(f'train loss: {train_loss:.4f}')
                self.writer.add_scalar('train/train_loss', train_loss, global_step=self.cur_step)

                for k, v in summary_tensor_dic.items():
                    self.writer.add_scalar('train/' + k, v, global_step=self.cur_step)
                self.cur_step += 1

    def smooth_loss(self, losses):
        if len(losses) < 100:
            return sum(losses) / len(losses)
        else:
            return sum(losses[-100:]) / 100

    def train_one_step(self, input):
        self.model.train()
        if self.config.device >= 0:
            input = input.cuda()
        loss, summary_tensor_dic = self.model.calculate_loss(input)
        train_loss = loss.item()

        self.optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20, norm_type=2) #梯度剪裁
        self.optimizer.step()
        return train_loss, summary_tensor_dic

    def train(self):
        self.model.set_phase("train")
        while self.cur_epoch < self.epochs:
            self.train_one_epoch()
            self.test()
            if self.cur_epoch - self.best_epoch > self.early_stop_rounds:
                self.logger.info("early stop...")
                break
            self.cur_epoch += 1
        self.writer.add_hparams(self.config.to_parm_dict(), {'test/best_result': self.best_result})

    def evaluate(self, pred_dict):
        self.model.set_phase("evaluate")
        torch.save(pred_dict, os.path.join(self.save_floder, 'saved_embed.pkl'))
        ans = {}
        ans['AUC'] = roc_auc_score(pred_dict['y_trues'], pred_dict['y_preds'])
        return ans

    def test(self):
        self.model.set_phase("test")
        pred_dict = self.eval_one_epoch(loader=self.loader.test_dataloader())
        metrics = self.evaluate(pred_dict)
        self.logger.info(f'[epoch={self.cur_epoch}, step={self.cur_step}] {dict2str(metrics)}')
        if metrics['AUC'] > self.best_result:
            self.best_result = metrics['AUC']
            self.best_epoch = self.cur_epoch
            self.best_step = self.cur_step
            self.save_model()
        for key, value in metrics.items():
            self.writer.add_scalar(f'test/{key}', value, global_step=self.cur_step)
        return metrics

    @torch.no_grad()
    def eval_one_epoch(self, loader):
        self.model.eval()
        pred_dict = defaultdict(list)
        with tqdm.tqdm(loader, ncols=100) as pbar:
            for input in pbar:
                if self.config.device >= 0:
                    input = input.cuda()
                res = self.model.predict(input)
                if not isinstance(res, tuple):
                    scores, out_dict = res, {}
                else:
                    scores, out_dict = res
                pred_dict["y_trues"].append(input.label)
                pred_dict["y_preds"].append(scores)
                pred_dict["user_id"].append(input.user_id)
                for k, v in out_dict.items():
                    pred_dict[k].append(v)
            for key, val in pred_dict.items():
                pred_dict[key] = torch.cat(val, dim=0).cpu().numpy()
        return pred_dict

    def save_model(self):
        filename = os.path.join(
            self.save_floder, f"epoch={self.cur_epoch}-step={self.cur_step}-auc={self.best_result}.pth"
        )
        state = {
            'best_epoch': self.best_epoch,
            'cur_epoch': self.cur_epoch,
            'cur_step': self.cur_step,
            'state_dict': self.model.state_dict(),
        }
        torch.save(state, filename)

    def load_model(self):
        # use the best pth
        modelfiles = glob.glob(os.path.join(self.save_floder, '*.pth'))
        auces = [re.search("auc=(.*).pth", file).group(1) for file in modelfiles]
        auces = [float(auc) for auc in auces]
        idx = np.argmax(auces)

        state = torch.load(modelfiles[idx])
        self.cur_epoch = state['cur_epoch']
        self.cur_step = state['cur_step']
        self.best_epoch = state['best_epoch']
        self.model.load_state_dict(state['state_dict'])


class SeqDataLoader(object):
    def __init__(self, config, model, update=False):
        self.config = config
        self.model = model
        self.update = update
        self._load_data()
        self._build_dataset()

    def _load_data(self):
        print("load raw data")
        frac = self.config.sample
        task = self.config.task
        self.parts = 10 if (frac == 0.25 or frac == 0.0001 or frac == 0.2) else 1
        if not self.update:
            self.train_data = torch.load(f'./data/{task}/sampled/{frac}/train_samples.pkl')
            self.test_data = torch.load(f'./data/{task}/sampled/{frac}/test_samples.pkl')
        else:
            self.train_data = torch.load(f'./data/{task}/sampled/{frac}/train_samples_update.pkl')
            self.test_data = torch.load(f'./data/{task}/sampled/{frac}/test_samples_update.pkl')

    def _deserialize(self, data, keys):
        feed_dict = {}
        for idx, key in enumerate(keys):
            val = [tup[idx] for tup in data]
            feed_dict[key] = torch.stack(val, dim=0)
        return Input(**feed_dict)

    def _build_dataset(self):
        self.train_dataset = self.model.SeqDataset(self.config, dataset=self.train_data, phase='train')

        self.test_dataset = self.model.SeqDataset(self.config, dataset=self.test_data, phase='test')

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.train_batch_size,
            collate_fn=partial(self._deserialize, keys=self.train_dataset.keys),
            shuffle=True
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.config.eval_batch_size,
            collate_fn=partial(self._deserialize, keys=self.test_dataset.keys),
            shuffle=False
        )


class Input(object):
    def __init__(self, **kwgs):
        for key, value in kwgs.items():
            if isinstance(value, dict):
                value = Input(**value)
            setattr(self, key, value)

    def cuda(self):
        for key in self.__dict__:
            val = getattr(self, key)
            if isinstance(val, Input):
                val.cuda()
            else:
                setattr(self, key, val.cuda())
        return self