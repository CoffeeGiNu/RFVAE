import os
import sys
import copy
from tqdm.auto import tqdm

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def step(model, inputs, labels, optimizer, criterion, device, is_train=True):
    # model = model.to(device)
    if device == 'cuda':
        inputs = torch.from_numpy(np.array(inputs)).half().to(device)
    else:
        inputs = torch.from_numpy(np.array(inputs)).to(device)
    # labels = torch.from_numpy(np.array(labels)).to(device)
    with torch.set_grad_enabled(is_train):
        optimizer.zero_grad()
        results = model(inputs)
        # loss = results['loss']
        loss_dict = criterion(
            inputs=inputs, 
            mean=results['mean'], 
            log_var=results['log_var'], 
            outputs=results['outputs'])
        loss = loss_dict['loss']
        if is_train:
            if not torch.isnan(loss):
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
                optimizer.step()
            # loss.backward()
            # optimizer.step()
    # results['loss'] = loss
    # results['reconstraction_error'] = reconstraction_error
    # results['q_latent_loss'] = q_latent_loss
    # results['e_latent_loss'] = e_latent_loss
    return model, results, loss_dict


def epoch_loop(model, data_set, optimizer, criterion, device, epoch, num_epochs, 
    batch_size, earlystopping=None, is_train=True, profiler=None, writer=None):
    if is_train:
        model.train().to(device)
    else:
        model.eval().to(device)
    with tqdm(
        total=len(data_set),
        bar_format=None if 'ipykernel' in sys.modules else '{l_bar}{bar:15}{r_bar}{bar:-10b}',
        ncols=None if 'ipykernel' in sys.modules else 108,
        unit='batch',
        leave=True
    ) as pbar:
        total = loss_sum = accuracy_sum = 0
        pbar.set_description(
            f"Epoch[{epoch}/{num_epochs}]({'train' if is_train else 'valid'})")
        for data in data_set:
            inputs = data['image']
            labels = data['label']
            model, results, loss_dict = step(
                model, inputs, labels, optimizer, criterion, device, is_train=is_train)
            if writer:
                writer.add_scalar("loss_train/loss", 
                    loss_dict['loss'].cpu().detach().numpy(), epoch + total)
                writer.add_scalar("loss_train/reconstraction_loss", 
                    loss_dict['reconstruction_loss'].cpu().detach().numpy(), epoch + total)
                writer.add_scalar("loss_train/kl_divergence", 
                    loss_dict['kl_divergence'].cpu().detach().numpy(), epoch + total)
            total += batch_size
            loss_sum += float(loss_dict['loss'].cpu().detach().numpy()) * batch_size
            running_loss = loss_sum / total
            # accuracy_sum += (torch.argmax(preds, axis=1).detach().cpu().numpy() == labels).sum()
            # running_accuracy = accuracy_sum.item() / total
            pbar.set_postfix(
                {"loss":round(running_loss, 3), 
                #  "accuracy":round(running_accuracy, 3)
                }
            )
            if profiler:
                profiler.step()
            pbar.update(1)
        if epoch % 10:
            if writer:
                if not is_train:
                    r = results['outputs']
                    grid = torchvision.utils.make_grid(r, nrow=r.shape[0])
                    writer.add_image("reconstraction_image", grid, global_step=epoch)
        if earlystopping:
            earlystopping((running_loss), model)
    return model


class VAELoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.eps = np.spacing(1)

    def forward(self, inputs, mean, log_var, outputs):
        # reconstruction_loss = -torch.mean(torch.sum(
        #     inputs * torch.log(outputs + self.eps) 
        #     + (1 - inputs) * torch.log(1 - outputs + self.eps), dim=1))
        # reconstruction_loss = -torch.sum(
        #     inputs * torch.log(outputs + self.eps) 
        #     + (1 - inputs) * torch.log(1 - outputs + self.eps))
        # reconstruction_loss = -torch.mean(torch.sum(
        #     inputs * torch.log(outputs) + (1-inputs) * torch.log(1-outputs), dim=1))
        # reconstruction_loss = F.binary_cross_entropy(inputs, outputs)
        reconstruction_loss = outputs.shape[2] * outputs.shape[3] * F.mse_loss(inputs, outputs)
        kl_divergence = -1/2 * torch.mean(torch.sum(
            1 + log_var - torch.pow(mean, 2) - torch.exp(log_var), dim=1))
        loss = reconstruction_loss + kl_divergence

        return dict(reconstruction_loss=reconstruction_loss, 
            kl_divergence=kl_divergence, loss=loss)


# 参考：PyTorchでEarlyStoppingを実装する
# https://qiita.com/ku_a_i/items/ba33c9ce3449da23b503
class EarlyStopping:
    """earlystoppingクラス"""

    def __init__(self, patience=5, verbose=False, path='models/', model_file_name='checkpoint.pth'):
        """引数：最小値の非更新数カウンタ、表示設定、モデル格納path"""

        self.patience = patience    #設定ストップカウンタ
        self.verbose = verbose      #表示の有無
        self.counter = 0            #現在のカウンタ値
        self.best_score = None      #ベストスコア
        self.early_stop = False     #ストップフラグ
        self.val_loss_min = np.Inf   #前回のベストスコア記憶用
        self.path = path             #ベストモデル格納path
        self.model_file_name = model_file_name
        os.makedirs(path) if not os.path.exists(path) else None

    def __call__(self, val_loss, model):
        """
        特殊(call)メソッド
        実際に学習ループ内で最小lossを更新したか否かを計算させる部分
        """
        score = -val_loss

        if self.best_score is None:  #1Epoch目の処理
            self.best_score = score   #1Epoch目はそのままベストスコアとして記録する
            self.checkpoint(val_loss, model)  #記録後にモデルを保存してスコア表示する
        elif score < self.best_score:  # ベストスコアを更新できなかった場合
            self.counter += 1   #ストップカウンタを+1
            if self.verbose:  #表示を有効にした場合は経過を表示
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')  #現在のカウンタを表示する 
            if self.counter >= self.patience:  #設定カウントを上回ったらストップフラグをTrueに変更
                self.early_stop = True
        else:  #ベストスコアを更新した場合
            self.best_score = score  #ベストスコアを上書き
            self.checkpoint(val_loss, model)  #モデルを保存してスコア表示
            self.counter = 0  #ストップカウンタリセット

    def checkpoint(self, val_loss, model):
        '''ベストスコア更新時に実行されるチェックポイント関数'''
        if self.verbose:  #表示を有効にした場合は、前回のベストスコアからどれだけ更新したか？を表示
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.to('cpu').state_dict(), self.path+self.model_file_name)  #ベストモデルを指定したpathに保存
        self.val_loss_min = val_loss  #その時のlossを記録する