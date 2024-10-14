import os.path
import shutil
import argparse
import torch
import numpy as np
import torch.nn.functional as F
from get_adsb_dataset import get_pretrain_dataloader
# from learner import Learner
from STC_Loss import STCLoss
from utils import set_seed, get_logger_and_writer
from torch.optim.lr_scheduler import StepLR


def train(net, loss_fn, optim, sche, dataloader, device):
    """
    :param net: 网络
    :param optimizer: 优化器
    :param dataloader: 训练集
    :param device: 可用显卡
    :return:
    """
    net.train()
    correct = 0
    loss_batch_value = 0
    loss_ce_value = 0
    loss_tri_value = 0
    loss_center_value = 0
    for x_spt, y_spt in dataloader:
        x_spt, y_spt = x_spt.to(device), y_spt.long().to(device)
        optim.zero_grad()
        embedding, logic = net(x_spt)
        loss_batch, loss_ce, loss_tri, loss_center = loss_fn(logic, embedding, y_spt)
        loss_batch.backward()
        optim.step()
        if sche is not None:
            sche.step()
        predict = F.softmax(logic, dim=1).argmax(dim=1)
        correct += torch.eq(predict, y_spt).sum().item()
        loss_batch_value += loss_batch.item()
        loss_ce_value += loss_ce.item()
        loss_tri_value += loss_tri.item()
        loss_center_value += loss_center.item()
    return correct, loss_batch_value / len(dataloader), loss_ce_value / len(dataloader), loss_tri_value / len(dataloader), loss_center_value / len(
        dataloader)


def test(net, loss_fn, dataloader, device):
    """
    :param net:  网络
    :param x_qry:   [query_size, c_, h, w]
    :param y_qry:   [query_size]
    :return:
    """
    net.eval()
    correct = 0
    loss_batch_value = 0
    loss_ce_value = 0
    loss_tri_value = 0
    loss_center_value = 0
    with torch.no_grad():
        for x_qry, y_qry in dataloader:
            x_qry, y_qry = x_qry.to(device), y_qry.long().to(device)
            embedding_q, logic_q = net(x_qry)
            loss_batch, loss_ce, loss_tri, loss_center = loss_fn(logic_q, embedding_q, y_qry)
            predict_q = F.softmax(logic_q, dim=1).argmax(dim=1)
            correct += torch.eq(predict_q, y_qry).sum().item()
            loss_batch_value += loss_batch.item()
            loss_ce_value += loss_ce.item()
            loss_tri_value += loss_tri.item()
            loss_center_value += loss_center.item()
    return correct, loss_batch_value / len(dataloader), loss_ce_value / len(dataloader), loss_tri_value / len(dataloader), loss_center_value / len(
        dataloader)


def train_and_test(logger, writer, net, loss_fn, optim, sche, update_step_test, train_data_loader, test_data_loader, save_path, device):
    """
    :param x_spt:   [set_size, c_, h, w]
    :param y_spt:   [set_size]
    :return:
    """
    losses_train = [0 for _ in range(update_step_test)]
    losses_ce_train = [0 for _ in range(update_step_test)]
    losses_tri_train = [0 for _ in range(update_step_test)]
    losses_center_train = [0 for _ in range(update_step_test)]
    corrects_train = [0 for _ in range(update_step_test)]

    losses_test = [0 for _ in range(update_step_test)]
    losses_ce_test = [0 for _ in range(update_step_test)]
    losses_tri_test = [0 for _ in range(update_step_test)]
    losses_center_test = [0 for _ in range(update_step_test)]
    corrects_test = [0 for _ in range(update_step_test)]
    best_loss = 100000
    for k in range(0, update_step_test):
        corrects_train[k], losses_train[k], losses_ce_train[k], losses_tri_train[k], losses_center_train[k] = train(net, loss_fn, optim, sche,
                                                                                                                    train_data_loader, device)
        corrects_test[k], losses_test[k], losses_ce_test[k], losses_tri_test[k], losses_center_test[k] = test(net, loss_fn, test_data_loader, device)
        logger.info(
            "epoch:{}, train/loss_total:{:.7f}, train/loss_ce:{:.7f}, train/loss_tri:{:.7f}, train/loss_center:{:.7f}, train/acc:{:.4f}, test/loss_total:{:.7f}, test/loss_ce:{:.7f}, test/loss_tri:{:.7f}, test/loss_center:{:.7f}, test/acc:{:.4f}".format(
                k,
                losses_train[k],
                losses_ce_train[k],
                losses_tri_train[k],
                losses_center_train[k],
                np.array(corrects_train[k]) / len(train_data_loader.dataset),
                losses_test[k],
                losses_ce_test[k],
                losses_tri_test[k],
                losses_center_test[k],
                np.array(corrects_test[k]) / len(test_data_loader.dataset)
            ))
        writer.add_scalar('train/loss_total', losses_train[k], k)
        writer.add_scalar('train/loss_ce', losses_ce_train[k], k)
        writer.add_scalar('train/loss_tri', losses_tri_train[k], k)
        writer.add_scalar('train/loss_center', losses_center_train[k], k)
        writer.add_scalar('train/acc', np.array(corrects_train[k]) / len(train_data_loader.dataset), k)
        writer.add_scalar('test/loss_total', losses_test[k], k)
        writer.add_scalar('test/loss_ce', losses_ce_test[k], k)
        writer.add_scalar('test/loss_tri', losses_tri_test[k], k)
        writer.add_scalar('test/loss_center', losses_center_test[k], k)
        writer.add_scalar('test/acc', np.array(corrects_test[k]) / len(test_data_loader.dataset), k)
        if best_loss > losses_test[k] or k == 0:
            best_loss = losses_test[k]
            torch.save(net.state_dict(), os.path.join(save_path, "model.pth"))
            logger.info(f"best model is saved in epoch{k}, best loss is decreased.")


class Config:
    def __init__(
            self,
            random_seed: int = 2024,
            encoder: str = "STC",
            batch_size: int = 32,
            epochs: int = 300,
            lr: float = 0.01,
            loss_type: int = 2,
            n_classes: int = 90,
            feature_dim: int = 1024,
            save_path: str = "./runs",
    ):
        argparser = argparse.ArgumentParser()
        argparser.add_argument("--encoder", "-e", type=str, default=encoder)
        argparser.add_argument("--loss_type", "-t", type=int, default=loss_type)
        opts = argparser.parse_args()
        self.random_seed = random_seed
        self.encoder = opts.encoder
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.loss_type = opts.loss_type
        self.n_classes = n_classes
        self.feature_dim = feature_dim
        self.save_path = save_path + "/PT_" + opts.encoder
        self.root_path = save_path + "/PT_" + opts.encoder


def main():
    conf = Config()
    if conf.encoder == "STC":
        from STC_Model import STC_Model as Model
    elif conf.encoder == "KTSLAN":
        from KTSLAN import KTSLAN as Model
    elif conf.encoder == "ResNet18":
        from ResNet18 import ResNet18 as Model
    else:
        raise ValueError("encoder not found.")
    set_seed(conf.random_seed)
    logger, writer, conf.save_path, conf.root_path = get_logger_and_writer(conf.save_path)

    train_data_loader, test_data_loader = get_pretrain_dataloader(num_class=conf.n_classes, batch_size=conf.batch_size, random_seed=conf.random_seed)
    model = Model(n_classes=conf.n_classes, feature_dim=conf.feature_dim).to(device)
    loss_fn = STCLoss(weights=(1.0, 0.01, 0.01), triplet_margin=5, num_classes=conf.n_classes, feat_dim=conf.feature_dim, device=device, triplet_loss_type=conf.loss_type)

    optim = torch.optim.AdamW(model.parameters(), lr=conf.lr, weight_decay=0)
    # optim = torch.optim.Adam(model.parameters(), lr=conf.lr, weight_decay=0)
    # sche = StepLR(optim, step_size=50, gamma=0.1)
    sche = None
    train_and_test(logger, writer, model, loss_fn, optim, sche, conf.epochs, train_data_loader, test_data_loader, conf.save_path, device)
    shutil.copy(os.path.join(conf.save_path, "model.pth"), os.path.join(conf.root_path, "model.pth"))
    writer.close()


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()
