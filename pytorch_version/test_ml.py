import argparse
import os
import numpy as np
import shutil
import pandas as pd
from tqdm import tqdm
import torch
import torch.optim as optim
from get_adsb_dataset import get_finetune_dataloader
from utils import set_seed, get_logger_and_writer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def extract_features(encoder, dataloader, device, step="train"):
    encoder.eval()
    features = []
    targets = []
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc=f"Extracting features in {step} step"):
            features.append(encoder(inputs.float().to(device)).cpu().numpy())
            targets.append(labels.numpy())
    features = np.concatenate(features, axis=0)
    targets = np.concatenate(targets, axis=0)
    return features, targets


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    set_seed(2024)
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--shot", "-s", type=int, nargs='+', default=1)
    argparser.add_argument("--encoder", "-e", type=str, default="STC")
    argparser.add_argument("--classifier", "-c", type=str, nargs='+', default="LR")
    argparser.add_argument("--data_class", "-d", type=int, default=20)
    opts = argparser.parse_args()
    shots = [opts.shot] if not isinstance(opts.shot, list) else opts.shot
    classifiers = [opts.classifier] if not isinstance(opts.classifier, list) else opts.classifier
    if opts.encoder == "STC":
        from STC_Model import STC_Model as Model
    elif opts.encoder == "KTSLAN":
        from KTSLAN import KTSLAN as Model
    elif opts.encoder == "ResNet18":
        from ResNet18 import ResNet18 as Model
    else:
        raise ValueError("encoder not found.")
    for shot in shots:
        for ml_type in classifiers:
            ml_type = ml_type.upper()
            logger, writer, save_path, root_path = get_logger_and_writer(f"runs/PT_{opts.encoder}_FT_{ml_type}_{opts.data_class}Class/{shot}Shot")
            accs = []
            for i in range(100):
                logger.info("---------------------------------------------------------")
                logger.info(f"Iteration {i + 1}/100")
                set_seed(2024 + i)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = Model(n_classes=90, feature_dim=1024)
                model.load_state_dict(torch.load(f"runs/PT_{opts.encoder}/model.pth"))
                model = model.encoder.to(device)
                if ml_type == "KNN":
                    classifier = KNeighborsClassifier()
                elif ml_type == "LR":
                    classifier = LogisticRegression(max_iter=1000)
                else:
                    ml_type = "SVM"
                    classifier = SVC()

                train_dataloader, test_dataloader = get_finetune_dataloader(num_class=opts.data_class, shot=shot, random_seed=2024 + i)
                model.eval()
                features, labels = extract_features(model, train_dataloader, device, "train")
                classifier.fit(features, labels)
                features, labels = extract_features(model, test_dataloader, device, "test")
                test_accuracy = classifier.score(features, labels)
                logger.info(f"Test Accuracy: {test_accuracy * 100:.2f}%")
                writer.add_scalar("Test Accuracy", test_accuracy, i)
                accs.append(test_accuracy * 100)
            df = pd.DataFrame(accs)
            df.to_excel(os.path.join(save_path, f"{shot}Shot.xlsx"))
            shutil.copy(os.path.join(save_path, f"{shot}Shot.xlsx"), os.path.join(os.path.dirname(root_path), f"{shot}Shot.xlsx"))
            writer.close()
