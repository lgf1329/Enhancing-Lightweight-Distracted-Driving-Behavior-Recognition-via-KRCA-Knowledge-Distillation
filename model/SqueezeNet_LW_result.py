import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import time
import random
import os
from torch import nn, optim
import torch
import torchvision
from ptflops import get_model_complexity_info
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix

from SqueezeNet.model_SQ import SqueezeNet1
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from pathlib import Path
from PIL import Image
from torchvision import models
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from torchvision.transforms.functional import to_pil_image
import cv2

class DrivingDataset():
    def __init__(self, data_dir, img_size=224):
        self.data_dir = Path(data_dir)


        self.class_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]


        self.img_paths = []
        self.light_conditions = []
        for class_dir in self.class_dirs:
            class_name = class_dir.name


            for lighting_dir in class_dir.iterdir():
                if lighting_dir.is_dir():
                    light_condition = lighting_dir.name.lower()


                    img_files = list(lighting_dir.rglob("*.jpg")) + list(lighting_dir.rglob("*.png"))


                    for img_path in img_files:
                        self.img_paths.append((str(img_path.absolute()), class_name))
                        self.light_conditions.append(light_condition)



        light_counts = {}
        for light_condition in set(self.light_conditions):
            count = sum(1 for lc in self.light_conditions if lc == light_condition)
            light_counts[light_condition] = count


        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


        self.classes = sorted(set(c for _, c in self.img_paths))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )


        self.light_conditions_set = sorted(set(self.light_conditions))


        self.inv_normalize = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, class_name = self.img_paths[idx]

        try:

            img = Image.open(img_path).convert('RGB')

            label = self.class_to_idx[class_name]
        except Exception as e:

            return torch.zeros(3, 224, 224), 0


        return self.transform(img), label
class Tester:
    def __init__(self,model, model_path, test_data_dir):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        # self.model = model.to(self.device)
        # self.model.load_state_dict(torch.load(model_path))
        # self.model.eval()

        self.model = model.to(self.device)
        for module in self.model.modules():
            if isinstance(module, torch.nn.ReLU):
                module.inplace = False
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()


        self.test_dataset = DrivingDataset(test_data_dir)
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=64,
            shuffle=True,
            num_workers=4
        )


        self.criterion = nn.CrossEntropyLoss()
        self.class_names = self.test_dataset.classes

        self.gradients = None
        self.activations = None
        self._register_hooks()
    def evaluate(self):

        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                y_pred = self.model(x)[-1]
                preds = torch.argmax(y_pred, dim=1)
                loss = self.criterion(y_pred, y)


                total_loss += loss.item() * x.size(0)
                _, predicted = torch.max(y_pred.data, 1)
                correct += (predicted == y).sum().item()
                total += y.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())


        avg_loss = total_loss / len(self.test_loader.dataset)

        report_str = classification_report(
            all_labels, all_preds,
            target_names=self.class_names,
            zero_division=0
        )
        report_dict = classification_report(
            all_labels, all_preds,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )

        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy_score(all_labels, all_preds),
            'class_report_str': report_str,
            'class_report_dict': report_dict,
            'confusion_matrix': confusion_matrix(all_labels, all_preds)
        }
        return metrics

    def test_fps(self, batch_size=64, n_warmup=10, n_test=100):


        dummy_input = torch.randn(batch_size, 3, 224, 224).to(self.device)


        print("Warming up...")
        for _ in range(n_warmup):
            _ = self.model(dummy_input)


        print("Benchmarking...")
        start_time = time.time()
        for _ in range(n_test):
            _ = self.model(dummy_input)
        total_time = time.time() - start_time

        fps = n_test / total_time
        print(f"FPS: {fps:.2f} (batch_size={batch_size})")
        return fps

    def visualize_results(self, metrics, save_path=None):

        plt.rcParams.update({
            'font.family': 'Times New Roman',
            'font.weight': 'bold',
            'font.size': 20,
            'axes.titlesize': 14,
            'axes.titleweight': 'bold',
            'axes.labelsize': 13,
            'axes.labelweight': 'bold',
            'xtick.labelsize': 14,
            'ytick.labelsize': 14
        })

        class_names = ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9']
        cm_percent = metrics['confusion_matrix'].astype('float') / metrics['confusion_matrix'].sum(axis=1)[:,
                                                                   np.newaxis] * 100

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_percent,
                    annot=True, fmt='.2f', cmap='Blues',
                    annot_kws={
                        'fontsize': 12,
                        'fontfamily': 'Times New Roman'
                    },
                    xticklabels=class_names,
                    yticklabels=class_names)
        # plt.title('Confusion Matrix')
        # plt.xlabel('Predicted')
        # plt.ylabel('True')

        # ax.set_xlabel('Predicted', fontdict={'fontsize': 13, 'fontfamily': 'Times New Roman'})
        # ax.set_ylabel('Actual', fontdict={'fontsize': 13, 'fontfamily': 'Times New Roman'})


        plt.xticks(fontname='Times New Roman')
        plt.yticks(fontname='Times New Roman')

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def generate_report(self, metrics, report_path):
        report = metrics['class_report_str']
        with open(report_path, 'w') as f:
            f.write("===== Classification Report =====\n")
            f.write(report)

            f.write("\n\n===== Detailed Metrics =====\n")
            f.write(f"Total Samples: {len(self.test_dataset)}\n")
            f.write(f"Average Loss: {metrics['loss']:.4f}\n")
            f.write(f"Overall Accuracy: {metrics['accuracy']:.2%}\n")


            f.write("\nPer-class Metrics:\n")
            for cls in self.class_names:
                f.write(f"{cls}:\n")
                f.write(f"  Precision: {metrics['class_report_dict'][cls]['precision']:.4f}\n")
                f.write(f"  Recall:    {metrics['class_report_dict'][cls]['recall']:.4f}\n")
                f.write(f"  F1-score:  {metrics['class_report_dict'][cls]['f1-score']:.4f}\n\n")

    def _register_hooks(self):

        # target_layer = self.model.features[12].expand3x3
        # target_layer = self.model.features[3].expand_3x3
        target_layer = self.model.features
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].clone().detach()

        def forward_hook(module, input, output):
            self.activations = output.clone().detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def _get_features(self, x):

        _ = self.model.features(x)
        return self.activations.mean([2, 3])

    def plot_tsne(self, metrics, save_path=None, n_samples=500):
        indices = np.random.choice(len(self.test_dataset), min(n_samples, len(self.test_dataset)), replace=False)
        features = []
        labels = []


        with torch.no_grad():
            for i in indices:
                img, label = self.test_dataset[i]
                img = img.unsqueeze(0).to(self.device)
                features.append(self._get_features(img).cpu().numpy())
                labels.append(label)

        features = np.concatenate(features)
        labels = np.array(labels)


        tsne = TSNE(n_components=2, random_state=42,init='random', learning_rate=200.0)
        projections = tsne.fit_transform(features)


        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(projections[:, 0], projections[:, 1],
                              c=labels, cmap=plt.cm.get_cmap('tab10', 10),
                              alpha=0.6, edgecolors='w', linewidths=0.5)


        plt.legend(handles=scatter.legend_elements()[0],
                   labels=self.class_names,
                   bbox_to_anchor=(1.05, 1),
                   loc='upper left')

        plt.title("t-SNE Feature Projection")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.colorbar(scatter, boundaries=np.arange(11) - 0.5).set_ticks(np.arange(10))

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def _apply_grad_cam(self, img_tensor):

        img_var = img_tensor.unsqueeze(0).to(self.device).requires_grad_(True)
        output = self.model(img_var)[-1]
        # _,_,output = self.model(img_var)


        self.model.zero_grad()
        class_idx = torch.argmax(output).item()
        output[0, class_idx].backward()


        gradients = self.gradients.cpu().numpy()[0]
        activations = self.activations.cpu().numpy()[0]


        weights = np.mean(gradients, axis=(1, 2), keepdims=True)
        cam = np.sum(weights * activations, axis=0)


        # cam = np.maximum(cam, 0)
        # cam = cam / (cam.max() + 1e-8)
        # cam[cam < 0.2] = 0
        cam = cv2.resize(cam, (224, 224))

        # original_img = to_pil_image(img_tensor).convert("RGB")
        # original_img = np.array(original_img)
        #

        # cam = np.maximum(cam, 0)
        # cam = cam / (cam.max() + 1e-8)
        # cam = cv2.resize(cam, (224, 224))
        # cam[cam < 0.2] = 0
        # heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_VIRIDIS)
        # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        #

        # superimposed_img = heatmap * 0.3 + original_img * 0.7
        # return superimposed_img
        return cam

    # def plot_grad_cam(self, n_examples=5, save_path=None):

    #     indices = np.random.choice(len(self.test_dataset), n_examples)
    #
    #     plt.figure(figsize=(20, 15))
    #     for i, idx in enumerate(indices, 1):

    #         img_tensor, label = self.test_dataset[idx]
    #

    #         img_tensor_denorm = self.test_dataset.inv_normalize(img_tensor)
    #         original_img = to_pil_image(img_tensor_denorm)
    #

    #         superimposed_img = self._apply_grad_cam(img_tensor)
    #

    #         plt.subplot(2, n_examples, i)
    #         plt.imshow(original_img)
    #         plt.axis('off')
    #         plt.title(f"True: {self.class_names[label]}")
    #
    #         plt.subplot(2, n_examples, i + n_examples)
    #         plt.imshow(superimposed_img)
    #         plt.axis('off')
    #         plt.title(f"Saliency Map")
    #
    #     plt.suptitle("Grad-CAM Visualization", fontsize=16)
    #     if save_path:
    #         plt.savefig(save_path, bbox_inches='tight')
    #         plt.close()
    #     else:
    #         plt.show()

    def plot_grad_cam(self, n_examples=5, save_path=None):

        # indices = np.random.choice(len(self.test_dataset), n_examples)
        # for j in len(self.test_dataset):
        #     plt.figure(figsize=(20, 15))
        #     for i, idx in enumerate(j, 1):
        if save_path:
            os.makedirs(save_path, exist_ok=True)

        for idx in range(len(self.test_dataset)):

            img_tensor, label = self.test_dataset[idx]
            class_name = self.class_names[label]

            class_save_path = os.path.join(save_path, class_name)
            os.makedirs(class_save_path, exist_ok=True)


            img_tensor_denorm = self.test_dataset.inv_normalize(img_tensor)
            original_img = to_pil_image(img_tensor_denorm)
            img_width, img_height = original_img.size

            cam = self._apply_grad_cam(img_tensor)

            cam_norm = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            # heatmap = cm.jet_r(cam_norm)[..., :3]
            heatmap = cm.jet(cam_norm)[..., :3]

            # plt.subplot(2, n_examples, i)
            # plt.imshow(original_img)
            # plt.axis('off')
            # plt.title(f"True: {self.class_names[label]}")
            plt.figure(figsize=(img_width / 100, img_height / 100), dpi=100)
            # plt.subplot(2, n_examples, i + n_examples)
            plt.imshow(original_img)
            plt.imshow(heatmap, alpha=0.5)
            plt.axis('off')
            # plt.title(f"Saliency Map")
            #
            # plt.suptitle("Grad-CAM Visualization", fontsize=16)
            if save_path:

                save_name = f"{class_name}_heatmap_{idx:03d}.png"
                plt.savefig(
                    os.path.join(class_save_path, save_name),
                    bbox_inches='tight',
                    pad_inches=0,
                    dpi=100
                )
                plt.close()
            else:
                plt.show()


if __name__ == "__main__":
    '---------------SF---------------------'
    # data_dir_test = r'/home/luo/data/data_usa/test'
    # model = SqueezeNet1(version='custom', num_classes=10)

    # model_weight_path = r"/home/luo/KRCA_model_python/model_pth/SF/SqueezeNet_KRAC.pth"#2-our-99.52%
    # save_model = "test_report_SqueezeNet1_SF-our.txt"
    # save_svg ="SqueezeNet1_SF-our.svg"
    # save_grad_cam = r"/home/luo/SqueezeNet-our"




    '---------------ASU---------------------'
    data_dir_test = r'/home/luo/data/data_ASU_all/test'
    model = SqueezeNet1(version='custom', num_classes=10)

    model_weight_path = r"/home/luo/KRCA_model_python/model_pth/ASU/SqueezeNet_KRAC.pth"#our-99.83%
    save_model = "test_report_SqueezeNet1_ASU-our.txt"
    save_svg = "SqueezeNet1_ASU-our.svg"
    save_grad_cam = r"/home/luo/SqueezeNet_ASU-our"







    tester = Tester(
        model=model,
        model_path=model_weight_path,
        test_data_dir=data_dir_test
    )

    fps = tester.test_fps()
    print(fps,'fps')
    with open(save_model, 'a') as f:
        f.write(f"\nFPS: {fps:.2f}\n")


    # metrics = tester.evaluate()
    #

    # print(f"Test Accuracy: {metrics['accuracy']:.2%}")
    # print(f"Average Loss: {metrics['loss']:.4f}")
    #
    #
    # tester.visualize_results(metrics,save_path=save_svg)
    # tester.generate_report(metrics, report_path=save_model)
    #
