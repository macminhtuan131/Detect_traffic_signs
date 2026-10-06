import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
import numpy as np
from tqdm.autonotebook import tqdm
from dataset import CocoDataset, collater, Normalizer, Augmenter, Resizer
from model import EfficientDet


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class Detector():
    def __init__(self, verbose=1):
        self.system_dict = {};
        self.system_dict["verbose"] = verbose;
        self.system_dict["local"] = {};
        self.system_dict["dataset"] = {};
        self.system_dict["dataset"]["train"] = {};
        self.system_dict["dataset"]["val"] = {};
        self.system_dict["dataset"]["val"]["status"] = False;

        self.system_dict["params"] = {};
        self.system_dict["params"]["image_size"] = 512;
        self.system_dict["params"]["batch_size"] = 8;
        self.system_dict["params"]["num_workers"] = 3;
        self.system_dict["params"]["use_gpu"] = True;
        self.system_dict["params"]["gpu_devices"] = [0];
        self.system_dict["params"]["lr"] = 0.0001;
        self.system_dict["params"]["num_epochs"] = 10;
        self.system_dict["params"]["val_interval"] = 1;
        self.system_dict["params"]["es_min_delta"] = 0.0;
        self.system_dict["params"]["es_patience"] = 0;


        self.system_dict["output"] = {};
        self.system_dict["output"]["log_path"] = "tensorboard/signatrix_efficientdet_coco";
        self.system_dict["output"]["saved_path"] = "trained/";
        self.system_dict["output"]["best_epoch"] = 0;
        self.system_dict["output"]["best_loss"] = 1e5;



    def Train_Dataset(self, root_dir, coco_dir, img_dir, set_dir, batch_size=8, image_size=512,
                      use_gpu=True, num_workers=3, include_empty=False):
        if image_size <= 0 or image_size % 128 != 0:
            raise ValueError("image_size must be a positive multiple of 128")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.system_dict["dataset"]["train"]["root_dir"] = root_dir;
        self.system_dict["dataset"]["train"]["coco_dir"] = coco_dir;
        self.system_dict["dataset"]["train"]["img_dir"] = img_dir;
        self.system_dict["dataset"]["train"]["set_dir"] = set_dir;


        self.system_dict["params"]["batch_size"] = batch_size;
        self.system_dict["params"]["image_size"] = image_size;
        self.system_dict["params"]["use_gpu"] = use_gpu;
        self.system_dict["params"]["num_workers"] = num_workers;

        use_cuda = self.system_dict["params"]["use_gpu"] and torch.cuda.is_available()
        self.system_dict["local"]["num_gpus"] = torch.cuda.device_count() if use_cuda else 1
        self.system_dict["local"]["device"] = torch.device("cuda" if use_cuda else "cpu")
        torch.manual_seed(123)
        np.random.seed(123)
        random.seed(123)
        if use_cuda:
            torch.cuda.manual_seed_all(123)
            torch.backends.cudnn.benchmark = True

        data_generator = torch.Generator()
        data_generator.manual_seed(123)

        self.system_dict["local"]["training_params"] = {"batch_size": self.system_dict["params"]["batch_size"],
                                                           "shuffle": True,
                                                           "drop_last": False,
                                                           "collate_fn": collater,
                                                           "num_workers": self.system_dict["params"]["num_workers"],
                                                           "pin_memory": use_cuda,
                                                           "persistent_workers": self.system_dict["params"]["num_workers"] > 0,
                                                           "worker_init_fn": seed_worker,
                                                           "generator": data_generator}

        self.system_dict["local"]["training_set"] = CocoDataset(root_dir=self.system_dict["dataset"]["train"]["root_dir"] + "/" + self.system_dict["dataset"]["train"]["coco_dir"],
                                                            img_dir = self.system_dict["dataset"]["train"]["img_dir"],
                                                            set_dir = self.system_dict["dataset"]["train"]["set_dir"],
                                                            transform = transforms.Compose([Augmenter(), Normalizer(), Resizer(image_size)]),
                                                            include_empty=include_empty)

        self.system_dict["local"]["training_generator"] = DataLoader(self.system_dict["local"]["training_set"],
                                                                    **self.system_dict["local"]["training_params"]);


    def Val_Dataset(self, root_dir, coco_dir, img_dir, set_dir, annotation_file=None,
                    image_dir=None, correction_file=None):
        self.system_dict["dataset"]["val"]["status"] = True;
        self.system_dict["dataset"]["val"]["root_dir"] = root_dir;
        self.system_dict["dataset"]["val"]["coco_dir"] = coco_dir;
        self.system_dict["dataset"]["val"]["img_dir"] = img_dir;
        self.system_dict["dataset"]["val"]["set_dir"] = set_dir;

        self.system_dict["local"]["val_params"] = {"batch_size": self.system_dict["params"]["batch_size"],
                                                   "shuffle": False,
                                                   "drop_last": False,
                                                   "collate_fn": collater,
                                                   "num_workers": self.system_dict["params"]["num_workers"]}

        self.system_dict["local"]["val_set"] = CocoDataset(root_dir=self.system_dict["dataset"]["val"]["root_dir"] + "/" + self.system_dict["dataset"]["val"]["coco_dir"],
                                                    img_dir = self.system_dict["dataset"]["val"]["img_dir"],
                                                    set_dir = self.system_dict["dataset"]["val"]["set_dir"],
                                                    transform=transforms.Compose([Normalizer(), Resizer(self.system_dict["params"]["image_size"])]),
                                                    category_ids=list(self.system_dict["local"]["training_set"].coco_labels.values()),
                                                    annotation_file=annotation_file,
                                                    image_dir=image_dir,
                                                    correction_file=correction_file)

        self.system_dict["local"]["test_generator"] = DataLoader(self.system_dict["local"]["val_set"],
                                                                **self.system_dict["local"]["val_params"])


    def Model(self, gpu_devices=[0], checkpoint_path=None):
        num_classes = self.system_dict["local"]["training_set"].num_classes();
        efficientdet = EfficientDet(num_classes=num_classes)
        target_class_names = [
            self.system_dict["local"]["training_set"].labels[index]
            for index in range(num_classes)
        ]
        efficientdet.class_names = target_class_names

        if checkpoint_path is not None:
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            checkpoint_model = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            checkpoint_model = checkpoint_model.module if isinstance(checkpoint_model, nn.DataParallel) else checkpoint_model
            source_class_names = getattr(checkpoint_model, 'class_names', None)
            if not source_class_names or len(source_class_names) != checkpoint_model.num_classes:
                categories = sorted(
                    self.system_dict["local"]["training_set"].coco.dataset.get('categories', []),
                    key=lambda category: category['id'],
                )
                # Legacy models included the unused `A` category. Prefer the
                # original category prefix when no checkpoint metadata exists.
                source_class_names = [
                    category['name'] for category in categories[:checkpoint_model.num_classes]
                ]
            copied_classes = self._warm_start_detector(
                efficientdet, checkpoint_model, target_class_names, source_class_names
            )
            print(f"Warm-started detector from {checkpoint_path} ({copied_classes} classes copied)")

        device = self.system_dict["local"]["device"]
        if device.type == "cuda":
            self.system_dict["params"]["gpu_devices"] = gpu_devices
        efficientdet = efficientdet.to(device)
        if device.type == "cuda" and len(gpu_devices) > 1:
            efficientdet = torch.nn.DataParallel(efficientdet, device_ids=gpu_devices)

        self.system_dict["local"]["model"] = efficientdet;
        self.system_dict["local"]["model"].train();

    @staticmethod
    def _warm_start_detector(target_model, source_model, target_class_names, source_class_names):
        target_state = target_model.state_dict()
        compatible_state = {
            name: value for name, value in source_model.state_dict().items()
            if name in target_state
            and target_state[name].shape == value.shape
            and name not in {'classifier.header.weight', 'classifier.header.bias'}
        }
        target_model.load_state_dict(compatible_state, strict=False)

        source_lookup = {name: index for index, name in enumerate(source_class_names)}
        shared_names = [name for name in target_class_names if name in source_lookup]
        source_classes = source_model.num_classes
        target_classes = target_model.num_classes
        source_anchors = source_model.classifier.num_anchors
        target_anchors = target_model.classifier.num_anchors
        if source_anchors != target_anchors:
            return 0

        with torch.no_grad():
            source_weight = source_model.classifier.header.weight.view(
                source_anchors, source_classes, *source_model.classifier.header.weight.shape[1:]
            )
            target_weight = target_model.classifier.header.weight.view(
                target_anchors, target_classes, *target_model.classifier.header.weight.shape[1:]
            )
            source_bias = source_model.classifier.header.bias.view(source_anchors, source_classes)
            target_bias = target_model.classifier.header.bias.view(target_anchors, target_classes)
            for target_index, class_name in enumerate(target_class_names):
                source_index = source_lookup.get(class_name)
                if source_index is None or source_index >= source_classes:
                    continue
                target_weight[:, target_index].copy_(source_weight[:, source_index])
                target_bias[:, target_index].copy_(source_bias[:, source_index])
        return len(shared_names)


    def Set_Hyperparams(self, lr=0.0001, val_interval=1, es_min_delta=0.0, es_patience=0):
        self.system_dict["params"]["lr"] = lr;
        self.system_dict["params"]["val_interval"] = val_interval;
        self.system_dict["params"]["es_min_delta"] = es_min_delta;
        self.system_dict["params"]["es_patience"] = es_patience;


        self.system_dict["local"]["optimizer"] = torch.optim.Adam(self.system_dict["local"]["model"].parameters(),
                                                                    self.system_dict["params"]["lr"]);

        self.system_dict["local"]["scheduler"] = torch.optim.lr_scheduler.ReduceLROnPlateau(self.system_dict["local"]["optimizer"],
                                                                    patience=3)


    def Train(self, num_epochs=2, model_output_dir="trained/", export_onnx=False,
              use_amp=True, freeze_backbone_bn=True):
        self.system_dict["output"]["log_path"] = "tensorboard/signatrix_efficientdet_coco";
        self.system_dict["output"]["saved_path"] = model_output_dir;
        self.system_dict["params"]["num_epochs"] = num_epochs;

        os.makedirs(self.system_dict["output"]["log_path"], exist_ok=True)

        os.makedirs(self.system_dict["output"]["saved_path"], exist_ok=True)

        writer = SummaryWriter(self.system_dict["output"]["log_path"])
        device = self.system_dict["local"]["device"]
        amp_enabled = use_amp and device.type == 'cuda'
        scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)

        num_iter_per_epoch = len(self.system_dict["local"]["training_generator"])
        print('self.system_dict["dataset"]["val"]["status"]',self.system_dict["dataset"]["val"]["status"])
        for epoch in range(self.system_dict["params"]["num_epochs"]):
            self.system_dict["local"]["model"].train()
            if freeze_backbone_bn:
                model = self.system_dict["local"]["model"]
                model = model.module if isinstance(model, nn.DataParallel) else model
                for module in model.backbone_net.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        module.eval()

            epoch_loss = []
            progress_bar = tqdm(self.system_dict["local"]["training_generator"])
            for iter, data in enumerate(progress_bar):
                self.system_dict["local"]["optimizer"].zero_grad(set_to_none=True)
                device = self.system_dict["local"]["device"]
                non_blocking = device.type == 'cuda'
                images = data['img'].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
                annotations = data['annot'].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                    cls_loss, reg_loss = self.system_dict["local"]["model"]([images, annotations])
                    cls_loss = cls_loss.mean()
                    reg_loss = reg_loss.mean()
                    loss = cls_loss + reg_loss
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss encountered at epoch {epoch + 1}, iteration {iter + 1}")
                if loss.item() == 0:
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(self.system_dict["local"]["optimizer"])
                torch.nn.utils.clip_grad_norm_(self.system_dict["local"]["model"].parameters(), 0.1)
                scaler.step(self.system_dict["local"]["optimizer"])
                scaler.update()
                epoch_loss.append(loss.item())
                total_loss = float(np.mean(epoch_loss))

                progress_bar.set_description(
                    'Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. Batch loss: {:.5f} Total loss: {:.5f}'.format(
                        epoch + 1, self.system_dict["params"]["num_epochs"], iter + 1, num_iter_per_epoch,
                        cls_loss.item(), reg_loss.item(), loss.item(), total_loss))
                writer.add_scalar('Train/Total_loss', total_loss, epoch * num_iter_per_epoch + iter)
                writer.add_scalar('Train/Regression_loss', reg_loss.item(), epoch * num_iter_per_epoch + iter)
                writer.add_scalar('Train/Classification_loss (focal loss)', cls_loss.item(), epoch * num_iter_per_epoch + iter)

            if not epoch_loss:
                raise RuntimeError("No training batches produced a usable loss")
            train_loss = float(np.mean(epoch_loss))
            monitored_loss = train_loss
            validation_enabled = self.system_dict["dataset"]["val"]["status"]
            validation_epoch = (
                validation_enabled
                and (epoch + 1) % self.system_dict["params"]["val_interval"] == 0
            )
            if validation_epoch:
                monitored_loss = self._validate()
                writer.add_scalar('Validation/Total_loss', monitored_loss, epoch + 1)
                print(f"Validation loss: {monitored_loss:.5f}")

            self.system_dict["local"]["scheduler"].step(monitored_loss)

            # Keep a resumable latest checkpoint and a separate best checkpoint.
            # Selecting on validation loss avoids silently shipping the final,
            # potentially overfit, epoch.
            torch.save(
                self.system_dict["local"]["model"],
                os.path.join(self.system_dict["output"]["saved_path"], "signatrix_efficientdet_coco_latest.pth"),
            )
            compare_checkpoint = not validation_enabled or validation_epoch
            if (compare_checkpoint
                    and monitored_loss < self.system_dict["output"]["best_loss"]
                    - self.system_dict["params"]["es_min_delta"]):
                self.system_dict["output"]["best_loss"] = monitored_loss
                self.system_dict["output"]["best_epoch"] = epoch + 1
                torch.save(
                    self.system_dict["local"]["model"],
                    os.path.join(self.system_dict["output"]["saved_path"], "signatrix_efficientdet_coco.pth"),
                )
            elif (validation_epoch
                  and self.system_dict["params"]["es_patience"] > 0
                  and epoch + 1 - self.system_dict["output"]["best_epoch"]
                  >= self.system_dict["params"]["es_patience"]):
                print(f"Early stopping at epoch {epoch + 1}; best epoch was {self.system_dict['output']['best_epoch']}")
                break

            if export_onnx:
                self._export_onnx()


        writer.close()

    def _validate(self):
        model = self.system_dict["local"]["model"]
        model.eval()
        device = self.system_dict["local"]["device"]
        losses = []
        with torch.inference_mode():
            for data in self.system_dict["local"]["test_generator"]:
                images = data['img'].to(device=device, dtype=torch.float32)
                annotations = data['annot'].to(device=device, dtype=torch.float32)
                cls_loss, reg_loss = model([images, annotations])
                loss = cls_loss.mean() + reg_loss.mean()
                if torch.isfinite(loss):
                    losses.append(loss.item())
        if not losses:
            raise RuntimeError("Validation produced no finite losses")
        return float(np.mean(losses))

    def _export_onnx(self):
        model = self.system_dict["local"]["model"]
        model = model.module if isinstance(model, nn.DataParallel) else model
        image_size = self.system_dict["params"]["image_size"]
        dummy_input = torch.rand(1, 3, image_size, image_size, device=self.system_dict["local"]["device"])
        model.backbone_net.model.set_swish(memory_efficient=False)
        try:
            torch.onnx.export(
                model,
                dummy_input,
                os.path.join(self.system_dict["output"]["saved_path"], "signatrix_efficientdet_coco.onnx"),
                verbose=False,
                opset_version=11,
            )
        except Exception as exc:
            print(f"ONNX export skipped: {exc}")
        finally:
            model.backbone_net.model.set_swish(memory_efficient=True)
