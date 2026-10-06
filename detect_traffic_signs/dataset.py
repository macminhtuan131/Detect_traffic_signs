import os
import torch
import numpy as np
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import cv2
import json


class CocoDataset(Dataset):
    def __init__(self, root_dir, img_dir="images", set_dir='train2017', transform=None,
                 include_empty=True, include_unused_categories=False, category_ids=None,
                 annotation_file=None, image_dir=None, correction_file=None):

        self.root_dir = root_dir
        self.img_dir = img_dir
        self.set_name = set_dir
        self.transform = transform

        annotation_dir = os.path.join(self.root_dir, 'annotations')
        annotation_file = annotation_file or os.path.join(
            annotation_dir, 'instances_' + self.set_name + '.json'
        )
        self.image_dir = image_dir or os.path.join(self.root_dir, self.img_dir, self.set_name)
        self.coco = COCO(annotation_file)
        self.verified_empty_image_ids = set()
        correction_path = correction_file or os.path.join(
            annotation_dir, 'instances_' + self.set_name + '_corrections.json'
        )
        if os.path.isfile(correction_path):
            with open(correction_path, 'r', encoding='utf-8') as correction_stream:
                corrections = json.load(correction_stream)
            existing_annotation_ids = {
                annotation['id'] for annotation in self.coco.dataset.get('annotations', [])
            }
            added_annotations = corrections.get('annotations', [])
            added_ids = [annotation['id'] for annotation in added_annotations]
            if len(added_ids) != len(set(added_ids)):
                raise ValueError(f"Correction file contains duplicate annotation IDs: {correction_path}")
            duplicate_ids = existing_annotation_ids.intersection(
                annotation['id'] for annotation in added_annotations
            )
            if duplicate_ids:
                raise ValueError(f"Duplicate correction annotation IDs: {sorted(duplicate_ids)}")
            image_by_id = {
                image['id']: image for image in self.coco.dataset.get('images', [])
            }
            valid_category_ids = {
                category['id'] for category in self.coco.dataset.get('categories', [])
            }
            for annotation in added_annotations:
                image_id = annotation.get('image_id')
                category_id = annotation.get('category_id')
                bbox = annotation.get('bbox', [])
                if image_id not in image_by_id:
                    raise ValueError(f"Correction uses unknown image ID {image_id}")
                if category_id not in valid_category_ids:
                    raise ValueError(f"Correction uses unknown category ID {category_id}")
                if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
                    raise ValueError(f"Invalid correction bbox for annotation {annotation['id']}: {bbox}")
                image = image_by_id[image_id]
                x, y, width, height = bbox
                if (x < 0 or y < 0 or x + width > image['width']
                        or y + height > image['height']):
                    raise ValueError(
                        f"Correction bbox exceeds image bounds for annotation {annotation['id']}: {bbox}"
                    )
            self.coco.dataset.setdefault('annotations', []).extend(added_annotations)
            verified_empty_ids = set(corrections.get('verified_empty_image_ids', []))
            unknown_empty_ids = verified_empty_ids.difference(image_by_id)
            if unknown_empty_ids:
                raise ValueError(f"Unknown verified-empty image IDs: {sorted(unknown_empty_ids)}")
            annotated_image_ids = {
                annotation['image_id'] for annotation in self.coco.dataset['annotations']
            }
            contradictory_ids = verified_empty_ids.intersection(annotated_image_ids)
            if contradictory_ids:
                raise ValueError(
                    f"Images cannot be both annotated and verified empty: {sorted(contradictory_ids)}"
                )
            self.verified_empty_image_ids.update(verified_empty_ids)
            self.coco.createIndex()
        self.image_ids = self.coco.getImgIds()
        if not include_empty:
            self.image_ids = [
                image_id for image_id in self.image_ids
                if self.coco.getAnnIds(imgIds=image_id, iscrowd=False)
                or image_id in self.verified_empty_image_ids
            ]

        self.load_classes(include_unused_categories, category_ids)

    def load_classes(self, include_unused_categories=False, category_ids=None):

        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])
        if category_ids is not None:
            category_ids = list(category_ids)
            category_by_id = {category['id']: category for category in categories}
            missing = set(category_ids).difference(category_by_id)
            if missing:
                raise ValueError(f"Unknown category IDs in shared class mapping: {sorted(missing)}")
            categories = [category_by_id[category_id] for category_id in category_ids]
        elif not include_unused_categories:
            used_category_ids = {
                annotation['category_id'] for annotation in self.coco.dataset.get('annotations', [])
            }
            categories = [category for category in categories if category['id'] in used_category_ids]

        mapped_ids = {category['id'] for category in categories}
        annotated_ids = {
            annotation['category_id'] for annotation in self.coco.dataset.get('annotations', [])
        }
        unmapped_ids = annotated_ids.difference(mapped_ids)
        if unmapped_ids:
            raise ValueError(
                f"Annotations use categories missing from the class mapping: {sorted(unmapped_ids)}"
            )

        self.classes = {}
        self.coco_labels = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.image_dir, image_info['file_name'])
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not read dataset image: {os.path.abspath(path)}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # if len(img.shape) == 2:
        #     img = skimage.color.gray2rgb(img)

        return img.astype(np.float32) / 255.

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5), dtype=np.float32)

        # some images appear to miss annotations
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['area'] == 0:
                continue

            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5), dtype=np.float32)
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = self.coco_label_to_label(a['category_id'])
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]

    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def num_classes(self):
        return len(self.classes)


def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]

    imgs = torch.stack(imgs, dim=0) if torch.is_tensor(imgs[0]) else torch.from_numpy(np.stack(imgs, axis=0))

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    imgs = imgs.permute(0, 3, 1, 2)

    return {'img': imgs, 'annot': annot_padded, 'scale': scales}


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, common_size=512):
        self.common_size = common_size

    def __call__(self, sample):
        common_size = self.common_size
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        if height > width:
            scale = common_size / height
            resized_height = common_size
            resized_width = int(width * scale)
        else:
            scale = common_size / width
            resized_height = int(height * scale)
            resized_width = common_size

        image = cv2.resize(image, (resized_width, resized_height))

        new_image = np.zeros((common_size, common_size, 3), dtype=np.float32)
        new_image[0:resized_height, 0:resized_width] = image

        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': scale}


class Augmenter(object):
    """Apply traffic-sign-safe photometric augmentation."""

    def __init__(self, flip_x=0.0, brightness=0.12, contrast=0.15):
        self.flip_x = flip_x
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        if self.contrast > 0:
            contrast_factor = np.random.uniform(1.0 - self.contrast, 1.0 + self.contrast)
            image = image * contrast_factor
        if self.brightness > 0:
            brightness_delta = np.random.uniform(-self.brightness, self.brightness)
            image = image + brightness_delta
        image = np.clip(image, 0.0, 1.0).astype(np.float32, copy=False)

        # Disabled by default: mirroring directional signs or numerals creates
        # invalid examples unless their labels and artwork are also remapped.
        if np.random.rand() < self.flip_x:
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

        return {'img': image, 'annot': annots}


class Normalizer(object):

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]], dtype=np.float32)
        self.std = np.array([[[0.229, 0.224, 0.225]]], dtype=np.float32)

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}
