import argparse
import glob
import json
import os
import random
import time
import warnings

import cv2
import numpy as np
import torch


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ANNOTATIONS = os.path.join(PROJECT_DIR, "Fruit", "annotations", "instances_Images.json")


class Infer:
    def __init__(self, verbose=1, confidence_threshold=0.60, nms_threshold=0.50,
                 cross_class_nms_threshold=0.50, max_detections=100, use_amp=True,
                 image_size=512, cpu_threads=4):
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if not 0.0 <= nms_threshold <= 1.0:
            raise ValueError("nms_threshold must be between 0 and 1")
        if cross_class_nms_threshold is not None and not 0.0 <= cross_class_nms_threshold <= 1.0:
            raise ValueError("cross_class_nms_threshold must be between 0 and 1, or None")
        if max_detections <= 0:
            raise ValueError("max_detections must be positive")
        if image_size <= 0 or image_size % 128 != 0:
            raise ValueError("image_size must be a positive multiple of 128")
        if cpu_threads is not None and cpu_threads <= 0:
            raise ValueError("cpu_threads must be positive, or None")

        self.system_dict = {"verbose": verbose, "local": {}}
        local = self.system_dict["local"]
        local["common_size"] = image_size
        local["mean"] = np.array([[[0.485, 0.456, 0.406]]], dtype=np.float32)
        local["std"] = np.array([[[0.229, 0.224, 0.225]]], dtype=np.float32)
        local["confidence_threshold"] = confidence_threshold
        local["nms_threshold"] = nms_threshold
        local["cross_class_nms_threshold"] = cross_class_nms_threshold
        local["max_detections"] = max_detections
        local["use_amp"] = use_amp
        local["cpu_threads"] = cpu_threads

    @staticmethod
    def _resolve_path(path):
        if os.path.isabs(path):
            return path
        working_directory_path = os.path.abspath(path)
        if os.path.exists(working_directory_path):
            return working_directory_path
        return os.path.join(PROJECT_DIR, path)

    @staticmethod
    def _load_class_names(annotation_file, num_classes):
        if not annotation_file or not os.path.isfile(annotation_file):
            return [f"Class_{index}" for index in range(num_classes)]

        with open(annotation_file, "r", encoding="utf-8") as annotation_stream:
            dataset = json.load(annotation_stream)
        categories = sorted(dataset.get("categories", []), key=lambda category: category["id"])
        used_ids = {annotation["category_id"] for annotation in dataset.get("annotations", [])}
        active_categories = [category for category in categories if category["id"] in used_ids]

        # Legacy checkpoints include the unused `A` background output; newly
        # trained models correctly omit categories that have no annotations.
        if len(categories) == num_classes:
            selected_categories = categories
        elif len(active_categories) == num_classes:
            selected_categories = active_categories
        else:
            warnings.warn(
                f"Annotation metadata has {len(categories)} total/{len(active_categories)} active "
                f"classes, but the model has {num_classes}; using generic class names.",
                stacklevel=2,
            )
            return [f"Class_{index}" for index in range(num_classes)]
        return [category["name"] for category in selected_categories]

    def Model(self, model_dir="trained_weight", annotation_file=DEFAULT_ANNOTATIONS):
        model_dir = self._resolve_path(model_dir)
        model_path = model_dir if os.path.isfile(model_dir) else os.path.join(
            model_dir, "signatrix_efficientdet_coco.pth"
        )
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cpu" and self.system_dict["local"]["cpu_threads"]:
            torch.set_num_threads(self.system_dict["local"]["cpu_threads"])
        loaded_model = torch.load(model_path, map_location=device, weights_only=False)
        model = loaded_model.module if hasattr(loaded_model, "module") else loaded_model
        model = model.to(device).eval()

        local = self.system_dict["local"]
        local["device"] = device
        local["model"] = model
        model.inference_score_threshold = local["confidence_threshold"]
        model.inference_nms_threshold = local["nms_threshold"]
        model.inference_cross_class_nms_threshold = local["cross_class_nms_threshold"]
        model.inference_max_detections = local["max_detections"]

        annotation_file = self._resolve_path(annotation_file) if annotation_file else None
        checkpoint_class_names = getattr(model, "class_names", None)
        if checkpoint_class_names and len(checkpoint_class_names) == model.num_classes:
            local["class_names"] = list(checkpoint_class_names)
        else:
            local["class_names"] = self._load_class_names(annotation_file, model.num_classes)
        model.inference_ignored_class_indices = [
            index for index, name in enumerate(local["class_names"]) if name.strip().lower() == "a"
        ]
        return self

    def Predict(self, input_image, confidence_threshold=None):
        if input_image is None:
            raise ValueError("input_image must be a valid OpenCV image")
        local = self.system_dict["local"]
        if "model" not in local:
            raise RuntimeError("Load a model with Model() before calling Predict()")
        if confidence_threshold is not None:
            if not 0.0 <= confidence_threshold <= 1.0:
                raise ValueError("confidence_threshold must be between 0 and 1")
            local["confidence_threshold"] = confidence_threshold
            local["model"].inference_score_threshold = confidence_threshold

        image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = (image - local["mean"]) / local["std"]
        height, width = image.shape[:2]
        scale = local["common_size"] / max(height, width)
        resized_height = max(1, int(height * scale))
        resized_width = max(1, int(width * scale))
        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        padded_image = np.zeros(
            (local["common_size"], local["common_size"], 3), dtype=np.float32
        )
        padded_image[:resized_height, :resized_width] = image
        model_input = torch.from_numpy(padded_image).permute(2, 0, 1).unsqueeze(0)
        model_input = model_input.to(local["device"], non_blocking=local["device"].type == "cuda")

        use_amp = local["use_amp"] and local["device"].type == "cuda"
        if local["device"].type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode(), torch.autocast(
            device_type=local["device"].type, dtype=torch.float16, enabled=use_amp
        ):
            scores, labels, boxes = local["model"](model_input)
        if local["device"].type == "cuda":
            torch.cuda.synchronize()
        duration = time.perf_counter() - start

        boxes = boxes.float()
        boxes = torch.stack((
            boxes[:, 0].clamp(min=0, max=resized_width),
            boxes[:, 1].clamp(min=0, max=resized_height),
            boxes[:, 2].clamp(min=0, max=resized_width),
            boxes[:, 3].clamp(min=0, max=resized_height),
        ), dim=1)
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[valid].div(scale).cpu()
        scores = scores[valid].float().cpu()
        labels = labels[valid].cpu()
        return duration, scores, labels, boxes

    def class_name(self, label):
        class_names = self.system_dict["local"].get("class_names", [])
        return class_names[label] if 0 <= label < len(class_names) else f"Unknown_{label}"


def draw_detections(image, scores, labels, boxes, detector):
    result = image.copy()
    image_height, image_width = result.shape[:2]
    for score, label, box in zip(scores.tolist(), labels.tolist(), boxes.tolist()):
        x1, y1, x2, y2 = box
        x1 = int(np.clip(round(x1), 0, image_width - 1))
        y1 = int(np.clip(round(y1), 0, image_height - 1))
        x2 = int(np.clip(round(x2), 0, image_width - 1))
        y2 = int(np.clip(round(y2), 0, image_height - 1))
        color = ((37 * label + 60) % 255, (17 * label + 150) % 255, (97 * label + 220) % 255)
        color = tuple(int(channel) for channel in color)
        caption = f"{detector.class_name(label)} {score:.2f}"

        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        (text_width, text_height), baseline = cv2.getTextSize(
            caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        text_left = max(0, min(x1, image_width - text_width - 4))
        text_top = max(0, y1 - text_height - baseline - 4)
        cv2.rectangle(
            result,
            (text_left, text_top),
            (min(image_width - 1, text_left + text_width + 4), y1),
            color,
            thickness=-1,
        )
        cv2.putText(
            result,
            caption,
            (text_left + 2, max(text_height, y1 - baseline - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return result


def _parse_args():
    parser = argparse.ArgumentParser(description="Detect traffic signs in an image")
    parser.add_argument("--input", help="Image to process; defaults to a random image from test/")
    parser.add_argument("--model-dir", default="trained_weight", help="Checkpoint file or directory")
    parser.add_argument("--annotations", default=DEFAULT_ANNOTATIONS, help="COCO JSON used for class names")
    parser.add_argument("--output", default=os.path.join(PROJECT_DIR, "detect_result.jpg"))
    parser.add_argument("--confidence", type=float, default=0.60)
    parser.add_argument("--nms-iou", type=float, default=0.50)
    parser.add_argument("--cross-class-nms-iou", type=float, default=0.50)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--show", action="store_true", help="Open the result in an OpenCV window")
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.input:
        image_path = os.path.abspath(args.input)
    else:
        test_images = []
        for extension in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            test_images.extend(glob.glob(os.path.join(PROJECT_DIR, "test", extension)))
        if not test_images:
            raise FileNotFoundError(f"No test images found in {os.path.join(PROJECT_DIR, 'test')}")
        image_path = random.choice(test_images)

    input_image = cv2.imread(image_path)
    if input_image is None:
        raise FileNotFoundError(f"Could not read input image: {image_path}")

    cross_class_nms = args.cross_class_nms_iou if args.cross_class_nms_iou >= 0 else None
    detector = Infer(
        confidence_threshold=args.confidence,
        nms_threshold=args.nms_iou,
        cross_class_nms_threshold=cross_class_nms,
        max_detections=args.max_detections,
        image_size=args.image_size,
        cpu_threads=args.cpu_threads,
    ).Model(args.model_dir, annotation_file=args.annotations)

    total_start = time.perf_counter()
    duration, scores, labels, boxes = detector.Predict(input_image)
    result = draw_detections(input_image, scores, labels, boxes, detector)
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if not cv2.imwrite(output_path, result):
        raise OSError(f"Could not save result image: {output_path}")

    print(f"Input: {image_path}")
    print(f"Detections: {len(scores)}")
    for score, label in zip(scores.tolist(), labels.tolist()):
        print(f"  {detector.class_name(label)}: {score:.3f}")
    print(f"Model inference: {duration * 1000:.1f} ms")
    print(f"End-to-end: {(time.perf_counter() - total_start) * 1000:.1f} ms")
    print(f"Saved result to: {output_path}")

    if args.show:
        cv2.imshow("Traffic sign detections", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
