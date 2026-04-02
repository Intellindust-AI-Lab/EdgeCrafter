"""
EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation
Copyright (c) 2026 The EdgeCrafter Authors. All Rights Reserved.
"""

import concurrent.futures
import os
import queue
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from engine.core import YAMLConfig


@dataclass
class Result:
    label: int
    score: float
    keypoints: np.ndarray


# COCO keypoint skeleton (1-based in standard definition)
COCO_SKELETON = [
    (16, 14), (14, 12), (17, 15), (15, 13), (12, 13),
    (6, 12), (7, 13), (6, 7), (6, 8), (7, 9),
    (8, 10), (9, 11), (2, 3), (1, 2), (1, 3),
    (2, 4), (3, 5), (4, 6), (5, 7),
]
COCO_SKELETON = [(a - 1, b - 1) for a, b in COCO_SKELETON]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _decode_keypoints(keypoints: np.ndarray):
    """Decode keypoints into xy and visibility arrays.

    Supports: [K,2], [K,3], [2K], [3K].
    """
    kpts = np.asarray(keypoints)

    if kpts.ndim == 2:
        if kpts.shape[1] < 2:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        xy = kpts[:, :2].astype(np.float32)
        vis = kpts[:, 2].astype(np.float32) if kpts.shape[1] >= 3 else np.ones((kpts.shape[0],), dtype=np.float32)
        return xy, vis

    if kpts.ndim == 1:
        if kpts.size % 3 == 0:
            kpts_ = kpts.reshape(-1, 3)
            return kpts_[:, :2].astype(np.float32), kpts_[:, 2].astype(np.float32)
        if kpts.size % 2 == 0:
            kpts_ = kpts.reshape(-1, 2)
            return kpts_.astype(np.float32), np.ones((kpts_.shape[0],), dtype=np.float32)

    return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)


def get_draw_params(image_shape: tuple[int, int, int]):
    height, width = image_shape[:2]
    min_side = max(1, min(height, width))
    base = min_side / 640.0
    font_scale = max(0.9, 0.95 * base)
    text_thickness = max(2, int(round(1.6 * base)))
    point_radius = max(2, int(round(3.0 * base)))
    skeleton_thickness = max(2, int(round(2.0 * base)))
    return font_scale, text_thickness, point_radius, skeleton_thickness


def draw_to_numpy(image: Image.Image, results: list[Result], draw_skeleton: bool = True):
    im_np = np.array(image, copy=True)
    font_scale, text_thickness, point_radius, line_thickness = get_draw_params(im_np.shape)

    for res in results:
        kpts, vis = _decode_keypoints(res.keypoints)
        if kpts.shape[0] == 0:
            continue

        kpts = kpts.astype(np.int32)
        vis = np.isfinite(vis) & (vis > 0)

        for idx, (x, y) in enumerate(kpts):
            if vis[idx]:
                cv2.circle(im_np, (int(x), int(y)), point_radius, (0, 255, 0), -1)

        if draw_skeleton:
            for a, b in COCO_SKELETON:
                if a < len(kpts) and b < len(kpts) and vis[a] and vis[b]:
                    xa, ya = kpts[a]
                    xb, yb = kpts[b]
                    cv2.line(im_np, (int(xa), int(ya)), (int(xb), int(yb)), (255, 128, 0), line_thickness)

        min_xy = np.min(kpts[vis], axis=0) if np.any(vis) else np.min(kpts, axis=0)
        min_xy = np.maximum(min_xy, 0)
        text = f"{res.label} {res.score:.2f}"
        cv2.putText(
            im_np,
            text,
            (int(min_xy[0]), int(max(min_xy[1] - 5, 10))),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )

    return im_np


def draw_pose(image: Image.Image, results: list[Result], draw_skeleton: bool = True):
    return Image.fromarray(draw_to_numpy(image, results, draw_skeleton=draw_skeleton).astype(np.uint8))


def _parse_pose_outputs(outputs):
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 3:
        raise RuntimeError(f"Unexpected pose model outputs. Expected (scores, labels, keypoints), got type={type(outputs)}")
    scores, labels, keypoints = outputs
    return scores, labels, keypoints


class ECPoseInferencer:
    def __init__(self, model, device, size, thresh):
        self.model = model
        self.device = device
        self.size = size
        self.thresh = thresh
        self.transforms = T.Compose([
            T.Resize(self.size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def infer(self, image: Image.Image):
        orig_sizes = torch.tensor([[image.size[0], image.size[1]]], device=self.device, dtype=torch.int64)
        tensor = self.transforms(image).unsqueeze(0).to(self.device)
        outputs = self.model(tensor, orig_sizes)
        scores, labels, keypoints = _parse_pose_outputs(outputs)

        keep = scores[0] > self.thresh
        scs = scores[0][keep]
        lbs = labels[0][keep]
        kps = keypoints[0][keep]

        results = []
        for j in range(len(scs)):
            results.append(
                Result(
                    label=int(lbs[j].item()),
                    score=float(scs[j].item()),
                    keypoints=kps[j].detach().cpu().numpy(),
                )
            )
        return results


class VideoReader(threading.Thread):
    def __init__(self, cap, queue_size=32):
        super().__init__()
        self.cap = cap
        self.q = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.daemon = True

    def run(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.q.put(None)
                break
            self.q.put(frame)

    def read(self):
        return self.q.get()

    def stop(self):
        self.stopped = True
        while not self.q.empty():
            self.q.get()


def process_image(inferencer: ECPoseInferencer, path: Path, draw_skeleton: bool = True):
    image = Image.open(path).convert("RGB")
    results = inferencer.infer(image)
    image = draw_pose(image, results, draw_skeleton=draw_skeleton)

    output_path = path.with_stem(f"{path.stem}_torch_pose_inference")
    image.save(output_path, quality=95, subsampling=0)
    print(f"Saved result to {output_path}")
    print(f"Detected {len(results)} poses")


def process_image_dir(inferencer: ECPoseInferencer, dir_path: Path, draw_skeleton: bool = True):
    image_paths = sorted([p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    if not image_paths:
        raise ValueError(f"No image files found in directory: {dir_path}")

    print(f"Found {len(image_paths)} images in {dir_path}")
    for idx, img_path in enumerate(image_paths, start=1):
        print(f"[{idx}/{len(image_paths)}] Processing {img_path.name}")
        process_image(inferencer, img_path, draw_skeleton=draw_skeleton)


def process_video(inferencer: ECPoseInferencer, path: Path, num_workers: int = 4, draw_skeleton: bool = True):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = path.with_stem(f"{path.stem}_torch_pose_inference")
    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to create output video writer: {output_path}")

    reader = VideoReader(cap)
    reader.start()

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
    futures_queue = queue.Queue(maxsize=max(32, num_workers * 4))

    def process_and_draw(pil_img: Image.Image, results: list[Result]):
        res_np = draw_to_numpy(pil_img, results, draw_skeleton=draw_skeleton)
        return cv2.cvtColor(res_np, cv2.COLOR_RGB2BGR)

    frame_count = 0
    writer_error: dict[str, Exception | None] = {"exc": None}

    def writer_worker():
        nonlocal frame_count
        try:
            while True:
                future = futures_queue.get()
                if future is None:
                    break
                frame_out = future.result()
                out.write(frame_out)
                frame_count += 1
                if frame_count % 10 == 0:
                    print(f"Processed {frame_count}/{total_frame} frames")
        except Exception as exc:  # noqa: BLE001
            writer_error["exc"] = exc

    writer_thread = threading.Thread(target=writer_worker, daemon=True)
    writer_thread.start()

    try:
        while True:
            if writer_error["exc"] is not None:
                raise writer_error["exc"]

            frame = reader.read()
            if frame is None:
                break

            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            results = inferencer.infer(pil_img)
            future = executor.submit(process_and_draw, pil_img, results)
            futures_queue.put(future)
    finally:
        reader.stop()
        futures_queue.put(None)
        writer_thread.join()
        executor.shutdown()
        cap.release()
        out.release()

    if writer_error["exc"] is not None:
        raise writer_error["exc"]

    print(f"Saved video result to {output_path}")


def build_model(config_path: str, resume_path: str, device: torch.device):
    cfg = YAMLConfig(config_path, resume=resume_path)

    if "ViTAdapter" in cfg.yaml_cfg:
        cfg.yaml_cfg["ViTAdapter"]["skip_load_backbone"] = True

    checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_target_sizes)

    model = Model().to(device)
    model.eval()
    return model, tuple(cfg.yaml_cfg["eval_spatial_size"]), cfg.yaml_cfg.get("task", "")


def main(args):
    device = torch.device(args.device)
    model, img_size, task = build_model(args.config, args.resume, device)
    if task != "pose":
        print(f"Warning: config task is '{task}', script is specialized for pose inference.")

    inferencer = ECPoseInferencer(
        model=model,
        device=device,
        size=img_size,
        thresh=args.thresh,
    )

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_dir():
        process_image_dir(
            inferencer,
            input_path,
            draw_skeleton=not args.no_skeleton,
        )
    elif input_path.suffix.lower() in IMAGE_EXTS:
        process_image(
            inferencer,
            input_path,
            draw_skeleton=not args.no_skeleton,
        )
    else:
        process_video(
            inferencer,
            input_path,
            num_workers=args.num_workers,
            draw_skeleton=not args.no_skeleton,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ECPose Torch Inference")
    parser.add_argument("-c", "--config", required=True, type=str)
    parser.add_argument("-r", "--resume", required=True, type=str)
    parser.add_argument("-i", "--input", required=True, type=str, help="Image path, image directory path, or video path")
    parser.add_argument("-d", "--device", default="cuda:0", type=str)
    parser.add_argument("-t", "--thresh", default=0.4, type=float)
    parser.add_argument("--num-workers", type=int, default=2, help="Thread workers for video draw/write")
    parser.add_argument("--no-skeleton", action="store_true", help="Draw keypoints only, no skeleton links")
    main(parser.parse_args())
