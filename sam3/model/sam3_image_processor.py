# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe
from typing import Dict, List

import numpy as np
import PIL
import torch
from sam3.model import box_ops
from sam3.model.data_misc import FindStage, interpolate
from torchvision.transforms import v2

# For batched inference
from sam3.train.data.sam3_image_dataset import InferenceMetadata, FindQueryLoaded, Image as SAMImage, Datapoint
from sam3.train.data.collator import collate_fn_api as collate
from sam3.train.transforms.basic_for_api import ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI
from sam3.eval.postprocessors import PostProcessImage
from sam3.model.utils.misc import copy_data_to_device


class Sam3Processor:
    """ """

    def __init__(self, model, resolution=1008, device="cuda", confidence_threshold=0.5):
        self.model = model
        self.resolution = resolution
        self.device = device
        self.transform = v2.Compose(
            [
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(resolution, resolution)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.confidence_threshold = confidence_threshold

        self.find_stage = FindStage(
            img_ids=torch.tensor([0], device=device, dtype=torch.long),
            text_ids=torch.tensor([0], device=device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )

    @torch.inference_mode()
    def set_image(self, image, state=None):
        """Sets the image on which we want to do predictions."""
        if state is None:
            state = {}

        if isinstance(image, PIL.Image.Image):
            width, height = image.size
        elif isinstance(image, (torch.Tensor, np.ndarray)):
            height, width = image.shape[-2:]
        else:
            raise ValueError("Image must be a PIL image or a tensor")

        image = v2.functional.to_image(image).to(self.device)
        image = self.transform(image).unsqueeze(0)

        state["original_height"] = height
        state["original_width"] = width
        state["backbone_out"] = self.model.backbone.forward_image(image)
        inst_interactivity_en = self.model.inst_interactive_predictor is not None
        if inst_interactivity_en and "sam2_backbone_out" in state["backbone_out"]:
            sam2_backbone_out = state["backbone_out"]["sam2_backbone_out"]
            sam2_backbone_out["backbone_fpn"][0] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(
                    sam2_backbone_out["backbone_fpn"][0]
                )
            )
            sam2_backbone_out["backbone_fpn"][1] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(
                    sam2_backbone_out["backbone_fpn"][1]
                )
            )
        return state

    @torch.inference_mode()
    def set_image_batch(self, images: List[np.ndarray], state=None):
        """Sets the image batch on which we want to do predictions."""
        if state is None:
            state = {}

        if not isinstance(images, list):
            raise ValueError("Images must be a list of PIL images or tensors")
        assert len(images) > 0, "Images list must not be empty"
        assert isinstance(images[0], PIL.Image.Image), (
            "Images must be a list of PIL images"
        )

        state["original_heights"] = [image.height for image in images]
        state["original_widths"] = [image.width for image in images]

        images = [
            self.transform(v2.functional.to_image(image).to(self.device))
            for image in images
        ]
        images = torch.stack(images, dim=0)
        state["backbone_out"] = self.model.backbone.forward_image(images)
        inst_interactivity_en = self.model.inst_interactive_predictor is not None
        if inst_interactivity_en and "sam2_backbone_out" in state["backbone_out"]:
            sam2_backbone_out = state["backbone_out"]["sam2_backbone_out"]
            sam2_backbone_out["backbone_fpn"][0] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(
                    sam2_backbone_out["backbone_fpn"][0]
                )
            )
            sam2_backbone_out["backbone_fpn"][1] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(
                    sam2_backbone_out["backbone_fpn"][1]
                )
            )
        return state

    @torch.inference_mode()
    def set_text_prompt(self, prompt: str, state: Dict):
        """Sets the text prompt and run the inference"""

        if "backbone_out" not in state:
            raise ValueError("You must call set_image before set_text_prompt")

        text_outputs = self.model.backbone.forward_text([prompt], device=self.device)
        # will erase the previous text prompt if any
        state["backbone_out"].update(text_outputs)
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()

        return self._forward_grounding(state)

    @torch.inference_mode()
    def set_text_prompts(self, prompts: List[str], state: Dict):
        """Segments multiple text prompts at once and returns results for each.

        Args:
            prompts: List of text prompts (e.g., ["wheel", "windshield", "headlight"])
            state: State from set_image()

        Returns:
            Dict mapping each prompt to its results:
            {
                "wheel": {"masks": [...], "boxes": [...], "scores": [...]},
                "windshield": {"masks": [...], "boxes": [...], "scores": [...]},
                ...
            }
        """
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before set_text_prompts")

        results = {}
        for prompt in prompts:
            self.reset_all_prompts(state)
            state = self.set_text_prompt(prompt=prompt, state=state)
            # Clone tensors so they don't get overwritten by next prompt
            results[prompt] = {
                "masks": state.get("masks", torch.tensor([])).clone(),
                "boxes": state.get("boxes", torch.tensor([])).clone(),
                "scores": state.get("scores", torch.tensor([])).clone(),
            }

        # Store combined results in state for convenience
        state["multi_prompt_results"] = results
        return results

    @torch.inference_mode()
    def set_text_prompts_batched(self, prompts: List[str], image: PIL.Image.Image):
        """Segments multiple text prompts in a SINGLE forward pass (faster).

        Unlike set_text_prompts() which loops through prompts sequentially,
        this method processes all prompts in one forward pass through the model.

        Args:
            prompts: List of text prompts (e.g., ["wheel", "windshield", "headlight"])
            image: PIL Image (required - doesn't use state, processes image fresh)

        Returns:
            Dict mapping each prompt to its results:
            {
                "wheel": {"masks": Tensor, "boxes": Tensor, "scores": Tensor},
                ...
            }
        """
        # Setup transforms and postprocessor
        transform = ComposeAPI(
            transforms=[
                RandomResizeAPI(sizes=self.resolution, max_size=self.resolution, square=True, consistent_transform=False),
                ToTensorAPI(),
                NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        postprocessor = PostProcessImage(
            max_dets_per_img=-1,
            iou_type="segm",
            use_original_sizes_box=True,
            use_original_sizes_mask=True,
            convert_mask_to_rle=False,
            detection_threshold=self.confidence_threshold,
            to_cpu=False,
        )

        # Create datapoint with image
        w, h = image.size
        datapoint = Datapoint(find_queries=[], images=[])
        datapoint.images = [SAMImage(data=image, objects=[], size=[h, w])]

        # Add all text prompts
        prompt_ids = {}
        for idx, prompt in enumerate(prompts):
            datapoint.find_queries.append(
                FindQueryLoaded(
                    query_text=prompt,
                    image_id=0,
                    object_ids_output=[],
                    is_exhaustive=True,
                    query_processing_order=0,
                    inference_metadata=InferenceMetadata(
                        coco_image_id=idx,
                        original_image_id=idx,
                        original_category_id=1,
                        original_size=[w, h],
                        object_id=0,
                        frame_index=0,
                    )
                )
            )
            prompt_ids[idx] = prompt

        # Transform and collate
        datapoint = transform(datapoint)
        batch = collate([datapoint], dict_key="dummy")["dummy"]
        batch = copy_data_to_device(batch, torch.device(self.device), non_blocking=True)

        # Single forward pass for ALL prompts
        output = self.model(batch)

        # Postprocess results
        processed_results = postprocessor.process_results(output, batch.find_metadatas)

        # Organize results by prompt name
        results = {}
        for idx, prompt in prompt_ids.items():
            if idx in processed_results:
                res = processed_results[idx]
                results[prompt] = {
                    "masks": res.get("masks", torch.tensor([])),
                    "boxes": res.get("boxes", torch.tensor([])),
                    "scores": res.get("scores", torch.tensor([])),
                }
            else:
                results[prompt] = {
                    "masks": torch.tensor([]),
                    "boxes": torch.tensor([]),
                    "scores": torch.tensor([]),
                }

        return results

    @torch.inference_mode()
    def set_point_prompt(self, state: Dict, point_xy, label: int = 1,
                          multimask_output: bool = True) -> Dict:
        """Run the interactive (point-prompted) predictor on the current image.

        ``set_image`` must have been called.  Forwards to the underlying
        ``SAM3InteractiveImagePredictor`` via ``model.predict_inst`` which
        re-uses the already-computed backbone features in ``state``.

        Args:
          state: state dict returned by ``set_image``.
          point_xy: ``(x, y)`` pixel coords for a foreground click (or an
            ``Nx2`` array for multiple points).
          label: ``1`` for foreground, ``0`` for background.  When
            ``point_xy`` is a single point this is broadcast.
          multimask_output: if True, returns 3 candidate masks; we keep
            the highest-IoU one.

        Returns:
          state dict with ``masks`` (1xHxW bool), ``masks_logits``
          (1xHxW float), ``boxes`` (Nx4 xyxy float), ``scores`` (length-N
          float) — same shape contract as ``set_text_prompt`` so callers
          can treat the result uniformly.
        """
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before set_point_prompt")
        if self.model.inst_interactive_predictor is None:
            raise RuntimeError(
                "This SAM3 model was loaded without an interactive predictor; "
                "point prompts are not available.")

        pts = np.asarray(point_xy, dtype=np.float32)
        if pts.ndim == 1:
            pts = pts[None, :]
        if pts.shape[-1] != 2:
            raise ValueError(f"point_xy must be (x,y) or Nx2, got {pts.shape}")
        labels = np.full((pts.shape[0],), int(label), dtype=np.int32)

        masks_np, ious_np, _ = self.model.predict_inst(
            state,
            point_coords=pts,
            point_labels=labels,
            multimask_output=multimask_output,
            return_logits=False,
            normalize_coords=True,
        )
        # masks_np: (C, H, W) float in [0,1] (already thresholded since
        # return_logits=False — values are 0/1).  ious_np: (C,) float.
        if masks_np.ndim == 2:
            masks_np = masks_np[None, ...]
            ious_np = np.asarray([float(ious_np)])

        # Pick the highest-IoU mask when multimask_output=True.
        best = int(np.argmax(ious_np)) if ious_np.size > 0 else 0
        best_mask = masks_np[best].astype(bool)
        best_score = float(ious_np[best]) if ious_np.size > 0 else 0.0

        # Compute xyxy bbox in pixel coords on the original image.
        H, W = best_mask.shape
        if best_mask.any():
            ys, xs = np.where(best_mask)
            x0, y0 = float(xs.min()), float(ys.min())
            x1, y1 = float(xs.max()), float(ys.max())
        else:
            x0 = y0 = x1 = y1 = 0.0

        device = self.device
        mask_t = torch.from_numpy(best_mask).to(device).unsqueeze(0)
        logits_t = torch.from_numpy(masks_np[best].astype(np.float32)).to(
            device).unsqueeze(0).unsqueeze(0)
        box_t = torch.tensor([[x0, y0, x1, y1]], device=device,
                              dtype=torch.float32)
        score_t = torch.tensor([best_score], device=device,
                                dtype=torch.float32)

        state["masks"] = mask_t
        state["masks_logits"] = logits_t
        state["boxes"] = box_t
        state["scores"] = score_t
        return state

    @torch.inference_mode()
    def add_geometric_prompt(self, box: List, label: bool, state: Dict):
        """Adds a box prompt and run the inference.
        The image needs to be set, but not necessarily the text prompt.
        The box is assumed to be in [center_x, center_y, width, height] format and normalized in [0, 1] range.
        The label is True for a positive box, False for a negative box.
        """
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before set_text_prompt")

        if "language_features" not in state["backbone_out"]:
            # Looks like we don't have a text prompt yet. This is allowed, but we need to set the text prompt to "visual" for the model to rely only on the geometric prompt
            dummy_text_outputs = self.model.backbone.forward_text(
                ["visual"], device=self.device
            )
            state["backbone_out"].update(dummy_text_outputs)

        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()

        # adding a batch and sequence dimension
        boxes = torch.tensor(box, device=self.device, dtype=torch.float32).view(1, 1, 4)
        labels = torch.tensor([label], device=self.device, dtype=torch.bool).view(1, 1)
        state["geometric_prompt"].append_boxes(boxes, labels)

        return self._forward_grounding(state)

    def reset_all_prompts(self, state: Dict):
        """Removes all the prompts and results"""
        if "backbone_out" in state:
            backbone_keys_to_del = [
                "language_features",
                "language_mask",
                "language_embeds",
            ]
            for key in backbone_keys_to_del:
                if key in state["backbone_out"]:
                    del state["backbone_out"][key]

        keys_to_del = ["geometric_prompt", "boxes", "masks", "masks_logits", "scores"]
        for key in keys_to_del:
            if key in state:
                del state[key]

    @torch.inference_mode()
    def set_confidence_threshold(self, threshold: float, state=None):
        """Sets the confidence threshold for the masks"""
        self.confidence_threshold = threshold
        if state is not None and "boxes" in state:
            # we need to filter the boxes again
            # In principle we could do this more efficiently since we would only need
            # to rerun the heads. But this is simpler and not too inefficient
            return self._forward_grounding(state)
        return state

    @torch.inference_mode()
    def _forward_grounding(self, state: Dict):
        outputs = self.model.forward_grounding(
            backbone_out=state["backbone_out"],
            find_input=self.find_stage,
            geometric_prompt=state["geometric_prompt"],
            find_target=None,
        )

        out_bbox = outputs["pred_boxes"]
        out_logits = outputs["pred_logits"]
        out_masks = outputs["pred_masks"]
        out_probs = out_logits.sigmoid()
        presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
        out_probs = (out_probs * presence_score).squeeze(-1)

        keep = out_probs > self.confidence_threshold
        out_probs = out_probs[keep]
        out_masks = out_masks[keep]
        out_bbox = out_bbox[keep]

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        img_h = state["original_height"]
        img_w = state["original_width"]
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h]).to(self.device)
        boxes = boxes * scale_fct[None, :]

        out_masks = interpolate(
            out_masks.unsqueeze(1),
            (img_h, img_w),
            mode="bilinear",
            align_corners=False,
        ).sigmoid()

        state["masks_logits"] = out_masks
        state["masks"] = out_masks > 0.5
        state["boxes"] = boxes
        state["scores"] = out_probs
        return state
