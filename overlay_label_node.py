import torch
import numpy as np
from PIL import Image, ImageOps


class OverlayLabelNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generated_image": ("IMAGE",),
                "label_image": ("IMAGE",),
                "label_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("blended_image",)
    FUNCTION = "blend_label"
    CATEGORY = "image"

    def tensor_to_pil(self, tensor):
        # tensor: [1, H, W, C] → squeeze batch → transpose channels
        array = tensor.squeeze(0).cpu().numpy()  # [H, W, C]
        array = np.clip(array * 255.0, 0, 255).astype(np.uint8)

        if array.ndim == 2:
            return Image.fromarray(array, mode="L")
        elif array.shape[2] == 3:
            return Image.fromarray(array, mode="RGB")
        elif array.shape[2] == 4:
            return Image.fromarray(array, mode="RGBA")
        else:
            raise ValueError(f"Unsupported array shape for PIL: {array.shape}")


    def pil_to_tensor(self, image):
        array = np.array(image).astype(np.float32) / 255.0
        if array.ndim == 2:
            array = array[None, :, :]
        else:
            array = np.transpose(array, (2, 0, 1))
        return torch.from_numpy(array)

    def blend_label(self, generated_image, label_image, label_mask):
        print("generated_image shape:", generated_image.shape)
        print("label_image shape:", label_image.shape)
        print("label_mask shape:", label_mask.shape)

        # Convert inputs to PIL
        gen_img = self.tensor_to_pil(generated_image).convert("RGBA")
        label_img = self.tensor_to_pil(label_image).convert("RGBA")

        # Get mask bounding box (where label should go)
        mask_np = label_mask.squeeze(0).cpu().numpy()
        mask_img = Image.fromarray((mask_np * 255).astype(np.uint8), mode="L")
        bbox = mask_img.getbbox()
        if bbox is None:
            print("Empty mask, skipping.")
            return (generated_image,)

        print("Target bounding box from mask:", bbox)

        # Get alpha channel and crop to content area (non-transparent label)
        label_alpha = label_img.split()[-1]
        content_bbox = label_alpha.getbbox()
        if content_bbox is None:
            print("Label image is fully transparent.")
            return (generated_image,)

        print("Visible content bbox in label:", content_bbox)

        # Crop label to visible content
        label_cropped = label_img.crop(content_bbox)
        alpha_cropped = label_alpha.crop(content_bbox)

        # Resize both to match mask region
        target_width = bbox[2] - bbox[0]
        target_height = bbox[3] - bbox[1]
        label_resized = label_cropped.resize((target_width, target_height), Image.LANCZOS)
        alpha_resized = alpha_cropped.resize((target_width, target_height), Image.LANCZOS)

        # Overlay onto the generated image
        overlay = Image.new("RGBA", gen_img.size)
        overlay.paste(label_resized, (bbox[0], bbox[1]), mask=alpha_resized)
        result = Image.alpha_composite(gen_img, overlay)

        # Convert back to ComfyUI format
        result_tensor = self.pil_to_tensor(result)
        result_tensor = result_tensor.permute(1, 2, 0).unsqueeze(0)  # [C,H,W] → [1,H,W,C]
        return (result_tensor,)
