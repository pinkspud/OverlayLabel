import numpy as np
from PIL import Image
import torch

class OverlayLabelNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generated_image": ("IMAGE",),
                "label_image": ("IMAGE",),
                "label_mask": ("MASK",),  # ComfyUI mask type (1, H, W)
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("blended_image",)
    FUNCTION = "blend_label"
    CATEGORY = "image"

    def tensor_to_pil(self, tensor):
        tensor = tensor.cpu().numpy()
        tensor = np.clip(tensor * 255.0, 0, 255).astype(np.uint8)
        if tensor.shape[0] == 1:
            return Image.fromarray(tensor[0], mode="L")
        elif tensor.shape[0] == 3:
            return Image.fromarray(np.transpose(tensor, (1, 2, 0)), mode="RGB")
        elif tensor.shape[0] == 4:
            return Image.fromarray(np.transpose(tensor, (1, 2, 0)), mode="RGBA")
        else:
            raise ValueError("Unsupported channel size")

    def pil_to_tensor(self, pil_image):
        image = np.array(pil_image).astype(np.float32) / 255.0
        if image.ndim == 2:
            image = image[None, :, :]
        else:
            image = np.transpose(image, (2, 0, 1))  # HWC → CHW
        return torch.from_numpy(image)

    def blend_label(self, generated_image, label_image, label_mask):
        # Convert to PIL images
        gen = self.tensor_to_pil(generated_image).convert("RGBA")
        label = self.tensor_to_pil(label_image).convert("RGBA")

        # Resize mask to match generated image
        H, W = gen.size
        mask_tensor = label_mask[0]  # [1, H, W] → [H, W]
        mask_np = mask_tensor.cpu().numpy()

        # label_mask is a torch tensor of shape [1, H, W]
        mask_np = label_mask.squeeze().cpu().numpy()  # shape → [H, W]

        # Convert to PIL-safe grayscale image
        mask_img_resized = Image.fromarray((mask_np * 255).astype(np.uint8), mode="L").resize(gen.size)

        # Back to NumPy float32 in [0, 1]
        mask_np_resized = np.array(mask_img_resized).astype(np.float32) / 255.0


        # Find bounding box of non-zero mask
        coords = np.argwhere(mask_np_resized > 0.05)
        if coords.shape[0] == 0:
            return (self.pil_to_tensor(gen),)  # Nothing to apply

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        box_w, box_h = x_max - x_min, y_max - y_min

        # Resize label to fit mask region
        label_resized = label.resize((box_w, box_h))
        label_np = np.array(label_resized).astype(np.float32)

        # Extract region from gen and mask
        mask_crop = mask_np_resized[y_min:y_max, x_min:x_max][..., None]
        gen_np = np.array(gen).astype(np.float32)
        gen_crop = gen_np[y_min:y_max, x_min:x_max]

        # Blend
        blended_crop = label_np * mask_crop + gen_crop * (1.0 - mask_crop)
        gen_np[y_min:y_max, x_min:x_max] = blended_crop

        # Convert back to tensor
        result_img = Image.fromarray(np.clip(gen_np, 0, 255).astype(np.uint8))
        return (self.pil_to_tensor(result_img),)


NODE_CLASS_MAPPINGS = {
    "OverlayLabelNode": OverlayLabelNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OverlayLabelNode": "🧩 Overlay Label (with Mask)"
}
