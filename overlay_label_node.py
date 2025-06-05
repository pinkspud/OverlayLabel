import torch
import numpy as np
from PIL import Image

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

        # Just pass through for now to test connections
        return (generated_image,)
