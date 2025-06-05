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
        # Ensure input is [C, H, W] torch tensor, values in [0, 1]
        array = tensor.cpu().numpy()
        array = np.clip(array * 255.0, 0, 255).astype(np.uint8)

        if array.shape[0] == 1:
            return Image.fromarray(array[0], mode="L")  # 2D grayscale
        elif array.shape[0] == 3:
            return Image.fromarray(np.transpose(array, (1, 2, 0)), mode="RGB")  # 3D RGB
        elif array.shape[0] == 4:
            return Image.fromarray(np.transpose(array, (1, 2, 0)), mode="RGBA")  # 3D RGBA
        else:
            raise ValueError(f"Unsupported tensor shape: {array.shape}")


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
