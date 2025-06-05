import numpy as np
from PIL import Image

class OverlayLabelNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generated_image": ("IMAGE",),
                "label_image": ("IMAGE",),
                "label_mask": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("blended_image",)
    FUNCTION = "blend_label"
    CATEGORY = "Custom/Image"

    def blend_label(self, generated_image, label_image, label_mask):
        gen = generated_image.convert("RGBA")
        label = label_image.convert("RGBA")
        mask = label_mask.convert("L")

        label = label.resize(gen.size)
        mask = mask.resize(gen.size)

        gen_np = np.array(gen).astype(np.float32)
        label_np = np.array(label).astype(np.float32)
        mask_np = np.array(mask).astype(np.float32) / 255.0

        blended_np = label_np * mask_np[..., None] + gen_np * (1.0 - mask_np[..., None])
        blended_img = Image.fromarray(np.clip(blended_np, 0, 255).astype(np.uint8))

        return (blended_img,)

NODE_CLASS_MAPPINGS = {
    "OverlayLabelNode": OverlayLabelNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OverlayLabelNode": "🧩 Overlay Label (Mask)"
}
