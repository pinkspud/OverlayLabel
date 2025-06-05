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

        # Convert to PIL images
        gen_img = self.tensor_to_pil(generated_image).convert("RGBA")
        label_img = self.tensor_to_pil(label_image).convert("RGBA")

        # Get 2D mask and bounding box on the generated image
        mask_np = label_mask.squeeze(0).cpu().numpy()  # [H, W]
        mask_img = Image.fromarray((mask_np * 255).astype(np.uint8), mode="L")
        bbox = mask_img.getbbox()
        if bbox is None:
            print("No mask region found.")
            return (generated_image,)

        print("Mask bbox:", bbox)

        # Resize label image and its alpha to the target region
        target_width = bbox[2] - bbox[0]
        target_height = bbox[3] - bbox[1]
        label_resized = label_img.resize((target_width, target_height), resample=Image.LANCZOS)

        # Extract alpha for transparency
        alpha_resized = label_resized.split()[-1]

        # Create overlay and paste resized label using its alpha
        overlay = Image.new("RGBA", gen_img.size)
        overlay.paste(label_resized, (bbox[0], bbox[1]), mask=alpha_resized)

        # Composite over the original image
        result = Image.alpha_composite(gen_img, overlay)

        # Convert back to tensor
        result_tensor = self.pil_to_tensor(result)
        result_tensor = result_tensor.permute(1, 2, 0).unsqueeze(0)  # [C,H,W] → [1,H,W,C]

        return (result_tensor,)

