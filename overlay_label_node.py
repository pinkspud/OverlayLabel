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

        # Convert tensors to PIL images
        gen_img = self.tensor_to_pil(generated_image)
        label_img = self.tensor_to_pil(label_image)

        # Convert label mask to 2D NumPy array
        mask_np = label_mask.squeeze(0).cpu().numpy()  # [H, W]
        mask_img = Image.fromarray((mask_np * 255).astype(np.uint8), mode="L")
        print("Mask image shape:", mask_np.shape)
        print("Mask image size:", mask_img.size)
        # Get bounding box of the non-zero mask area
        bbox = mask_img.getbbox()
        if bbox is None:
            print("No non-zero area in mask.")
            return (generated_image,)  # just return original if mask is empty

        # Resize label image to match mask bounding box
    

        label_resized = ImageOps.fit(label_img, (bbox[2] - bbox[0], bbox[3] - bbox[1]), method=Image.LANCZOS)


        print("Bounding box:", bbox)
        print("Label resized size:", label_resized.size)

        # Prepare overlay canvas
        gen_rgba = gen_img.convert("RGBA")
        overlay = Image.new("RGBA", gen_rgba.size)

        # Paste resized label into the masked region
        overlay.paste(label_resized, (bbox[0], bbox[1]), mask=label_resized.split()[-1])  # use alpha as mask

        # Blend the overlay onto the generated image
        result = Image.alpha_composite(gen_rgba, overlay)

        # Convert back to tensor
        result_tensor = self.pil_to_tensor(result)

        result_tensor = result_tensor.permute(1, 2, 0).unsqueeze(0)  # [C,H,W] → [1,H,W,C]

        return (result_tensor,)
