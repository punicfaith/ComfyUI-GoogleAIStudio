import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

class BatchImageNormalizer:
    """
    Normalize batch images to the same size with dynamic input support.
    Supports resolution limiting and canvas expansion.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "inputcount": ("INT", {"default": 2, "min": 2, "max": 1000, "step": 1}),
                "resize_mode": (["max_resolution", "min_resolution", "first_image", "largest_image"], {"default": "largest_image"}),
                "resolution_value": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 64}),
                "upscale_method": (["bilinear", "bicubic", "nearest", "area", "lanczos"], {"default": "bicubic"}),
                "canvas_position": (["center", "top-left", "top-right", "bottom-left", "bottom-right"], {"default": "center"}),
                "fill_color": (["black", "white", "gray", "edge_extend"], {"default": "black"}),

                "image_1": ("IMAGE",),
            },
            "optional": {
                "image_2": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("normalized_images",)
    FUNCTION = "normalize"
    CATEGORY = "Google AI/Utils"
    DESCRIPTION = """
Normalize multiple images to the same size with dynamic inputs.
Click 'Update inputs' to change the number of image inputs.

Resize modes:
- max_resolution: Limit to max resolution, then expand canvas
- min_resolution: Ensure minimum resolution
- first_image: Match first image size
- largest_image: Match largest image in batch
"""
    
    def get_fill_value(self, fill_color):
        """Get the fill value based on color choice"""
        if fill_color == "black":
            return 0.0
        elif fill_color == "white":
            return 1.0
        elif fill_color == "gray":
            return 0.5
        return None  # edge_extend will be handled separately
    
    def resize_image_with_aspect(self, img_tensor, target_width, target_height, upscale_method):
        """
        Resize image while maintaining aspect ratio
        img_tensor: (1, C, H, W)
        Returns: (1, C, new_H, new_W) where new dimensions fit within target
        """
        _, _, h, w = img_tensor.shape
        aspect_ratio = w / h
        target_aspect_ratio = target_width / target_height
        
        # Determine new dimensions to fit within target while maintaining aspect ratio
        if aspect_ratio > target_aspect_ratio:
            # Width is the limiting factor
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
        else:
            # Height is the limiting factor
            new_height = target_height
            new_width = int(target_height * aspect_ratio)
        
        # Ensure dimensions are at least 1
        new_width = max(1, new_width)
        new_height = max(1, new_height)
        
        # Upscale the image
        if upscale_method == "lanczos":
            # PyTorch doesn't support lanczos, use PIL instead
            img_pil = Image.fromarray((img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            img_pil = img_pil.resize((new_width, new_height), Image.LANCZOS)
            resized = torch.from_numpy(np.array(img_pil).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        else:
            resized = F.interpolate(
                img_tensor,
                size=(new_height, new_width),
                mode=upscale_method,
                align_corners=False if upscale_method in ["bilinear", "bicubic"] else None
            )
        
        return resized, new_width, new_height
    
    def place_on_canvas(self, img_tensor, canvas_width, canvas_height, canvas_position, fill_color):
        """
        Place image on canvas with specified position and fill
        img_tensor: (1, C, H, W)
        Returns: (1, C, canvas_height, canvas_width)
        """
        _, c, h, w = img_tensor.shape
        
        if fill_color == "edge_extend":
            # Calculate padding needed
            pad_left = (canvas_width - w) // 2 if canvas_position == "center" else 0
            pad_right = canvas_width - w - pad_left
            pad_top = (canvas_height - h) // 2 if canvas_position == "center" else 0
            pad_bottom = canvas_height - h - pad_top
            
            # Adjust padding based on position
            if canvas_position == "top-left":
                pad_left, pad_top = 0, 0
                pad_right, pad_bottom = canvas_width - w, canvas_height - h
            elif canvas_position == "top-right":
                pad_left, pad_top = canvas_width - w, 0
                pad_right, pad_bottom = 0, canvas_height - h
            elif canvas_position == "bottom-left":
                pad_left, pad_top = 0, canvas_height - h
                pad_right, pad_bottom = canvas_width - w, 0
            elif canvas_position == "bottom-right":
                pad_left, pad_top = canvas_width - w, canvas_height - h
                pad_right, pad_bottom = 0, 0
            
            # Use edge padding
            canvas = F.pad(img_tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='replicate')
        else:
            # Create canvas with fill color
            fill_value = self.get_fill_value(fill_color)
            canvas = torch.full((1, c, canvas_height, canvas_width), fill_value, dtype=img_tensor.dtype, device=img_tensor.device)
            
            # Calculate position to place the image
            if canvas_position == "center":
                y_offset = (canvas_height - h) // 2
                x_offset = (canvas_width - w) // 2
            elif canvas_position == "top-left":
                y_offset, x_offset = 0, 0
            elif canvas_position == "top-right":
                y_offset = 0
                x_offset = canvas_width - w
            elif canvas_position == "bottom-left":
                y_offset = canvas_height - h
                x_offset = 0
            elif canvas_position == "bottom-right":
                y_offset = canvas_height - h
                x_offset = canvas_width - w
            else:  # default to center
                y_offset = (canvas_height - h) // 2
                x_offset = (canvas_width - w) // 2
            
            # Place image on canvas
            canvas[:, :, y_offset:y_offset+h, x_offset:x_offset+w] = img_tensor
        
        return canvas
    
    def normalize(self, inputcount, resize_mode, resolution_value, upscale_method, 
                  canvas_position, fill_color, image_1, **kwargs):
        """
        Normalize all images to the same size based on the selected mode
        """
        # Collect all input images
        images = []
        
        # Add the first required image
        if image_1 is not None:
            if image_1.shape[0] > 1:
                for j in range(image_1.shape[0]):
                    images.append(image_1[j:j+1])
            else:
                images.append(image_1)
        
        # Add optional images from kwargs
        for i in range(2, inputcount + 1):
            img = kwargs.get(f"image_{i}")
            if img is not None:
                # Handle batch images
                if img.shape[0] > 1:
                    for j in range(img.shape[0]):
                        images.append(img[j:j+1])
                else:
                    images.append(img)
        
        if len(images) == 0:
            # Return empty tensor
            return (torch.zeros((1, 512, 512, 3)),)
        
        # Determine target dimensions based on resize_mode
        if resize_mode == "first_image":
            target_height = images[0].shape[1]
            target_width = images[0].shape[2]
        elif resize_mode == "largest_image":
            target_height = max(img.shape[1] for img in images)
            target_width = max(img.shape[2] for img in images)
        elif resize_mode == "max_resolution":
            # Limit to max_resolution and create square canvas
            # First find the largest dimension
            max_h = max(img.shape[1] for img in images)
            max_w = max(img.shape[2] for img in images)
            
            # Determine the limiting dimension
            largest_dim = max(max(max_h, max_w), resolution_value)
            
            target_height = largest_dim
            target_width = largest_dim
                
        elif resize_mode == "min_resolution":
            # Ensure all images are at least max_resolution
            max_h = max(img.shape[1] for img in images)
            max_w = max(img.shape[2] for img in images)
            
            largest_dim = min(resolution_value, max(max_h, max_w)) 
            
            target_height = largest_dim
            target_width = largest_dim
                
        # Process each image
        normalized_images = []
        for img in images:
            # Convert to (1, C, H, W) for processing
            img_tensor = img.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
            
            # Resize with aspect ratio preservation
            resized, new_w, new_h = self.resize_image_with_aspect(
                img_tensor, target_width, target_height, upscale_method
            )
            
            # Place on canvas if needed
            if new_w != target_width or new_h != target_height:
                canvas = self.place_on_canvas(
                    resized, target_width, target_height, canvas_position, fill_color
                )
            else:
                canvas = resized
            
            # Convert back to (B, H, W, C)
            result = canvas.permute(0, 2, 3, 1)
            normalized_images.append(result)
        
        # Concatenate all normalized images
        output = torch.cat(normalized_images, dim=0)
        
        return (output,)

