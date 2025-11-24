import torch
import numpy as np
from PIL import Image
import io
import base64
import os

# Try importing the new SDK
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    print("Warning: google-genai package not found. Please install it with: pip install google-genai")

class NanobananaNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "prompt": ("STRING", {"default": "Generate a high-quality image", "multiline": True}),
                "model": (["nano-banana-pro-preview", "nanobanana", "gemini-2.0-flash-exp"], {"default": "nanobanana"}),
                # "operation": (["generate", "edit", "style_transfer"], {"default": "generate"}),
                "api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "Google AI API Key"}),
            },
            "optional": {
                "images": ("IMAGE",),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_images"
    CATEGORY = "Google AI"

    def _tensor_to_pil(self, tensor):
        if tensor.ndim == 4:
            tensor = tensor.squeeze(0)
        return Image.fromarray(np.clip(255. * tensor.cpu().numpy(), 0, 255).astype(np.uint8))

    def _pil_to_tensor(self, pil_image):
        return torch.from_numpy(np.array(pil_image).astype(np.float32) / 255.0).unsqueeze(0)

    def _image_to_base64(self, pil_image):
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        return base64.b64encode(img_bytes).decode('utf-8')

    def process_images(self, seed, prompt, model, api_key, images=None, temperature=0.7):
        if not genai:
            raise ImportError("The 'google-genai' package is required. Please install it: pip install google-genai")
        
        if not api_key:
            # Try to get from env
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("API Key is required")

        client = genai.Client(api_key=api_key)
        
        # Model mapping: nanobanana -> gemini-2.0-flash-exp
        # nano-banana-pro-preview -> gemini-2.0-flash-exp-image-generation
        target_model = model
        # if model == "nanobanana":
        #     target_model = "gemini-2.0-flash-exp"

        # Prepare content parts
        parts = [{"text": prompt}]
        
        # Collect all reference images from the batch
        # images is (B, H, W, C) - batch of images
        # encoded_images = []
        
        if images is not None:
            # Determine image limit based on model
            if model == "gemini-2.0-flash-exp":
                image_limit = 1  # Only first image
            elif model == "nanobanana":
                image_limit = 5  # Max 5 images
            else:  # nano-banana-pro-preview
                image_limit = None  # No limit

            # Process images from the batch
            for idx, image_tensor in enumerate(images):
                # Apply image limit if set
                if image_limit is not None and idx >= image_limit:
                    break
                
                # Convert tensor to PIL image
                pil_image = self._tensor_to_pil(image_tensor)
                b64_data = self._image_to_base64(pil_image)
                
                # Add image to parts
                parts.append({
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": b64_data
                    }
                })

                # encoded_images.append(b64_data)
        
        # has_references = len(encoded_images) > 0

        output_images = []

        try:
            generation_config = types.GenerateContentConfig(
                temperature=temperature,
                response_modalities=['Text', 'Image'] 
            )

            model_id = target_model

            response = client.models.generate_content(
                model=model_id,
                contents=[{"parts": parts}],
                config=generation_config
            )

            # Extract images
            if hasattr(response, 'candidates') and response.candidates:
                for candidate in response.candidates:
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        for part in candidate.content.parts:
                            if hasattr(part, 'inline_data') and part.inline_data:
                                image_binary = part.inline_data.data
                                gen_img = Image.open(io.BytesIO(image_binary))
                                if gen_img.mode != "RGB":
                                    gen_img = gen_img.convert("RGB")
                                generated_image_tensor = self._pil_to_tensor(gen_img)
                                output_images.append(generated_image_tensor)
            
            if not output_images:
                print(f"No image generated. Response: {response}")
                # Return a blank image or the first reference as fallback?
                # Returning blank to indicate failure but keep flow alive
                return (torch.zeros((1, 512, 512, 3)),)

        except Exception as e:
            print(f"Error generating image: {e}")
            return (torch.zeros((1, 512, 512, 3)),)

        # Stack all generated images (usually 1, unless n>1 requested in config)
        # We need to ensure they are same size for stacking
        if len(output_images) > 1:
            first_shape = output_images[0].shape
            resized_outputs = []
            for img in output_images:
                if img.shape != first_shape:
                    img_p = img.permute(0, 3, 1, 2)
                    target_h, target_w = first_shape[1], first_shape[2]
                    img_r = torch.nn.functional.interpolate(img_p, size=(target_h, target_w), mode='bilinear')
                    img_out = img_r.permute(0, 2, 3, 1)
                    resized_outputs.append(img_out)
                else:
                    resized_outputs.append(img)
            return (torch.cat(resized_outputs, dim=0),)
        else:
            return (output_images[0],)
