import torch
import os
import json
from PIL import Image
import google.generativeai as genai
import numpy as np

class GoogleGeminiPrompt:
    _google_ai_models_cache = []

    def __init__(self):
        pass

    @classmethod
    def _get_google_ai_models(cls, api_key):
        if not genai:
            print("Google Generative AI SDK is not available.")
            return []
        if not api_key:
            print("Google AI API key not provided.")
            return []

        try:
            genai.configure(api_key=api_key)
            models = []
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods and "gemini" in m.name:
                    models.append(m.name.replace("models/", ""))
            cls._google_ai_models_cache = sorted(list(set(models)))
            if not cls._google_ai_models_cache:
                print("No suitable Gemini models found for Google AI provider.")
            else:
                print(f"Available Google AI (Gemini) Models: {cls._google_ai_models_cache}")
            return cls._google_ai_models_cache
        except Exception as e:
            print(f"Error fetching Google AI models: {str(e)}")
            return []

    @classmethod
    def INPUT_TYPES(cls):
        default_llm_model = "gemini-1.5-flash-latest"

        return {
            "required": {
                "google_api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "Your Google AI API Key"}),
                "llm_model": ("STRING", {"default": default_llm_model, "multiline": False, "placeholder": "e.g., gemini-1.5-flash-latest"}),
                "system_prompt": ("STRING", {"default": "", "multiline": True, "placeholder": "Optional system prompt"}),
                "user_prompt": ("STRING", {"default": "", "multiline": True, "placeholder": "Your main prompt or text"}),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "Google AI"

    def _convert_tensor_to_pil(self, image_tensor: torch.Tensor):
        if image_tensor is None:
            return None

        if image_tensor.ndim == 4 and image_tensor.shape[0] == 1:
            image_tensor = image_tensor.squeeze(0)

        img_np = image_tensor.cpu().numpy() * 255.0
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)

        return Image.fromarray(img_np)

    def execute(self, google_api_key, llm_model, system_prompt, user_prompt, image=None):
        if not genai:
            return ("Error: Google Generative AI SDK is not available. Please install it: pip install google-generativeai",)
        
        if not google_api_key:
            return ("Error: Google AI API key not provided.",)

        genai.configure(api_key=google_api_key)

        pil_image = None
        if image is not None:
            try:
                pil_image = self._convert_tensor_to_pil(image)
            except Exception as e:
                print(f"Warning: Could not convert image for Google AI: {e}")
                return (f"Error converting image: {e}",)

        effective_model_name = llm_model if llm_model.startswith("models/") else f"models/{llm_model}"

        prompt_parts = []
        if pil_image:
            prompt_parts.append(pil_image)
        prompt_parts.append(user_prompt)

        try:
            model_instance = genai.GenerativeModel(
                model_name=effective_model_name,
                system_instruction=system_prompt if system_prompt else None
            )

            generation_config = genai.types.GenerationConfig(
                temperature=0.7
            )
            
            response = model_instance.generate_content(
                prompt_parts,
                generation_config=generation_config
            )

            if not response.parts:
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                    block_message = response.prompt_feedback.block_reason_message or str(response.prompt_feedback.block_reason)
                    print(f"Google AI response blocked. Reason: {block_message}")
                    return (f"Google AI response blocked. Reason: {block_message}",)

                print("Google AI returned an empty response. The content might have been blocked.")
                return ("Google AI returned an empty response. The content might have been blocked.",)
            
            return (response.text,)

        except Exception as e:
            import traceback
            print(f"""Google AI API error: {str(e)}
{traceback.format_exc()}""")
            if "API key not valid" in str(e):
                return ("Error: Google AI API key is not valid. Please check your configuration.",)
            if "404" in str(e) and "models" in str(e):
                 return (f"Error: Google AI model '{llm_model}' not found or not accessible with your API key.",)
            return (f"Google AI API error: {str(e)}",)


