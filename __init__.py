import os
import folder_paths

from .prompt import GoogleGeminiPrompt

NODE_CLASS_MAPPINGS = {
    "GoogleGeminiPrompt" : GoogleGeminiPrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GoogleGeminiPrompt" : "Google Gemini Prompt",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
