import os
import folder_paths

from .prompt import GoogleGeminiPrompt
from .nanobanana import NanobananaNode

NODE_CLASS_MAPPINGS = {
    "GoogleGeminiPrompt" : GoogleGeminiPrompt,
    "NanobananaNode" : NanobananaNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GoogleGeminiPrompt" : "Google Gemini Prompt",
    "NanobananaNode" : "Nanobanana Node",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
