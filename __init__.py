import os
import folder_paths

from .prompt import GoogleGeminiPrompt
from .nanobanana import NanobananaNode
from .batch_image_normalizer import BatchImageNormalizer

NODE_CLASS_MAPPINGS = {
    "GoogleGeminiPrompt" : GoogleGeminiPrompt,
    "NanobananaNode" : NanobananaNode,
    "BatchImageNormalizer" : BatchImageNormalizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GoogleGeminiPrompt" : "Google Gemini Prompt",
    "NanobananaNode" : "Nanobanana Node",
    "BatchImageNormalizer" : "Batch Image Normalizer",
}

WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
