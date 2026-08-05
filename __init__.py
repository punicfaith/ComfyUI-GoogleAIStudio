import os
import folder_paths

from .prompt import GoogleGeminiPrompt
from .nanobanana import NanobananaNode
from .batch_image_normalizer import BatchImageNormalizer, BatchImageNormalizerBatch

NODE_CLASS_MAPPINGS = {
    "GoogleGeminiPrompt" : GoogleGeminiPrompt,
    "NanobananaNode" : NanobananaNode,
    "BatchImageNormalizer" : BatchImageNormalizer,
    "BatchImageNormalizerBatch" : BatchImageNormalizerBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GoogleGeminiPrompt" : "Google Gemini Prompt",
    "NanobananaNode" : "Nanobanana Node",
    "BatchImageNormalizer" : "Batch Image Normalizer",
    "BatchImageNormalizerBatch" : "Batch Image Normalizer (Batch In)",
}

WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
