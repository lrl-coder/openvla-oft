from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageTextToText, AutoProcessor

from .configuration_prismatic import OpenVLAConfig
from .modeling_prismatic import OpenVLAForActionPrediction
from .processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

AutoConfig.register("openvla", OpenVLAConfig)
AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
AutoModelForImageTextToText.register(OpenVLAConfig, OpenVLAForActionPrediction)
