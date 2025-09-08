from transformers import AutoTokenizer
from transformers.models.internvl import InternVLConfig, InternVLForConditionalGeneration

from prismatic.extern.hf.internvl import OpenVLAInternVLConfig, PrismaticInternVLProcessor
from prismatic.extern.hf.prismatic import OpenVLAConfig

OPENVLA_MODEL_PATH = "openvla/openvla-7b"
INTERNVL_MODEL_PATH = "OpenGVLab/InternVL3-1B-hf"
OUTPUT_PATH = "/mnt/harbor/projects/owa/checkpoints/OpenVLA-InternVL3-1B-hf"

original_openvla_config = OpenVLAConfig.from_pretrained(OPENVLA_MODEL_PATH)
original_internvl_config = InternVLConfig.from_pretrained(INTERNVL_MODEL_PATH)

merged_config = OpenVLAInternVLConfig(
    **original_internvl_config.to_dict(),
    norm_stats=original_openvla_config.norm_stats,
    n_action_bins=original_openvla_config.n_action_bins,
)
merged_config.architectures = ["OpenVLAInternVLForActionPrediction"]

model = InternVLForConditionalGeneration.from_pretrained(INTERNVL_MODEL_PATH)
# processor = InternVLProcessor.from_pretrained(INTERNVL_MODEL_PATH)
processor = PrismaticInternVLProcessor.from_pretrained(OPENVLA_MODEL_PATH)
# del processor.image_processor.auto_map
tokenizer = AutoTokenizer.from_pretrained(INTERNVL_MODEL_PATH)
processor.tokenizer = tokenizer


model.save_pretrained(OUTPUT_PATH)
processor.save_pretrained(OUTPUT_PATH)
merged_config.save_pretrained(OUTPUT_PATH)

# After the run, remove all "auto_map" from saved configs.
