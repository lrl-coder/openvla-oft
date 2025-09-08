# InternVL3-1B Implementation Plan for LIBERO

## Overview
This document outlines the complete plan to replace OpenVLA-7B with InternVL3-1B for LIBERO finetuning and evaluation while maintaining minimal code changes.

## Current Architecture Analysis

### Existing OpenVLA Structure
- **Vision**: SigLIP ViT-SO400M (224px)
- **Language**: Vicuna-7B
- **Interface**: `prismatic/extern/hf/` provides HuggingFace-compatible wrappers
- **Key Classes**:
  - `OpenVLAConfig` - Model configuration
  - `OpenVLAForActionPrediction` - Main model with action head
  - `PrismaticImageProcessor` - Image preprocessing
  - `PrismaticProcessor` - Combined processor

### Target InternVL3-1B Structure
- **Vision**: InternViT-300M-448px-V2_5 (448px)
- **Language**: Qwen2.5-0.5B
- **Checkpoint**: `OpenGVLab/InternVL3-1B-hf` (HuggingFace format)
- **Size**: ~938M parameters (vs 7B for OpenVLA)
- **Available Implementation**: `prismatic/extern/internvl/` (from HuggingFace transformers)

### Available InternVL Classes
- **Configuration**: `InternVLConfig`, `InternVLVisionConfig`
- **Models**: `InternVLForConditionalGeneration`, `InternVLModel`, `InternVLVisionModel`
- **Processing**: `InternVLProcessor` (combines image + text processing)

## Implementation Strategy

### Phase 1: Backup and Rename Existing Code
```bash
# Backup current implementation
mv prismatic/extern/hf prismatic/extern/hf_openvla_backup
```

### Phase 2: Create InternVL3 Wrapper Classes

#### 2.1 Configuration (`configuration_internvl3.py`)
- Extend `InternVLConfig` from existing implementation
- Add OpenVLA-specific parameters (n_action_bins, norm_stats)
- Map InternVL3 parameters to OpenVLA-compatible interface
- Handle vision/language component configurations

#### 2.2 Model (`modeling_internvl3.py`)
- Extend `InternVLForConditionalGeneration` from existing implementation
- Implement `InternVL3ForActionPrediction` class
- Add action prediction head (similar to OpenVLA's approach)
- Handle vision token processing and action token generation
- Support LoRA fine-tuning

#### 2.3 Processor (`processing_internvl3.py`)
- Use existing `InternVLProcessor` as base
- Create compatibility wrapper for OpenVLA interface
- Handle dynamic resolution and multi-image support
- Maintain compatibility with existing preprocessing pipeline

### Phase 3: Create Dual-Class Compatibility Layer

#### 3.1 Improved File Structure (`prismatic/extern/hf/`)
Create a clean modular structure supporting both implementations:

```python
# hf/__init__.py - Main entry point with dual import strategy
# Default imports (InternVL3-based)
from .internvl.configuration_internvl3 import InternVL3Config as OpenVLAConfig
from .internvl.modeling_internvl3 import InternVL3ForActionPrediction as OpenVLAForActionPrediction
from .internvl.processing_internvl3 import InternVL3Processor as PrismaticProcessor
from .internvl.processing_internvl3 import InternVL3ImageProcessor as PrismaticImageProcessor

# Original OpenVLA classes
from .prismatic.configuration_prismatic import OpenVLAConfig as OpenVLAConfigOriginal
from .prismatic.modeling_prismatic import OpenVLAForActionPrediction as OpenVLAForActionPredictionOriginal
from .prismatic.processing_prismatic import PrismaticProcessor as PrismaticProcessorOriginal
from .prismatic.processing_prismatic import PrismaticImageProcessor as PrismaticImageProcessorOriginal

# Export both sets for flexibility
__all__ = [
    # Default (InternVL3-based) classes
    "OpenVLAConfig", "OpenVLAForActionPrediction", "PrismaticProcessor", "PrismaticImageProcessor",
    # Original OpenVLA classes
    "OpenVLAConfigOriginal", "OpenVLAForActionPredictionOriginal",
    "PrismaticProcessorOriginal", "PrismaticImageProcessorOriginal"
]
```

#### 3.2 Single `__init__.py` Structure
Only one Python file needed - the main `hf/__init__.py` handles all imports:

### Phase 4: Flexible Import Strategy

#### 4.1 Default Behavior (InternVL3)
- Training Script (`vla-scripts/finetune.py`): No changes needed - imports remain the same
- Evaluation Scripts (`experiments/robot/openvla_utils.py`): No changes needed - imports remain the same
- LIBERO Evaluation (`experiments/robot/libero/run_libero_eval.py`): No changes needed - uses `get_model()` function
- **Default imports automatically use InternVL3 implementation**

#### 4.2 Multiple Import Options
Users can import in several ways depending on their needs:

```python
# Option 1: Default imports (InternVL3-based)
from prismatic.extern.hf import OpenVLAConfig, OpenVLAForActionPrediction

# Option 2: Original OpenVLA imports (via compatibility names)
from prismatic.extern.hf import OpenVLAConfigOriginal, OpenVLAForActionPredictionOriginal

# Option 3: Dynamic switching
USE_INTERNVL3 = True
if USE_INTERNVL3:
    from prismatic.extern.hf import OpenVLAConfig, OpenVLAForActionPrediction
else:
    from prismatic.extern.hf import OpenVLAConfigOriginal as OpenVLAConfig
    from prismatic.extern.hf import OpenVLAForActionPredictionOriginal as OpenVLAForActionPrediction
```

**Note**: Since subdirectories don't have `__init__.py` files, direct imports from `hf.internvl3` or `hf.prismatic` are not available. All imports go through the main `hf/__init__.py`.

### Phase 5: Configuration Updates

#### 5.1 Model Parameters
- Use checkpoint: `OpenGVLab/InternVL3-1B-hf`
- Update default image size: 224 → 448
- Leverage existing InternVL configuration
- Add action prediction settings (n_action_bins, norm_stats)

#### 5.2 Training Hyperparameters
- Adjust batch size for smaller model (1B vs 7B)
- Update learning rate for different architecture
- Modify memory requirements

## Implementation Details

### Key Technical Challenges

1. **Action Prediction Head Integration**
   - InternVL3 is designed for chat, not action prediction
   - Need to add action tokenization and prediction layers
   - Maintain compatibility with existing action space (7-dim continuous)

2. **Image Processing Differences**
   - OpenVLA: 224px square images
   - InternVL3: 448px with dynamic resolution
   - Need to handle LIBERO's image format appropriately

3. **Model Size Adaptation**
   - Smaller model may need different training strategies
   - Batch size and learning rate adjustments
   - Memory usage optimization

4. **Vision-Language Alignment**
   - Different vision encoder (InternViT vs SigLIP)
   - Different language model (Qwen2.5 vs Vicuna)
   - May need alignment layer adjustments

### File Structure
```
prismatic/extern/
├── internvl/                   # Existing InternVL implementation (from HF transformers)
│   ├── __init__.py
│   ├── configuration_internvl.py
│   ├── modeling_internvl.py
│   ├── processing_internvl.py
│   └── video_processing_internvl.py
└── hf/                         # Dual-class compatibility layer
    ├── __init__.py             # Main entry point - handles all imports
    ├── internvl/               # InternVL3-based implementation
    │   ├── config/
    │   │   └── configuration_internvl3.py  # InternVL3 config with action prediction
    │   ├── modeling/
    │   │   └── modeling_internvl3.py       # InternVL3 model with action prediction
    │   └── processing/
    │       └── processing_internvl3.py     # InternVL3 processor wrapper
    └── prismatic/              # Original OpenVLA implementation
        ├── config/
        │   └── configuration_prismatic.py  # Original OpenVLA config
        ├── modeling/
        │   └── modeling_prismatic.py       # Original OpenVLA model
        └── processing/
            └── processing_prismatic.py     # Original OpenVLA processor
```

## Testing Strategy

### Phase 1: Basic Integration
1. Test model loading with InternVL3-1B
2. Verify image processing pipeline
3. Check configuration compatibility

### Phase 2: Training Validation
1. Run short training on small dataset
2. Verify gradient flow and loss computation
3. Test LoRA integration

### Phase 3: LIBERO Evaluation
1. Test evaluation script with pretrained model
2. Compare performance with OpenVLA baseline
3. Validate action prediction accuracy

## Risk Mitigation

1. **Backup Strategy**: Keep original implementation intact
2. **Gradual Migration**: Test each component independently
3. **Fallback Option**: Easy revert to original OpenVLA
4. **Performance Monitoring**: Track metrics throughout migration

## Expected Benefits

1. **Reduced Memory**: ~1B vs 7B parameters
2. **Faster Training**: Smaller model, faster iterations
3. **Modern Architecture**: Latest InternVL3 improvements
4. **Better Vision**: Higher resolution (448px vs 224px)

## Dual-Class Implementation Approach

Given the existing InternVL implementation and your requirement for both classes to coexist:

1. **Preserve Original**: Keep original OpenVLA implementation in `hf_openvla_backup/`
2. **Leverage Existing Code**: Use `prismatic/extern/internvl/` as the foundation for InternVL3
3. **Extend for Action Prediction**: Add action prediction head to `InternVLForConditionalGeneration`
4. **Dual Export Strategy**: Export both original and InternVL3 classes from `hf/`
5. **Use HF Checkpoint**: Load `OpenGVLab/InternVL3-1B-hf` directly for InternVL3 classes

### Benefits of Dual-Class Approach:
- **Gradual Migration**: Test InternVL3 while keeping original as fallback
- **A/B Testing**: Easy comparison between OpenVLA-7B and InternVL3-1B
- **Backward Compatibility**: Existing code continues to work unchanged
- **Explicit Choice**: Users can explicitly choose which implementation to use

## Timeline Estimate

- **Phase 1**: 1 hour (backup existing code)
- **Phase 2**: 1-2 days (extend InternVL for action prediction)
- **Phase 3-4**: 4-6 hours (compatibility layer)
- **Phase 5**: 1 day (configuration and testing)
- **Total**: 2-3 days for complete implementation

## Implementation Strategy

### Inheritance Strategy Decision Matrix

Based on actual code analysis, here's the optimal inheritance approach:

| Component | Inheritance Strategy | Rationale |
|-----------|---------------------|-----------|
| **Config** | Inherit from `OpenVLAConfig` (Prismatic) | Keep action-specific config structure, update model specs |
| **Image Processor** | Inherit from `transformers.models.internvl.InternVLImageProcessor` | Leverage InternVL's optimized image processing |
| **Combined Processor** | Inherit from `PrismaticProcessor` (Prismatic) | Keep proven wrapper structure (image_processor + tokenizer) |
| **Model** | Inherit from `OpenVLAForActionPrediction` (Prismatic) | Keep action prediction logic, swap vision/language backends |
| **Action Methods** | Inherited from Prismatic | All action prediction functionality already implemented |

**Key Insight**: `PrismaticProcessor` is a wrapper that combines `image_processor` + `tokenizer` - there's no separate "text processor". We inherit the wrapper structure from Prismatic and plug in InternVL3 components.

### Directory Structure and Inheritance Decisions

#### File Structure
```
prismatic/extern/hf/
├── __init__.py                 # Main entry point - handles all imports
├── internvl/                   # InternVL3-based implementation
│   ├── configuration_internvl3.py     # Config: Inherit from OpenVLAConfig
│   ├── modeling_internvl3.py          # Model: Inherit from OpenVLAForActionPrediction
│   └── processing_internvl3.py        # Processor: Inherit from PrismaticProcessor + InternVLImageProcessor
└── prismatic/                  # Original OpenVLA implementation
    ├── configuration_prismatic.py     # Original OpenVLA config
    ├── modeling_prismatic.py          # Original OpenVLA model
    └── processing_prismatic.py        # Original OpenVLA processor
```

#### Inheritance Strategy for Each Component

**1. Configuration (`hf/internvl/configuration_internvl3.py`)**
```python
from ..prismatic.configuration_prismatic import OpenVLAConfig

class InternVL3Config(OpenVLAConfig):
    """Inherit from OpenVLA config, update model specifications for InternVL3"""
    def __init__(self, **kwargs):
        # Update default values for InternVL3-1B
        kwargs.setdefault("vision_backbone_id", "internvl3-vision")
        kwargs.setdefault("llm_backbone_id", "qwen2.5-0.5b")
        super().__init__(**kwargs)
```
- **Strategy**: Inherit from `OpenVLAConfig` (Prismatic)
- **Rationale**: Keep action-specific configuration structure, just update model specifications

**2. Image Processing (`hf/internvl/processing_internvl3.py`)**
```python
from transformers.models.internvl import InternVLImageProcessor

class InternVL3ImageProcessor(InternVLImageProcessor):
    """Use InternVL's optimized image processing for InternViT backbone"""
    pass
```
- **Strategy**: Inherit from `transformers.models.internvl.InternVLImageProcessor`
- **Rationale**: Leverage InternVL's optimized image preprocessing for InternViT vision backbone

**3. Combined Processor (`hf/internvl/processing_internvl3.py`)**
```python
from ..prismatic.processing_prismatic import PrismaticProcessor

class InternVL3Processor(PrismaticProcessor):
    """Keep Prismatic's processor structure, use InternVL3 components"""
    def __init__(self, image_processor=None, tokenizer=None):
        # Use InternVL3ImageProcessor + InternVL's tokenizer
        super().__init__(image_processor, tokenizer)
```
- **Strategy**: Inherit from `PrismaticProcessor` (Prismatic)
- **Rationale**: Keep the proven processor wrapper structure (image_processor + tokenizer), just plug in InternVL3 components

**4. Modeling (`hf/internvl/modeling_internvl3.py`)**
```python
from ..prismatic.modeling_prismatic import OpenVLAForActionPrediction

class InternVL3ForActionPrediction(OpenVLAForActionPrediction):
    """Keep action prediction logic, swap vision/language backends to InternVL3"""
    def __init__(self, config):
        super().__init__(config)
        # Replace vision_backbone and language_model with InternVL3 components
        # Keep all action prediction methods intact
```
- **Strategy**: Inherit from `OpenVLAForActionPrediction` (Prismatic)
- **Rationale**: Keep all the complex action prediction logic intact, just swap the vision/language backends

**5. Action-Specific Methods**
- **Strategy**: Inherited from Prismatic (no changes needed)
- **Rationale**: All action prediction functionality (`predict_action`, `_unnormalize_actions`, etc.) already implemented in OpenVLA

## Implementation Steps

1. **Reorganize existing OpenVLA implementation** to `hf/prismatic/` directory structure
2. **Create InternVL3 implementation** in `hf/internvl/` directory
3. **Implement inheritance strategy** for each component as outlined above
4. **Update `hf/__init__.py`** to expose both implementations
5. **Test basic model loading** with `OpenGVLab/InternVL3-1B-hf`
6. **Validate training pipeline** with both implementations
7. **Run LIBERO evaluation** and compare results

## Usage Examples After Implementation

### Default Usage (InternVL3-1B)
```python
# Existing code works unchanged - now uses InternVL3-1B
from prismatic.extern.hf import OpenVLAConfig, OpenVLAForActionPrediction
model = OpenVLAForActionPrediction.from_pretrained("OpenGVLab/InternVL3-1B-hf")
```

### Original OpenVLA Usage
```python
# Use original OpenVLA-7B implementation (via compatibility names)
from prismatic.extern.hf import OpenVLAConfigOriginal, OpenVLAForActionPredictionOriginal
model = OpenVLAForActionPredictionOriginal.from_pretrained("openvla/openvla-7b")
```

### A/B Testing
```python
# Easy comparison between implementations
from prismatic.extern.hf import OpenVLAForActionPrediction, OpenVLAForActionPredictionOriginal

models = {
    "internvl3_1b": OpenVLAForActionPrediction.from_pretrained("OpenGVLab/InternVL3-1B-hf"),
    "openvla_7b": OpenVLAForActionPredictionOriginal.from_pretrained("openvla/openvla-7b")
}
```
