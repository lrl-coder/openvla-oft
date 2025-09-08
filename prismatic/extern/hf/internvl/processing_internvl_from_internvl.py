from transformers.models.internvl import InternVLProcessor


class PrismaticInternVLProcessor(InternVLProcessor):
    def __call__(self, text, images, *args, **kwargs):
        output = super().__call__(*args, text=text, images=images, **kwargs)
        # reshape pixel_values to (B, C, H, W)
        # output["pixel_values"] = output["pixel_values"].reshape(-1, 3, 224, 224)
        return output
