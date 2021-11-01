import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
from cloths_segmentation.pre_trained_models import create_model

_model_seg = create_model("Unet_2020-10-30")
_model_seg.eval()
scripted_module = torch.jit.script(_model_seg)
optimized_scripted_module = optimize_for_mobile(scripted_module)
scripted_module._save_for_lite_interpreter("deeplabv3_scripted.ptl")
# using optimized lite interpreter model makes inference about 60% faster than the non-optimized lite interpreter model, which is about 6% faster than the non-optimized full jit model
optimized_scripted_module._save_for_lite_interpreter("deeplabv3_scripted_optimized.ptl")