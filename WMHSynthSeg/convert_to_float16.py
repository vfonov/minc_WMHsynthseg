from onnxconverter_common import auto_mixed_precision

import onnx
from onnxconverter_common import float16


model = onnx.load("minc_WMHsynthseg/WMHSynthSeg/WMH-SynthSeg_v10_231110.onnx")

model_fp16 = float16.convert_float_to_float16(model,keep_io_types=True)

onnx.save(model_fp16, "minc_WMHsynthseg/WMHSynthSeg/WMH-SynthSeg_v10_231110_f16.onnx")
