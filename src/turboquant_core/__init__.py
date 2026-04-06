from turboquant_core.core import (
    CodebookRegistry,
    RotationCache,
    QJLProjection,
    TQCodebook,
    TQGatedAttnKVCache,
    TQQuantizedCache,
    # Experimental, untested wrappers — importable but excluded from __all__:
    TQActivationCheckpoint as TQActivationCheckpoint,
    TQLoRAStorage as TQLoRAStorage,
    tq_quantize_mse,
    tq_dequantize_mse,
    tq_quantize_mse_ste,
    tq_quantize_prod,
    tq_rotate,
    tq_rotate_inv,
)
from turboquant_core.backends.qwen import Qwen35KVBackend, Qwen3DenseKVBackend, Qwen25DenseKVBackend
from turboquant_core.backends.qwen_hook import (
    patch_qwen35_with_tq, patch_qwen3_with_tq, patch_qwen25_with_tq, unpatch_model,
)
from turboquant_core.adapters.workflow_eval import TurboQuantAdapter, register_variant

__all__ = [
    "CodebookRegistry",
    "RotationCache",
    "QJLProjection",
    "TQCodebook",
    "TQGatedAttnKVCache",
    "TQQuantizedCache",
    "tq_quantize_mse",
    "tq_dequantize_mse",
    "tq_quantize_mse_ste",
    "tq_quantize_prod",
    "tq_rotate",
    "tq_rotate_inv",
    "Qwen35KVBackend",
    "Qwen3DenseKVBackend",
    "Qwen25DenseKVBackend",
    "patch_qwen35_with_tq",
    "patch_qwen3_with_tq",
    "patch_qwen25_with_tq",
    "unpatch_model",
    "TurboQuantAdapter",
    "register_variant",
]
