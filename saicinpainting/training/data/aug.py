import cv2
import albumentations as A


# imgaug used string border modes; albumentations' native Affine/Perspective use
# OpenCV border flags. This maps the old imgaug ``mode`` strings onto them.
_IMGAUG_MODE_TO_CV2 = {
    'reflect': cv2.BORDER_REFLECT_101,
    'symmetric': cv2.BORDER_REFLECT,
    'replicate': cv2.BORDER_REPLICATE,
    'edge': cv2.BORDER_REPLICATE,
    'constant': cv2.BORDER_CONSTANT,
    'wrap': cv2.BORDER_WRAP,
}

# imgaug interpolation ``order`` -> OpenCV interpolation flag.
_ORDER_TO_INTERPOLATION = {
    0: cv2.INTER_NEAREST,
    1: cv2.INTER_LINEAR,
    2: cv2.INTER_CUBIC,
    3: cv2.INTER_CUBIC,
    4: cv2.INTER_LANCZOS4,
}


def _to_range(value):
    """Mirror albumentations.to_tuple for a scalar: x -> (-x, x); pass tuples through."""
    if isinstance(value, (tuple, list)):
        return tuple(value)
    return (-value, value)


class IAAAffine2(A.Affine):
    """Drop-in replacement for the original imgaug-backed affine transform.

    The original class subclassed albumentations' ``DualIAATransform`` (removed in
    albumentations >= 1.0) and delegated to ``imgaug``. albumentations now provides
    a native, imgaug-free ``Affine`` transform, so we just map the old imgaug-style
    arguments used throughout the LaMa configs onto it. Passing ``scale``/``shear``
    as ``{'x': ..., 'y': ...}`` preserves the original per-axis (independent)
    sampling behaviour.

    Targets: image, mask
    """

    def __init__(
        self,
        scale=(0.7, 1.3),
        translate_percent=None,
        translate_px=None,
        rotate=0.0,
        shear=(-0.1, 0.1),
        order=1,
        cval=0,
        mode="reflect",
        always_apply=False,
        p=0.5,
    ):
        super().__init__(
            scale=dict(x=scale, y=scale),
            translate_percent=translate_percent,
            translate_px=translate_px,
            rotate=_to_range(rotate),
            shear=dict(x=shear, y=shear),
            interpolation=_ORDER_TO_INTERPOLATION.get(order, cv2.INTER_LINEAR),
            cval=cval,
            mode=_IMGAUG_MODE_TO_CV2.get(mode, cv2.BORDER_REFLECT_101),
            always_apply=always_apply,
            p=p,
        )


class IAAPerspective2(A.Perspective):
    """Native albumentations replacement for the imgaug-backed perspective transform.

    Targets: image, mask
    """

    def __init__(self, scale=(0.05, 0.1), keep_size=True, always_apply=False, p=0.5,
                 order=1, cval=0, mode="replicate"):
        super().__init__(
            scale=scale,
            keep_size=keep_size,
            pad_mode=_IMGAUG_MODE_TO_CV2.get(mode, cv2.BORDER_REPLICATE),
            pad_val=cval,
            interpolation=_ORDER_TO_INTERPOLATION.get(order, cv2.INTER_LINEAR),
            always_apply=always_apply,
            p=p,
        )
