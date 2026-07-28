"""Validation helpers for uploaded face images."""

from fastapi import HTTPException, UploadFile, status


ALLOWED_IMAGE_TYPES = frozenset(
    {
        "image/jpeg",
        "image/png",
        "image/webp",
    }
)


async def read_image_upload(upload: UploadFile, max_size_bytes: int) -> bytes:
    """Read a bounded image upload and reject obviously invalid input."""

    if upload.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={
                "code": "unsupported_image_type",
                "message": "The image must be JPEG, PNG, or WebP.",
            },
        )

    image_bytes = await upload.read(max_size_bytes + 1)
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "empty_image",
                "message": "The uploaded image is empty.",
            },
        )
    if len(image_bytes) > max_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "code": "image_too_large",
                "message": "The uploaded image exceeds the configured size limit.",
            },
        )

    return image_bytes
