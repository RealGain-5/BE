import base64
from io import BytesIO


def image_to_base64(pil_img):
    """
    PIL img => Base64
    without diskIO
    """
    buffered = BytesIO()
    # save to memory in PNG format
    pil_img.save(buffered, format="PNG")
    # encoding Base64
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    # transform to use on web(renderer)
    return f"data:image/png;base64,{img_str}"
