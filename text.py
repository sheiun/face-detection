from numpy import asarray
from PIL import Image, ImageDraw, ImageFont


def put_text(
    img,
    text: str,
    org,
    fontFace=None,
    fontScale=1,
    color="White",
    thickness=None,
    lineType=None,
    bottomLeftOrigin=None,
):
    """
    Usage:
    >>> from text import put_text
    >>> cv2.putText = put_text
    """
    font = ImageFont.truetype("./data/NotoSansCJKtc-Regular.otf", int(12 * fontScale))
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    draw.text((org[0], org[1]), text, font=font, fill=color)
    return asarray(img)
