from functools import partial
from six.moves import map, zip
import torch


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def bboxes_area(bboxes):
    whs = torch.clamp(bboxes[:, 2:4] - bboxes[:, :2], min=0.0)
    return whs.prod(-1)


def bboxes_clamp(boxes, bound):   # xyxy
    boxes[..., 0::2] = boxes[..., 0::2].clamp(min=bound[0], max=bound[2])   # x1 x2
    boxes[..., 1::2] = boxes[..., 1::2].clamp(min=bound[1], max=bound[3])   # y1 y2

    return boxes
