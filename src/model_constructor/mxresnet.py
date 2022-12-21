from functools import partial

from .activations import Mish
from .net import Net


__all__ = ["mxresnet_parameters", "mxresnet34", "mxresnet50"]


mxresnet_parameters = {"stem_sizes": [3, 32, 64, 64], "act_fn": Mish()}
mxresnet34 = partial(
    Net, name="MXResnet32", expansion=1, layers=[3, 4, 6, 3], **mxresnet_parameters
)
mxresnet50 = partial(
    Net, name="MXResnet50", expansion=4, layers=[3, 4, 6, 3], **mxresnet_parameters
)
