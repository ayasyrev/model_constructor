from torch import nn


def value_name(value) -> str:  # pragma: no cover
    name = getattr(value, "__name__", None)
    if name is not None:
        return name
    if isinstance(value, nn.Module):
        return value._get_name()  # pylint: disable=W0212
    return value


def ids_fn(key, value):
    return [f"{key[:2]}_{value_name(v)}" for v in value]
