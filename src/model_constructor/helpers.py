import importlib
from collections import OrderedDict
from functools import partial
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from pydantic import BaseModel
from torch import nn

ListStrMod = List[Tuple[str, nn.Module]]
ModSeq = Union[nn.Module, nn.Sequential]


def nn_seq(list_of_tuples: Iterable[Tuple[str, nn.Module]]) -> nn.Sequential:
    """return nn.Sequential from OrderedDict from list of tuples"""
    return nn.Sequential(OrderedDict(list_of_tuples))


def init_cnn(module: nn.Module) -> None:
    "Init module - kaiming_normal for Conv2d and 0 for biases."
    if getattr(module, "bias", None) is not None:
        nn.init.constant_(module.bias, 0)  # type: ignore
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight)
    for layer in module.children():
        init_cnn(layer)


def is_module(val: Any) -> bool:
    """Check if val is a nn.Module or partial of nn.Module."""

    to_check = val
    if isinstance(val, partial):
        to_check = val.func
    try:
        return issubclass(to_check, nn.Module)
    except TypeError:
        return False


def instantiate_module(
    name: str,
    default_path: Optional[str] = None,
) -> nn.Module:
    """Instantiate model from name."""
    if default_path is None:
        path_list = name.rsplit(".", 1)
        if len(path_list) == 1:
            default_path = "torch.nn"
            name = path_list[0]
        else:
            if path_list[0] == "nn":
                default_path = "torch.nn"
                name = path_list[1]
            else:
                default_path = path_list[0]
                name = path_list[1]
    try:
        mod = importlib.import_module(default_path)
    except ImportError as exc:
        raise ImportError(f"Module {default_path} not found") from exc
    if hasattr(mod, name):
        module = getattr(mod, name)
        if is_module(module):
            return module
        else:
            raise ImportError(f"Module {name} is not a nn.Module")
    else:
        raise ImportError(f"Module {name} not found at {default_path}")


class Cfg(BaseModel):
    """Base class for config."""

    name: Optional[str] = None

    def _get_str(self, value: Any) -> str:
        if isinstance(value, type):
            value = value.__name__
        elif isinstance(value, partial):
            value = f"{value.func.__name__} {value.keywords}"
        elif callable(value):
            value = value.__name__
        return value

    def _get_str_value(self, field: str) -> str:
        return self._get_str(getattr(self, field))

    def __repr__(self) -> str:
        return f"{self.__repr_name__()}(\n  {self.__repr_str__(chr(10) + '  ')})"

    def __str__(self) -> str:
        return f"{self.__repr_name__()}(\n  {self.__repr_str__(chr(10) + '  ')})"

    def __repr_args__(self) -> List[Tuple[str, str]]:
        return [
            (field, str_value)
            for field in self.model_fields
            if (str_value := self._get_str_value(field))
        ]

    def __repr_set_fields__(self) -> List[str]:
        """Return list repr for fields set at init"""
        return [
            f"{field}: {self._get_str_value(field)}"
            for field in self.model_fields_set  # pylint: disable=E1133:not-an-iterable
            if field != "name"
        ]

    def __repr_changed_fields__(self) -> List[str]:
        """Return list repr for changed fields"""
        return [
            f"{field}: {self._get_str_value(field)}"
            for field in self.changed_fields
            if field != "name"
        ]

    @property
    def changed_fields(self) -> Dict[str, Any]:
        # return "\n".join(self.__repr_changed_fields__())
        return {
            field: self._get_str_value(field)
            for field, value in self.model_fields.items()
            if getattr(self, field) != value.default
        }

    def print_cfg(self) -> None:
        """Print full config"""
        print(repr(self))

    def print_set_fields(self) -> None:
        """Print fields changed at init."""
        set_fields = self.__repr_set_fields__()
        if set_fields:
            print("Set fields:")
            for field in set_fields:
                print(field)
        else:
            print("Nothing changed")

    def print_changed_fields(self, show_default: bool = False, separator: str = " | ") -> None:
        """Print fields changed at init."""
        if self.changed_fields:
            default_value = ""
            print("Changed fields:")
            for field in self.changed_fields:
                if show_default:
                    default_value = f"{separator}{self._get_str(self.model_fields[field].default)}"
                print(f"{field}: {self._get_str_value(field)}{default_value}")
        else:
            print("Nothing changed")
