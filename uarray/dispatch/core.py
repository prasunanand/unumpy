import collections
import collections.abc
import contextvars
import dataclasses
import functools
import typing

__all__ = [
    "Box",
    "copy",
    "concrete",
    "map_children",
    "global_context",
    "ReplacementType",
    "ContextType",
    "KeyType",
    "children",
    "replace_inplace_generator",
    "key",
    "ChildrenType",
    "replace",
    "ChainCallable",
    "MapChainCallable",
]

T = typing.TypeVar("T")
T_cov = typing.TypeVar("T_cov", covariant=True)
T_box = typing.TypeVar("T_box", bound="Box")


@dataclasses.dataclass
class Box(typing.Generic[T_cov]):
    value: T_cov = typing.cast(T_cov, None)

    def replace(self: T_box, value: typing.Any = None) -> "T_box":
        return dataclasses.replace(self, value=value)

    @classmethod
    def _tuple_fields(cls) -> typing.Tuple[str, ...]:
        return tuple(
            f.name for f in dataclasses.fields(cls) if f.init and f.name != "value"
        )

    def _str_without_value(self) -> str:
        return f"{type(self).__qualname__}({', '.join(f'{f}={getattr(self, f)._str_without_value()}' for f in self._tuple_fields())})"


ReplacementType = typing.Callable[[Box], Box]


ChildrenType = typing.Sequence[Box]


@functools.singledispatch
def children(node) -> ChildrenType:
    return ()


@functools.singledispatch
def map_children(v: T, fn: typing.Callable[[typing.Any], typing.Any]) -> T:
    return v


KeyType = object


@functools.singledispatch
def key(node) -> KeyType:
    return type(node)


@functools.singledispatch
def copy(v: T, already_copied: typing.MutableMapping) -> T:
    if id(v) in already_copied:
        return already_copied[id(v)]
    new = map_children(v, lambda a: copy(a, already_copied))
    already_copied[id(v)] = new
    return new


@copy.register
def copy_box(v: Box, already_copied: typing.MutableMapping) -> Box:
    if id(v) in already_copied:
        return already_copied[id(v)]
    new = v.replace(copy(v.value, already_copied))
    already_copied[id(v)] = new
    return new


@functools.singledispatch
def concrete(x: typing.Any) -> bool:
    return True


ContextType = typing.MutableMapping[KeyType, ReplacementType]


class ChainCallable:
    """
    Like ChainMap but for callables. Combines a bunch of functions
    into one function, where each are tried in order on the arguments
    until one returns something other than `NotImplemented`.
    """

    def __init__(self, *callables: ReplacementType):
        self.callables = list(callables)

    def __call__(self, arg: Box) -> Box:
        for callable in self.callables:
            res = callable(arg)
            if res is not NotImplemented:
                return res
        return NotImplemented


class MapChainCallable(collections.abc.MutableMapping):
    """
    Mutable mapping of keys to a list of replacements.

    Setting a key adds the new replacement to the list, instead of overriding
    the existing ones.
    """

    def __init__(self):
        self.dict: typing.MutableMapping[
            KeyType, ChainCallable
        ] = collections.defaultdict(ChainCallable)

    def __setitem__(self, key: KeyType, value: ReplacementType) -> None:
        self.dict[key].callables = [value] + self.dict[key].callables

    def __getitem__(self, key: KeyType) -> ReplacementType:
        return self.dict[key]

    def __delitem__(self, key: KeyType) -> None:
        del self.dict[key]

    def __iter__(self) -> typing.Iterator[KeyType]:
        return iter(self.dict)

    def __len__(self) -> int:
        return len(self.dict)


global_context: contextvars.ContextVar[ContextType] = contextvars.ContextVar(
    "uarray.dispatch.global_context", default=MapChainCallable()
)


def replace(box: T_box) -> T_box:
    box = copy(box, {})
    for _ in replace_inplace_generator(box):
        pass
    return box


def replace_inplace_generator(box: Box) -> typing.Iterator[Box]:
    """
    Keeps calling replacemnts on the node, or it's children, until no more match.

    Returns a sequence of the just replaced box.
    """
    while True:
        replaced_box = replace_once_inplace(box)
        if replaced_box is None:
            return
        yield replaced_box


class NotBox(Exception):
    pass


def replace_once_inplace(box: Box) -> typing.Optional[Box]:
    """
    Returns the replaced box, or None if no boxes could be found to replace.
    """
    if not isinstance(box, Box):
        raise NotBox(f"Can only replace boxes, not {type(box)}")
    for child in children(box.value):
        try:
            replaced_box = replace_once_inplace(child)
        except NotBox:
            raise NotBox(f"Not box on child {child}")
        if replaced_box:
            return replaced_box

    context = global_context.get()
    try:
        replacement = context[key(box.value)]
    except KeyError:
        # no replacements registered for node
        return None
    new_box = replacement(box)

    if new_box == NotImplemented:
        return None

    box.value = new_box.value
    return box