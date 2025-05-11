from dataclasses import dataclass, field
from typing import Any, Optional, Union, List, Dict

from pydantic.v1 import BaseModel

from dom.history_tree_processor.service import DOMHistoryElement
from dom.views import DOMElementNode

SelectorMap = Dict[int, DOMElementNode]


@dataclass
class MobileState:
	app_package: str
	selector_map: SelectorMap
	screenshot: Optional[str] = None
	# vertical_offset: int = 0
	# horizontal_offset: int = 0
	errors_messages: List[str] = field(default_factory=list)


@dataclass
class MobileStateHistory:
	app_package: str
	# interacted_element: Union[List[Union[DOMHistoryElement, None]], List[None]]
	screenshot: Optional[str] = None

	def to_dict(self) -> Dict[str, Any]:
		data = dict()
		data['screenshot'] = self.screenshot
		# data['interacted_element'] = [el.to_dict() if el else None for el in self.interacted_element]
		data['app_package'] = self.app_package
		return data


class MobileError(Exception):
	"""Base class for all mobile errors"""