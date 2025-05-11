from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from dom.history_tree_processor.view import CoordinateSet, HashedDomElement, ViewportInfo, Coordinates
from utils import time_execution_sync

# Avoid circular import issues
if TYPE_CHECKING:
	from .views import DOMElementNode


@dataclass(frozen=False)
class DOMBaseNode:
	is_visible: bool
	# Use None as default and set parent later to avoid circular reference issues
	parent: Optional['DOMElementNode']


@dataclass(frozen=False)
class DOMTextNode(DOMBaseNode):
	text: str
	component_type: str = 'TEXT_COMPONENT'  # 组件类型（如android.widget.TextView）


@dataclass(frozen=False)
class DOMElementNode:
	"""
	xpath: the xpath of the element from the last root node (shadow root or iframe OR document if no shadow root or iframe).
	To properly reference the element we need to recursively switch the root node until we find the element (work you way up the tree with `.parent`)
	"""
	desc: str # 描述
	component_type: str  # 组件类型（如Button/TextView）
	text: str  # 基于移动端层级生成的XPath
	attributes: Dict[str, str]  # 移动端属性（resource-id、content-desc等）
	is_interactive: bool = False  # 是否可交互（基于clickable/focusable）
	is_in_viewport: bool = False  # 是否在可视区域（需计算坐标）

	viewport_coordinates: Optional[Coordinates] = None
	page_coordinates: Optional[CoordinateSet] = None
	viewport_info: Optional[ViewportInfo] = None

	def __repr__(self) -> str:
		tag_str = f'<{self.component_type}<{self.text}<{self.desc}'
		return tag_str

	@cached_property
	def hash(self) -> HashedDomElement:
		from dom.history_tree_processor.service import (
			HistoryTreeProcessor,
		)
		return HistoryTreeProcessor._hash_dom_element(self)