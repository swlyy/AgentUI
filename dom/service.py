import gc
import json
import logging
import os
from dataclasses import dataclass
from importlib import resources
from typing import Optional, Tuple, List
from urllib.parse import urlparse

from appium import webdriver

from dom.views import (
	DOMBaseNode,
	DOMElementNode,
	# DOMState,
	DOMTextNode,
	# SelectorMap,
)
from utils import time_execution_async

logger = logging.getLogger(__name__)

@dataclass
class ViewportInfo:
	# 屏幕尺寸
	width: int
	height: int

class DomService:
	def __init__(self, driver: Optional[webdriver.Remote]):
		self.driver = driver

	# region - Clickable elements
	# @time_execution_async('--get_clickable_elements')
	# async def get_clickable_elements(
	# 		self,
	# 		focus_element: int = -1,
	# 		viewport_expansion: int = 0,
	# ) -> DOMState:
	#
	# 	return DOMState(element_tree=element_tree, selector_map=selector_map)
