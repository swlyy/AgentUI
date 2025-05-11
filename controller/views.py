
from typing import Optional

from pydantic import model_validator
from pydantic.v1 import BaseModel


# Action Input Models

class WaitForElementAction(BaseModel):
	index: int
	timeout: Optional[int] = 10000  # Timeout in milliseconds


class ClickElementAction(BaseModel):
	index: int
	xpath: Optional[str] = None


class InputTextAction(BaseModel):
	index: int
	text: str
	xpath: Optional[str] = None


class DoneAction(BaseModel):
	text: str
	success: bool


class ScrollAction(BaseModel):
	amount: Optional[int] = None  # The number of pixels to scroll. If None, scroll down/up one page


class SendKeysAction(BaseModel):
	keys: str


class NoParamsAction(BaseModel):
	"""
	Accepts absolutely anything in the incoming data
	and discards it, so the final parsed model is empty.
	"""

	class Config:
		# If you want to silently allow unknown fields at top-level,
		# set extra = 'allow' as well:
		extra = 'allow'
