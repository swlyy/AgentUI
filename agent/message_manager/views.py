from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING, Any, List

from langchain_core.load import dumpd, load
from langchain_core.messages import BaseMessage, HumanMessage
# from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic.v1 import BaseModel, ConfigDict, Field

from warnings import filterwarnings
from langchain_core._api import LangChainBetaWarning
filterwarnings("ignore", category=LangChainBetaWarning)

if TYPE_CHECKING:
	from agent.views import AgentOutput


class MessageMetadata(BaseModel):
	"""Metadata for a message"""

	tokens: int = 0


class ManagedMessage(BaseModel):
	"""A message with its metadata"""

	message: BaseMessage
	metadata: MessageMetadata = Field(default_factory=MessageMetadata)

	model_config = ConfigDict(arbitrary_types_allowed=True)


class MessageHistory(BaseModel):
	"""History of messages with metadata"""

	messages: List[ManagedMessage] = []
	current_tokens: int = 0

	model_config = ConfigDict(arbitrary_types_allowed=True)

	def add_message(self, message: BaseMessage, metadata: MessageMetadata, position: int | None = None) -> None:
		"""Add message with metadata to history"""
		if position is None:
			self.messages.append(ManagedMessage(message=message, metadata=metadata))
		else:
			self.messages.insert(position, ManagedMessage(message=message, metadata=metadata))
		self.current_tokens += metadata.tokens

	def remove_last_state_message(self) -> None:
		"""Remove last state message from history"""
		if len(self.messages) > 2 and isinstance(self.messages[-1].message, HumanMessage):
			self.current_tokens -= self.messages[-1].metadata.tokens
			self.messages.pop()


class MessageManagerState(BaseModel):
	"""Holds the state for MessageManager"""

	history: MessageHistory = Field(default_factory=MessageHistory)
	tool_id: int = 1

	model_config = ConfigDict(arbitrary_types_allowed=True)