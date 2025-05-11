from __future__ import annotations

import json
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Type, Union

from langchain_core.language_models.chat_models import BaseChatModel
from openai import RateLimitError
from pydantic.v1 import BaseModel, ConfigDict, Field, ValidationError, create_model

from agent.message_manager.views import MessageManagerState
from mobile.views import MobileStateHistory, SelectorMap
from controller.registry.views import ActionModel
from dom.history_tree_processor.service import (
    DOMElementNode,
    DOMHistoryElement,
    HistoryTreeProcessor,
)

ToolCallingMethod = Literal['function_calling', 'json_mode', 'raw', 'auto']


class AgentSettings(BaseModel):
    """Options for the agent"""
    use_vision: bool = True
    use_vision_for_planner: bool = False
    save_conversation_path: Optional[str] = None
    save_conversation_path_encoding: Optional[str] = 'utf-8'
    max_failures: int = 3
    retry_delay: int = 10
    max_input_tokens: int = 128000
    validate_output: bool = False
    message_context: Optional[str] = None
    generate_gif: Union[bool, str] = False
    available_file_paths: Optional[List[str]] = None
    override_system_message: Optional[str] = None
    extend_system_message: Optional[str] = None

    max_actions_per_step: int = 10

    tool_calling_method: Optional[ToolCallingMethod] = 'auto'
    page_extraction_llm: Optional[BaseChatModel] = None
    planner_llm: Optional[BaseChatModel] = None
    planner_interval: int = 1  # Run planner every N steps


class AgentState(BaseModel):
    """Holds all state information for an Agent"""

    agent_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    n_steps: int = 1
    consecutive_failures: int = 0
    last_result: Optional[List['ActionResult']] = None
    history: AgentHistoryList = Field(default_factory=lambda: AgentHistoryList(history=[]))
    last_plan: Optional[str] = None
    paused: bool = False
    stopped: bool = False

    message_manager_state: MessageManagerState = Field(default_factory=MessageManagerState)


@dataclass
class AgentStepInfo:
    step_number: int
    max_steps: int

    def is_last_step(self) -> bool:
        """Check if this is the last step"""
        return self.step_number >= self.max_steps - 1


class ActionResult(BaseModel):
    """Result of executing an action"""

    is_done: Optional[bool] = False
    success: Optional[bool] = None
    extracted_content: Optional[str] = None
    error: Optional[str] = None
    include_in_memory: bool = False  # whether to include in past messages as context or not


class StepMetadata(BaseModel):
    """Metadata for a single step including timing and token information"""

    step_start_time: float
    step_end_time: float
    input_tokens: int  # Approximate tokens from message manager for this step
    step_number: int

    @property
    def duration_seconds(self) -> float:
        """Calculate step duration in seconds"""
        return self.step_end_time - self.step_start_time


class AgentBrain(BaseModel):
    """Current state of the agent"""

    evaluation_previous_goal: str
    memory: str
    next_goal: str


class AgentOutput(BaseModel):
    """Output model for agent

    @dev note: this model is extended with custom actions in AgentService. You can also use some fields that are not in this model as provided by the linter, as long as they are registered in the DynamicActions model.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    current_state: AgentBrain
    action: List[ActionModel] = Field(
        ...,
        description='List of actions to execute',
        json_schema_extra={'min_items': 1},  # Ensure at least one action is provided
    )

    @staticmethod
    def type_with_custom_actions(custom_actions: Type[ActionModel]) -> Type['AgentOutput']:
        """Extend actions with custom actions"""
        model_ = create_model(
            'AgentOutput',
            __base__=AgentOutput,
            action=(
                List[custom_actions],
                Field(..., description='List of actions to execute', json_schema_extra={'min_items': 1}),
            ),
            __module__=AgentOutput.__module__,
        )
        model_.__doc__ = 'AgentOutput model with custom actions'
        return model_


class AgentHistory(BaseModel):
    """History item for agent actions"""

    model_output: Union[AgentOutput, None]
    result: List[ActionResult]
    state: MobileStateHistory
    metadata: Optional[StepMetadata] = None

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    @staticmethod
    def get_interacted_element(model_output: AgentOutput, selector_map: SelectorMap) -> List[Union[DOMHistoryElement, None]]:
        elements = []
        for action in model_output.action:
            index = action.get_index()
            if index is not None and index in selector_map:
                el: DOMElementNode = selector_map[index]
                elements.append(HistoryTreeProcessor.convert_dom_element_to_history_element(el))
            else:
                elements.append(None)
        return elements

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Custom serialization handling circular references"""

        # Handle action serialization
        model_output_dump = None
        if self.model_output:
            action_dump = [action.dict(exclude_none=True) for action in self.model_output.action]
            model_output_dump = {
                'current_state': self.model_output.current_state.dict(),
                'action': action_dump,  # This preserves the actual action data
            }

        return {
            'model_output': model_output_dump,
            'result': [r.dict(exclude_none=True) for r in self.result],
            'state': self.state.to_dict(),
            'metadata': self.metadata.dict() if self.metadata else None,
        }


class AgentHistoryList(BaseModel):
    """List of agent history items"""

    history: List[AgentHistory]

    def total_duration_seconds(self) -> float:
        """Get total duration of all steps in seconds"""
        total = 0.0
        for h in self.history:
            if h.metadata:
                total += h.metadata.duration_seconds
        return total

    def total_input_tokens(self) -> int:
        """
        Get total tokens used across all steps.
        Note: These are from the approximate token counting of the message manager.
        For accurate token counting, use tools like LangChain Smith or OpenAI's token counters.
        """
        total = 0
        for h in self.history:
            if h.metadata:
                total += h.metadata.input_tokens
        return total

    def __str__(self) -> str:
        """Representation of the AgentHistoryList object"""
        return f'AgentHistoryList(all_results={self.action_results()}, all_model_outputs={self.model_actions()})'

    def __repr__(self) -> str:
        """Representation of the AgentHistoryList object"""
        return self.__str__()

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Custom serialization that properly uses AgentHistory's model_dump"""
        return {
            'history': [h.model_dump(**kwargs) for h in self.history],
        }

    def is_done(self) -> bool:
        """Check if the agent is done"""
        if self.history and len(self.history[-1].result) > 0:
            last_result = self.history[-1].result[-1]
            return last_result.is_done is True
        return False

    def is_successful(self) -> Union[bool, None]:
        """Check if the agent completed successfully - the agent decides in the last step if it was successful or not. None if not done yet."""
        if self.history and len(self.history[-1].result) > 0:
            last_result = self.history[-1].result[-1]
            if last_result.is_done is True:
                return last_result.success
        return None

    # get all actions with params
    def model_actions(self) -> List[dict]:
        """Get all actions from history"""
        outputs = []

        for h in self.history:
            if h.model_output:
                for action, interacted_element in zip(h.model_output.action, h.state.interacted_element):
                    output = action.model_dump(exclude_none=True)
                    output['interacted_element'] = interacted_element
                    outputs.append(output)
        return outputs

    def action_results(self) -> List[ActionResult]:
        """Get all results from history"""
        results = []
        for h in self.history:
            results.extend([r for r in h.result if r])
        return results

    def extracted_content(self) -> list[str]:
        """Get all extracted content from history"""
        content = []
        for h in self.history:
            content.extend([r.extracted_content for r in h.result if r.extracted_content])
        return content


class AgentError:
    """Container for agent error handling"""

    VALIDATION_ERROR = 'Invalid model output format. Please follow the correct schema.'
    RATE_LIMIT_ERROR = 'Rate limit reached. Waiting before retry.'
    NO_VALID_ACTION = 'No valid action found'

    @staticmethod
    def format_error(error: Exception, include_trace: bool = False) -> str:
        """Format error message based on error type and optionally include trace"""
        message = ''
        if isinstance(error, ValidationError):
            return f'{AgentError.VALIDATION_ERROR}\nDetails: {str(error)}'
        if isinstance(error, RateLimitError):
            return AgentError.RATE_LIMIT_ERROR
        if include_trace:
            return f'{str(error)}\nStacktrace:\n{traceback.format_exc()}'
        return f'{str(error)}'
