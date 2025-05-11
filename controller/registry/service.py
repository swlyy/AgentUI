import asyncio
from inspect import iscoroutinefunction, signature
from typing import Any, Callable, Dict, Generic, Optional, Type, TypeVar, Union, List

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic.v1 import BaseModel, Field, create_model

from mobile.context import MobileContext
from controller.registry.views import (
	ActionModel,
	ActionRegistry,
	RegisteredAction,
)
from utils import time_execution_async, time_execution_sync

Context = TypeVar('Context')


class Registry(Generic[Context]):
	"""Service for registering and managing actions"""

	def __init__(self):
		self.registry = ActionRegistry()

	@time_execution_sync('--create_param_model')
	def _create_param_model(self) -> Type[BaseModel]:
		"""Creates a Pydantic model from function signature"""
		# sig = signature(function)
		# params = {
		# 	name: (param.annotation, ... if param.default == param.empty else param.default)
		# 	for name, param in sig.parameters.items()
		# }
		# TODO: make the types here work
		return create_model(
			'create_param_model_parameters',
			__base__=ActionModel,
			# **params,  # type: ignore
		)

	def action(
		self,
		description: str,
		param_model: Optional[Type[BaseModel]] = None,
	):
		"""Decorator for registering actions"""

		def decorator(func: Callable):
			# Create param model from function if not provided
			actual_param_model = param_model or self._create_param_model()

			# Wrap sync functions to make them async
			if not iscoroutinefunction(func):

				async def async_wrapper(*args, **kwargs):
					return await asyncio.to_thread(func, *args, **kwargs)

				# Copy the signature and other metadata from the original function
				async_wrapper.__signature__ = signature(func)
				async_wrapper.__name__ = func.__name__
				async_wrapper.__annotations__ = func.__annotations__
				wrapped_func = async_wrapper
			else:
				wrapped_func = func

			action = RegisteredAction(
				name=func.__name__,
				description=description,
				function=wrapped_func,
				param_model=actual_param_model,
			)
			self.registry.actions[func.__name__] = action
			return func

		return decorator

	@time_execution_async('--execute_action')
	async def execute_action(
		self,
		action_name: str,
		params: dict,
		content: MobileContext
	) -> Any:
		"""Execute a registered action"""
		if action_name not in list(self.registry.actions.keys()):
			raise ValueError(f'Action {action_name} not found')

		action = self.registry.actions[action_name]
		try:
			# Create the validated Pydantic model
			validated_params = action.param_model(**params)

			# Check if the first parameter is a Pydantic model
			sig = signature(action.function)
			parameters = list(sig.parameters.values())
			is_pydantic = parameters and issubclass(parameters[0].annotation, BaseModel)

			# Prepare arguments based on parameter type
			extra_args = {}
			if is_pydantic:
				if action_name=='done':
					return await action.function(validated_params, **extra_args)
				return await action.function(validated_params,context=content, **extra_args)
			return await action.function(**validated_params.dict(), **extra_args)

		except Exception as e:
			raise RuntimeError(f'Error executing action {action_name}: {str(e)}') from e

	@time_execution_sync('--create_action_model')
	def create_action_model(self) -> Type[ActionModel]:
		"""Creates a Pydantic model from registered actions, used by LLM APIs that support tool calling & enforce a schema"""

		available_actions = self.registry.actions

		fields = {
			name: (
				Optional[action.param_model],
				Field(default=None, description=action.description),
			)
			for name, action in available_actions.items()
		}

		return create_model('ActionModel', __base__=ActionModel, **fields)  # type:ignore

	def get_prompt_description(self) -> str:
		"""Get a description of all actions for the prompt

		If page is provided, only include actions that are available for that page
		based on their filter_func
		"""
		return self.registry.get_prompt_description()