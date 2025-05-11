import asyncio
import enum
import json
import logging
import re
from telnetlib import EC
from typing import Dict, Generic, Optional, Type, TypeVar, List
import datetime

from appium.webdriver.common.appiumby import AppiumBy
from appium.webdriver.extensions.android.nativekey import AndroidKey
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import PromptTemplate

# from lmnr.sdk.laminar import Laminar
from pydantic.v1 import BaseModel
from selenium.webdriver.support.wait import WebDriverWait

from agent.views import ActionModel, ActionResult
from mobile.context import MobileContext
from controller.registry.service import Registry
from controller.views import (
    ClickElementAction,
    DoneAction,
    InputTextAction,
    NoParamsAction,
    SendKeysAction,
    WaitForElementAction,
)
from utils import time_execution_sync

logger = logging.getLogger(__name__)

Context = TypeVar('Context')


class Controller(Generic[Context]):
    def __init__(
            self,
            output_model: Optional[Type[BaseModel]] = None,
    ):
        self.registry = Registry[Context]()

        """Register all default mobile actions"""

        if output_model is not None:
            # Create a new model that extends the output model with success parameter
            class ExtendedOutputModel(BaseModel):  # type: ignore
                success: bool = True
                data: output_model

            @self.registry.action(
                'Complete task - with return text and if the task is finished (success=True) or not yet  completly finished (success=False), because last step is reached',
                param_model=ExtendedOutputModel,
            )
            async def done(params: ExtendedOutputModel):
                # Exclude success from the output JSON since it's an internal parameter
                output_dict = params.data.model_dump()

                # Enums are not serializable, convert to string
                for key, value in output_dict.items():
                    if isinstance(value, enum.Enum):
                        output_dict[key] = value.value

                return ActionResult(is_done=True, success=params.success, extracted_content=json.dumps(output_dict))
        else:

            @self.registry.action(
                'Complete task - with return text and if the task is finished (success=True) or not yet  completly finished (success=False), because last step is reached',
                param_model=DoneAction,
            )
            async def done(params: DoneAction):
                return ActionResult(is_done=True, success=params.success, extracted_content=params.text)

        # Basic Navigation Actions
        @self.registry.action('Go back', param_model=NoParamsAction)
        async def go_back(_: NoParamsAction, context: MobileContext):
            context.driver.back()
            msg = '🔙  Navigated back'
            logger.info(msg)
            return ActionResult(extracted_content=msg, include_in_memory=True)

        # wait for x seconds
        @self.registry.action('Wait for x seconds default 3')
        async def wait(seconds: int = 3):
            msg = f'🕒  Waiting for {seconds} seconds'
            logger.info(msg)
            await asyncio.sleep(seconds)
            return ActionResult(extracted_content=msg, include_in_memory=True)

        @self.registry.action('Wait for element to be visible', param_model=WaitForElementAction)
        async def wait_for_element(params: WaitForElementAction, context: MobileContext):
            """Waits for the element specified by the CSS selector to become visible within the given timeout."""

            if params.index not in await context.get_selector_map():
                raise Exception(f'Element with index {params.index} does not exist - retry or use alternative actions')

            element_node = await context.get_element_by_index(params.index)
            try:
                WebDriverWait(context.driver, params.timeout).until(
                    EC.visibility_of_element_located(
                        (AppiumBy.XPATH, element_node.xpath)
                    )
                )
                msg = f'👀  Element with index "{params.index}" became visible within {params.timeout}ms.'
                logger.info(msg)
                return ActionResult(extracted_content=msg, include_in_memory=True)
            except Exception as e:
                err_msg = f'❌  Failed to wait for element "{element_node.xpath}" within {params.timeout}ms: {str(e)}'
                logger.error(err_msg)
                raise Exception(err_msg)

        # Element Interaction Actions
        @self.registry.action('Click element by index', param_model=ClickElementAction)
        async def click_element_by_index(params: ClickElementAction, context: MobileContext):

            if params.index not in await context.get_selector_map():
                raise Exception(f'Element with index {params.index} does not exist - retry or use alternative actions')

            element_node = await context.get_element_by_index(params.index)
            try:
                await context._click_element_node(element_node)
                msg = f'🖱️  Clicked button with index {params.index}: {element_node.desc}'

                logger.info(msg)
                logger.debug(f'Element desc: {element_node.desc}')
                return ActionResult(extracted_content=msg, include_in_memory=True)
            except Exception as e:
                logger.warning(f'Element not clickable with index {params.index} - most likely the page changed')
                return ActionResult(error=str(e))

        @self.registry.action(
            'Input text into a input interactive element',
            param_model=InputTextAction,
        )
        async def input_text(params: InputTextAction, context: MobileContext):
            if params.index not in await context.get_selector_map():
                raise Exception(f'Element index {params.index} does not exist - retry or use alternative actions')

            element_node = await context.get_element_by_index(params.index)
            await context._input_text_element_node(element_node,params.text)
            msg = f'⌨️  Input {params.text} into index {params.index}'
            logger.info(msg)
            logger.debug(f'Element desc: {element_node.desc}')
            return ActionResult(extracted_content=msg, include_in_memory=True)

        # send keys
        @self.registry.action(
            'Send strings of special keys like Escape,Backspace, Insert, PageDown, Delete, Enter, Shortcuts such as `Control+o`, `Control+Shift+T` are supported as well. This gets used in keyboard.press. ',
            param_model=SendKeysAction,
        )
        async def send_keys(params: SendKeysAction, context: MobileContext):
            keycode = None
            if params.keys == 'Escape':     keycode=AndroidKey.ESCAPE
            elif params.keys == 'Backspace':     keycode = AndroidKey.BACK
            elif params.keys == 'Insert':       keycode = AndroidKey.INSERT
            elif params.keys == 'PageDown':     keycode = AndroidKey.PAGE_DOWN
            elif params.keys == 'Delete':       keycode = AndroidKey.DEL
            elif params.keys == 'Enter':        keycode = AndroidKey.ENTER

            try:
                await context.driver.press_keycode(keycode)
            except Exception as e:
                if 'Unknown key' in str(e):
                    # loop over the keys and try to send each one
                    for key in params.keys:
                        try:
                            await context.driver.press_keycode(keycode)
                        except Exception as e:
                            logger.debug(f'Error sending key {key}: {str(e)}')
                            raise e
                else:
                    raise e
            msg = f'⌨️  Sent keys: {params.keys}'
            logger.info(msg)
            return ActionResult(extracted_content=msg, include_in_memory=True)

        @self.registry.action(
            description='If you dont find something which you want to interact with, scroll to it',
        )
        async def scroll_to_text(text: str, context: MobileContext):  # type: ignore
            screen_size = context.driver.get_window_size()
            start_x = screen_size['width'] * 0.5
            start_y = screen_size['height'] * 0.8
            end_x = screen_size['width'] * 0.5
            end_y = screen_size['height'] * 0.2

            for _ in range(10):
                if text not in context.driver.page_source:
                    await context.driver.swipe(start_x, start_y, end_x, end_y, 800)
                    continue
                msg = f'🔍  Scrolled to text: {text}'
                logger.info(msg)
                return ActionResult(extracted_content=msg, include_in_memory=True)
            msg = f"Text '{text}' not found or not visible on page"
            logger.info(msg)
            return ActionResult(extracted_content=msg, include_in_memory=True)

    # Register ---------------------------------------------------------------

    def action(self, description: str, **kwargs):
        """Decorator for registering custom actions

        @param description: Describe the LLM what the function does (better description == better function calling)
        """
        return self.registry.action(description, **kwargs)

    # Act --------------------------------------------------------------------

    @time_execution_sync('--act')
    async def act(
            self,
            action: ActionModel,
            mobile_context: MobileContext,
    ) -> ActionResult:
        """Execute an action"""

        try:
            for action_name, params in action.dict(exclude_unset=True).items():
                if params is not None:
                    result = await self.registry.execute_action(
                        action_name,
                        params,
                        mobile_context
                    )
                    if isinstance(result, str):
                        return ActionResult(extracted_content=result)
                    elif isinstance(result, ActionResult):
                        return result
                    elif result is None:
                        return ActionResult()
                    else:
                        raise ValueError(f'Invalid action result type: {type(result)} of {result}')
            return ActionResult()
        except Exception as e:
            raise e