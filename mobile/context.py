"""
Playwright browser on steroids.
"""
import cv2
import asyncio
import base64
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, TypedDict, Union, List, Tuple, Dict

import requests
from PIL import Image
import io
from appium import webdriver
from appium.webdriver.common.appiumby import AppiumBy
from langchain_core.messages import SystemMessage, BaseMessage, HumanMessage

from dom.history_tree_processor.view import CoordinateSet, Coordinates
from mobile.mobile import Mobile

from mobile.views import (
    MobileError,
    MobileState, SelectorMap
)
from dom.service import DomService
from dom.views import DOMElementNode
from utils import time_execution_async, time_execution_sync
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


class APPContextWindowSize(TypedDict):
    width: int
    height: int


@dataclass
class MobileContextConfig:

    minimum_wait_page_load_time: float = 0.25
    maximum_wait_page_load_time: float = 5
    wait_between_actions: float = 0.5

    app_window_size: APPContextWindowSize = field(default_factory=lambda: {'width': 1280, 'height': 1100})

    viewport_expansion: int = 500
    highlight_elements: bool = True


class MobileContext:
    def __init__(
            self,
            mobile: Optional['Mobile'],
            config: MobileContextConfig = MobileContextConfig(),
    ):
        self.context_id = str(uuid.uuid4())
        logger.debug(f'Initializing new mobile context with id: {self.context_id}')
        self.mobile = mobile
        self.config = config
        self.driver = self.mobile.start()

    async def _update_state(self) -> MobileState:
        """Update and return state."""
        try:
            # content 是返回的 DomState 内容
            self.screenshot_b64 = await self.take_screenshot()
            self.page_source = self.driver.page_source
            time.sleep(2)
            self.driver.save_screenshot("screenshot.png")
            self.vision_data = self.get_vision_data()
            self.text_data = self.get_text_data()
            self.selector_map = await self.update_selector_map()
            # vertical_offset, horizontal_offset = await self.get_scroll_info(self.driver)
            self.current_state = MobileState(
                app_package = self.driver.current_package,
                selector_map=self.selector_map,
                screenshot=self.screenshot_b64,
                # vertical_offset=vertical_offset,
                # horizontal_offset=horizontal_offset,
            )
            # print('------------------------------更新成功--------------------------------')
            # print(self.current_state.selector_map)
            return self.current_state
        except Exception as e:
            logger.error(f'Failed to update state: {str(e)}')
            # Return last known good state if available
            if hasattr(self, 'current_state'):
                return self.current_state
            raise

    @time_execution_sync('--get_state')  # This decorator might need to be updated to handle async
    async def get_state(self) -> MobileState:
        """Get the current state of the mobile"""
        state = await self._update_state()

        return state

    # region - Browser Actions
    @time_execution_async('--take_screenshot')
    async def take_screenshot(self, full_page: bool = False) -> str:
        driver = self.driver
        screenshot_b64 = driver.get_screenshot_as_base64().encode('utf-8')
        return screenshot_b64

    def get_vision_data(self):
        pic_path = "screenshot.png"
        image = cv2.imread(pic_path)
        image_b = base64.b64encode(cv2.imencode(".png", image)[1].tobytes())
        image_source = bytes.decode(image_b)
        ui_infer_url = f"http://127.0.0.1:9092/vision/ui-infer"
        params = {"type": "base64", "image": image_source}
        response = requests.post(url=ui_infer_url, json=params)
        result = response.json()
        return result.get("data", [])

    def get_text_data(self):
        ui_text_url = f"http://127.0.0.1:9092/vision/text"
        params = {"image":"screenshot.png"}
        response = requests.post(url=ui_text_url, json=params)
        result = response.json()
        return result.get("data", [])

    async def update_selector_map(self) -> Dict[int, DOMElementNode]:
        input_message = [HumanMessage(
            content=f"""你是一个专业的UI元素分析师，需要根据截图中的文字识别结果{self.text_data}完成：
        			         使用json格式和中文输出结果，每个元素都要包含元素坐标、类型、元素的文字内容和功能描述，结果返回一个包含多个元素的列表
        					 功能描述示例输出：'elem_det_pos':[64, 36],
        									 'elem_det_type':'button',
        									 'text:'外卖',
        									 'description':'功能按钮，点击进入外卖相关页面'
        					 """
        )]
        llm = ChatOpenAI(
            model_name='deepseek-chat',
            openai_api_key='sk-3823750772cb4120ac390018fb0f6a1e',
            openai_api_base="https://api.deepseek.com",
        )
        vision_data = llm.invoke(input_message).content
        json_str = vision_data.split('```json\n')[1].split('\n```')[0]  # 提取 JSON 部分
        vision_data = json.loads(json_str)
        self.vision_data=vision_data
        selector_map: Dict[int, DOMElementNode] = {}
        for i in range(len(self.vision_data)):
            pos = vision_data[i]["elem_det_pos"]
            x, y = pos[0], pos[1]
            elementNode = DOMElementNode(
                desc = vision_data[i]["description"],
                component_type = vision_data[i]["elem_det_type"],
                text = vision_data[i]["text"],
                attributes = {},
                viewport_coordinates=Coordinates(
                    x=x,
                    y=y
                )
            )
            selector_map[i] = elementNode
        return selector_map

    # def upload_file_to_gitee(self):
    #     # 本地仓库路径（如果不存在会自动初始化）
    #     repo_path = "./meituan_capture"
    #     remote_url = "https://gitee.com/suwenlyy/Test_pytest.git"  # Gitee 仓库地址
    #     branch_name = "master"  # 分支名（默认可能是 master 或 main）
    #
    #     # 1. 初始化或克隆仓库
    #     if not os.path.exists(repo_path):
    #         repo = Repo.clone_from(remote_url, repo_path, branch=branch_name)
    #     else:
    #         repo = Repo(repo_path)
    #
    #     # 2. 添加文件（示例：创建一个新文件）
    #     image_file = "screenshot.png"
    #
    #     if not os.path.exists(os.path.join(repo_path, image_file)):
    #         raise FileNotFoundError(f"文件 {image_file} 不存在！")
    #
    #     # 3. 提交更改
    #     repo.git.add(image_file)
    #     repo.git.commit("-m", "Add single image: screenshot.png")  # 提交
    #
    #     # 4. 推送至 Gitee（需提前配置 SSH 或 HTTPS 认证）
    #     repo.git.push("origin", branch_name)

    # async def get_desc(self, vision_data, screenshot_b64):
    #     base_url = 'https://api.deepseek.com/v1/chat/completions'
    #     api_key = 'sk-3823750772cb4120ac390018fb0f6a1e'
    #     llm = ChatOpenAI(
    #         model_name='deepseek-chat',
    #         openai_api_key=api_key,
    #         openai_api_base=base_url,
    #         # max_tokens=1024,
    #     )
    #     system_prompt = f"""你是一个专业的UI元素分析师，需要根据输入的视觉检测数据:{vision_data}和截图:{screenshot_b64}完成：
    #     1. 识别每个元素的类型（icon/bg等）和功能描述
    #     2. 按检测概率从高到低排序
    #     3. 使用json格式输出结果，包含区域坐标、类型和文字描述"""
    #
    #     # input_messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]
    #     input_messages = [{"role": "user", "content": system_prompt}]
    #     output_data = llm.invoke(input=system_prompt)
    #     return output_data

    # endregion

    # region - User Actions

    @time_execution_async('--input_text_element_node')
    async def _input_text_element_node(self, element_node: DOMElementNode, text: str):
        """
        Input text into an element with proper error handling and state management.
        Handles different types of input fields and ensures proper element state before input.
        """
        try:
            # Highlight before typing
            # if element_node.highlight_index is not None:
            # 	await self._update_state(focus_element=element_node.highlight_index)

            # element_handle = await self.get_locate_element(element_node)
            #
            # if element_handle is None:
            #     raise MobileError(f'Element: {repr(element_node)} not found')

            # Ensure element is ready for input
            try:
                await element_node.wait_for_element_state('stable', timeout=1000)
                await element_node.scroll_into_view_if_needed(timeout=1000)
            except Exception:
                pass
            self.driver.f(
                [(element_node.viewport_coordinates.bottom_left.x, element_node.viewport_coordinates.bottom_left.y),
                 (element_node.viewport_coordinates.top_right.x, element_node.viewport_coordinates.top_right.y)], 1000)

            await element_node.fill(text)

            # Get element properties to determine input method
            # tag_handle = await element_handle.get_property("tagName")
            # tag_name = (await tag_handle.json_value()).lower()
            # is_contenteditable = await element_handle.get_property('isContentEditable')
            # readonly_handle = await element_handle.get_property("readOnly")
            # disabled_handle = await element_handle.get_property("disabled")
            #
            # readonly = await readonly_handle.json_value() if readonly_handle else False
            # disabled = await disabled_handle.json_value() if disabled_handle else False
            #
            # if (await is_contenteditable.json_value() or tag_name == 'input') and not (readonly or disabled):
            #     await element_handle.evaluate('el => el.textContent = ""')
            #     await element_handle.type(text, delay=5)
            # else:
            #     await element_handle.fill(text)

        except Exception as e:
            logger.debug(f'Failed to input text into element: {repr(element_node)}. Error: {str(e)}')
            raise MobileError(f'Failed to input text into index {element_node.highlight_index}')

    @time_execution_async('--click_element_node')
    async def _click_element_node(self, element_node: DOMElementNode) -> Optional[str]:
        """
        Optimized method to click an element using xpath.
        """
        # if self.driver.find_elements(AppiumBy.XPATH, element_node.xpath):
        #     raise Exception(f'Element: {repr(element_node)} not found')

        try:
            time.sleep(3)
            self.driver.tap([(element_node.viewport_coordinates.x,element_node.viewport_coordinates.y)], 500)
        except Exception as e:
            raise Exception(f'Failed to click element: {repr(element_node)}. Error: {str(e)}')

    # endregion

    # region - Helper methods for easier access to the DOM

    async def get_selector_map(self) -> SelectorMap:
        if self.mobile is None:
            return {}
        return self.current_state.selector_map

    async def get_element_by_index(self, index: int) -> DOMElementNode:
        selector_map = await self.get_selector_map()
        return selector_map[index]

    # async def get_scroll_info(self, driver: Optional[webdriver.Remote]) -> Tuple[int, int]:
    #     """Get scroll position information for the current page."""
    #
    #     window_size = driver.get_window_size()
    #     # print('------------------------------成功--------------------------------')
    #     viewport_height = window_size['height']
    #     # print('------------------------------成功--------------------------------')
    #     # total_height = driver.execute_script('mobile: getViewportSize').get('height')
    #     # print('------------------------------成功--------------------------------')
    #     # scroll_y = driver.execute_script('mobile: getScrollOffset')['y']
    #     scroll_container = driver.find_element(
    #         AppiumBy.ANDROID_UIAUTOMATOR,
    #         'new UiScrollable(new UiSelector().scrollable(true))'
    #     )
    #     # print('------------------------------成功--------------------------------')
    #     first_child = scroll_container.find_element(AppiumBy.ANDROID_UIAUTOMATOR, 'new UiSelector().index(0)')
    #     print('------------------------------成功--------------------------------')
    #     last_child = scroll_container.find_element(AppiumBy.ANDROID_UIAUTOMATOR, 'new UiSelector().index(-1)')
    #     print('------------------------------成功--------------------------------')
    #     total_height = last_child.rect['y'] + last_child.rect['height'] - first_child.rect['y']
    #     print('------------------------------成功--------------------------------')
    #
    #     scroll_y = int(scroll_container.get_attribute("scrollY"))
    #     print('------------------------------成功--------------------------------')
    #     vertical_offset = scroll_y
    #     horizontal_offset = total_height - (scroll_y + viewport_height)
    #     return vertical_offset, horizontal_offset

    async def _get_unique_filename(self, directory, filename):
        """Generate a unique filename by appending (1), (2), etc., if a file already exists."""
        base, ext = os.path.splitext(filename)
        counter = 1
        new_filename = filename
        while os.path.exists(os.path.join(directory, new_filename)):
            new_filename = f'{base} ({counter}){ext}'
            counter += 1
        return new_filename