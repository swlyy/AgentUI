# from agent.service import Agent
import base64
import logging
import os
from io import BytesIO
from typing import Dict, Optional

import dotenv
import requests
from logging_config import setup_logging
from langchain_openai.chat_models.base import BaseChatOpenAI
from openai import OpenAI
from pydantic.v1 import SecretStr

from agent.service import Agent
from dom.history_tree_processor.view import CoordinateSet, Coordinates
# from dom.service import DomService
from dom.views import DOMElementNode
# from agent.service import Agent
from mobile.context import MobileContext, MobileContextConfig
from mobile.mobile import MobileConfig, Mobile
import asyncio
from langchain_openai import ChatOpenAI

setup_logging()
dotenv.load_dotenv()

async def main(task):
    mobile = Mobile(config=MobileConfig())
    context = MobileContext(mobile=mobile, config=MobileContextConfig())

    base_url = 'https://api.deepseek.com'
    api_key = 'sk-3823750772cb4120ac390018fb0f6a1e'

    agent = Agent(
        task=task,
        llm=ChatOpenAI(
            model_name='deepseek-chat',
            openai_api_key=api_key,
            openai_api_base=base_url,
            # max_tokens=100000,
        ),
        # planner_llm=None,
        mobile=mobile,
        mobile_context=context
    )

    await agent.run()
    # await mobile.close()

    input("Press Enter to close...")


# async def main0():
# import cv2
# mobile = Mobile(config=MobileConfig())
# mobileContext = MobileContext(mobile, MobileContextConfig())
# driver = mobile.start()
# state = mobileContext._update_state()
# screenshot_b64 = driver.get_screenshot_as_base64().encode("utf-8")
# image_data = base64.b64decode(screenshot_b64)
# image = Image.open(BytesIO(image_data))
# image.show()


# pic_path = "Screenshot_2025-03-31-02-06-42-113_com.sankuai.me.png"
# image = cv2.imread(pic_path)
# image_b = base64.b64encode(cv2.imencode(".png", image)[1].tobytes())
# image_source = bytes.decode(image_b)
#
# ui_infer_url = f"http://127.0.0.1:9092/vision/ui-infer"
# params = {"type": "base64", "image": image_source}
# response = requests.post(url=ui_infer_url, json=params)
# result = response.json()
# print(result)
#
# selector_map = Dict[int, DOMElementNode]
# for item in result.get("data", []):
#     region = item.get("elem_det_region")
#     i = 0
#     x1, y1, x2, y2 = round(region[0]), round(region[1]), round(region[2]), round(region[3])
#     elementNode = DOMElementNode(
#         viewport_coordinates=CoordinateSet(
#             top_left=Coordinates(x=x1, y=y2),
#             top_right=Coordinates(x=x2, y=y2),
#             bottom_left=Coordinates(x=x1, y=y1),
#             bottom_right=Coordinates(x=x2, y=y1),
#             center=Coordinates(x=round((x1 + x2) / 2), y=round((y1 + y2) / 2)),
#             width=x2 - x1,
#             height=y2 - y1
#         )
#     )
#     selector_map[i] = elementNode
#     i = i + 1
#
#
# print(selector_map)
# print(driver.contexts)
# dom_service = DomService(driver)
# content 是返回的 DomState 内容
# vertical_offset, horizontal_offset = await mobileContext.get_scroll_info(driver)
# print(vertical_offset, horizontal_offset)
# content = await dom_service.get_clickable_elements(
#     focus_element=-1,
#     viewport_expansion=MobileContextConfig().viewport_expansion,
#     highlight_elements=MobileContextConfig().highlight_elements,
# )
# print(content)

if __name__ == "__main__":
    asyncio.run(main("进入到美食页面"))
    # from appium import webdriver
    # driver = None
    # desired_caps = {
    #     "platformName": "Android",
    #     "platformVersion": "11.0",
    #     "deviceName": "127.0.0.1:5555",
    #     "appPackage": "com.android.browser",
    #     "appActivity": "com.android.browser.BrowserActivity"
    # }
    # options = UiAutomator2Options().load_capabilities(desired_caps)
    # try:
    #     driver = webdriver.Remote("http://localhost:4723", options=options)
    # except Exception as e:
    #     logging.error(f"连接 Appium 服务器时出现错误: {e}", exc_info=True)
    # finally:
    #     if driver is not None:
    #         logging.info("关闭 Appium 会话。")
    #         driver.quit()
    # print("Appium is working!")
    # driver.quit()
    # asyncio.run(main(
    #     "In docs.google.com write my Papa a quick letter",
    #     "Android",
    #     "127.0.0.1:5555",
    #     "com.android.browser"
    # ))
