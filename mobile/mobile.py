"""
Appium mobile driver manager.
"""
import asyncio
import gc
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from appium import webdriver
from appium.options.android import UiAutomator2Options
from appium.webdriver.appium_service import AppiumService
from dotenv import load_dotenv

from utils import time_execution_async

# from mobile.context import MobileContextConfig


# from utils import time_execution_async
load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class MobileConfig:
    # new_context_config: MobileContextConfig = field(default_factory=MobileContextConfig)

    platform_name: str = "Android"
    # platform_version: str = "11.0"
    device_name: str = "adb-330df616-gpA2Rj._adb-tls-connect._tcp"
    appPackage: str = 'com.sankuai.meituan'
    appActivity: str = 'com.meituan.android.pt.homepage.activity.MainActivity'
    automation_name: str = "UiAutomator2"
    new_command_timeout: int = 30000
    uiautomator2ServerInstallTimeout: int = 60000
    no_reset: bool = True
    remote_url: str = 'http://localhost:4723'


# @singleton: TODO - think about id singleton makes sense here
# @dev By default this is a singleton, but you can create multiple instances if you need to.
class Mobile:
    """
    Playwright browser on steroids.

    This is persistant browser factory that can spawn multiple browser contexts.
    It is recommended to use only one instance of Browser per your application (RAM usage will grow otherwise).
    """

    def __init__(
            self,
            config: MobileConfig = MobileConfig(),
    ):
        logger.debug('Initializing new mobile')
        self.config = config
        self.driver: Optional[webdriver.Remote] = None
        self.appium_service = AppiumService()

    # async def new_context(self, config: MobileContextConfig = MobileContextConfig()) -> MobileContext:
    #     """Create a mobile context"""
    #     return MobileContext(config=config, mobile=self)

    @time_execution_async('--init (mobile)')
    async def _init(self):
        """Initialize the mobile session"""
        self.driver = self.start()

    def start(self):
        desired_caps = {
            'platformName': self.config.platform_name,
            'deviceName': self.config.device_name,
            'appPackage': self.config.appPackage,
            "appActivity": self.config.appActivity,
            "automationName": self.config.automation_name,
            "noReset": self.config.no_reset,
            "newCommandTimeout": self.config.new_command_timeout,
            "uiautomator2ServerInstallTimeout": self.config.uiautomator2ServerInstallTimeout
        }
        options = UiAutomator2Options().load_capabilities(desired_caps)
        try:
            return webdriver.Remote(self.config.remote_url, options=options)
        except Exception as e:
            logging.error(f"连接 Appium 服务器时出现错误: {e}", exc_info=True)

    async def close(self):
        """Close the mobile instance"""
        try:
            if self.driver:
                await self.driver.quit()
        except Exception as e:
            logger.debug(f'Failed to close browser properly: {e}')
        finally:
            self.driver = None

# if __name__ == '__main__':
    # mobile = Mobile(config=MobileConfig())
    # mobileContext = MobileContext(mobile,MobileContextConfig())
    # driver = mobile.start()
    #
    # mobileState = mobileContext.update_state()
    #
    # print(mobileState)



    # time.sleep(1)
    # context = driver.current_context
    # print(driver.current_context)
    # print(context.title())
    # driver.find_element(AppiumBy.ID,"com.android.browser:id/btn_agree").click()
    # print(driver.current_context)
    #
    # time.sleep(1)
    # driver.find_element(AppiumBy.ID,"com.android.browser:id/action_person").click()
    # print(driver.page_source)
    # print(driver.current_context)
    #
    # driver.find_element(AppiumBy.ID, "com.android.browser:id/person_novel_des").click()
    # print(driver.current_context)
    #
    #
    # driver.switch_to.context(context)
