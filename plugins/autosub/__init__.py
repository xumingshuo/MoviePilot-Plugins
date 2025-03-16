import copy
import os
import re
import subprocess
import tempfile
import time
import traceback
from datetime import timedelta, datetime
from pathlib import Path
from typing import Tuple, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import iso639
import psutil
import pytz
import srt
from apscheduler.schedulers.background import BackgroundScheduler
from lxml import etree

from app.core.config import settings
from app.log import logger
from app.plugins import _PluginBase
from app.utils.system import SystemUtils
from plugins.autosub.ffmpeg import Ffmpeg
from plugins.autosub.translate.openai import OpenAi


class AutoSub(_PluginBase):
    # 插件名称
    plugin_name = "AI字幕自动生成"
    # 插件描述
    plugin_desc = "使用whisper自动生成视频文件字幕。"
    # 插件图标
    plugin_icon = "autosubtitles.jpeg"
    # 主题色
    plugin_color = "#2C4F7E"
    # 插件版本
    plugin_version = "0.6"
    # 插件作者
    plugin_author = "olly"
    # 作者主页
    author_url = "https://github.com/lightolly"
    # 插件配置项ID前缀
    plugin_config_prefix = "autosub"
    # 加载顺序
    plugin_order = 14
    # 可使用的用户级别
    auth_level = 2

    # 私有属性
    _running = False
    # 语句结束符
    _end_token = ['.', '!', '?', '。', '！', '？', '。"', '！"', '？"', '."', '!"', '?"']
    _noisy_token = [('(', ')'), ('[', ']'), ('{', '}'), ('【', '】'), ('♪', '♪'), ('♫', '♫'), ('♪♪', '♪♪')]

    def __init__(self):
        super().__init__()
        # ChatGPT
        self.openai = None
        self._chatgpt = None
        self._openai_key = None
        self._openai_url = None
        self._openai_proxy = None
        self._openai_model = None
        self._scheduler = None
        self.process_count = None
        self.fail_count = None
        self.success_count = None
        self.skip_count = None
        self.faster_whisper_model_path = None
        self.faster_whisper_model = None
        self.asr_engine = None
        self.send_notify = None
        self.additional_args = None
        self.translate_only = None
        self.translate_zh = None
        self.whisper_model = None
        self.whisper_main = None
        self.file_size = None

    def init_plugin(self, config=None):
        self.additional_args = '-t 4 -p 1'
        self.translate_zh = False
        self.translate_only = False
        self.whisper_model = None
        self.whisper_main = None
        self.file_size = None
        self.process_count = 0
        self.skip_count = 0
        self.fail_count = 0
        self.success_count = 0
        self.send_notify = False
        self.asr_engine = 'whisper.cpp'
        self.faster_whisper_model = 'base'
        self.faster_whisper_model_path = None

        # 如果没有配置信息， 则不处理
        if not config:
            return

        self.translate_zh = config.get('translate_zh', False)
        if self.translate_zh:
            chatgpt = self.get_config("ChatGPT")
            if not chatgpt:
                logger.error(f"翻译依赖于ChatGPT，请先维护ChatGPT插件")
                return
            self._chatgpt = chatgpt and chatgpt.get("enabled")
            self._openai_key = chatgpt and chatgpt.get("openai_key")
            self._openai_url = chatgpt and chatgpt.get("openai_url")
            self._openai_proxy = chatgpt and chatgpt.get("proxy")
            self._openai_model = chatgpt and chatgpt.get("model")
            if not self._openai_key:
                logger.error(f"翻译依赖于ChatGPT，请先维护openai_key")
                return
            self.openai = OpenAi(api_key=self._openai_key, api_url=self._openai_url,
                                 proxy=settings.PROXY if self._openai_proxy else None,
                                 model=self._openai_model)

        # config.get('path_list') 用 \n 分割为 list 并去除重复值和空值
        path_list = list(set(config.get('path_list').split('\n')))
        # file_size 转成数字
        self.file_size = config.get('file_size')
        self.whisper_main = config.get('whisper_main')
        self.whisper_model = config.get('whisper_model')
        self.translate_only = config.get('translate_only', False)
        self.additional_args = config.get('additional_args', '-t 4 -p 1')
        self.send_notify = config.get('send_notify', False)
        self.asr_engine = config.get('asr_engine', 'faster_whisper')
        self.faster_whisper_model = config.get('faster_whisper_model', 'base')
        self.faster_whisper_model_path = config.get('faster_whisper_model_path',
                                                    self.get_data_path() / "faster-whisper-models")

        run_now = config.get('run_now')
        if not run_now:
            return

        config['run_now'] = False
        self.update_config(config)

        # 如果没有配置信息， 则不处理
        if not path_list or not self.file_size:
            logger.warn(f"配置信息不完整，不进行处理")
            return

        # 校验文件大小是否为数字
        if not self.file_size.isdigit():
            logger.warn(f"文件大小不是数字，不进行处理")
            return

        # asr 配置检查
        if not self.translate_only and not self.__check_asr():
            return

        if self._running:
            logger.warn(f"上一次任务还未完成，不进行处理")
            return

        if run_now:
            self._scheduler = BackgroundScheduler(timezone=settings.TZ)
            logger.info("AI字幕自动生成任务，立即运行一次")
            self._scheduler.add_job(func=self._do_autosub, kwargs={'path_list': path_list}, trigger='date',
                                    run_date=datetime.now(tz=pytz.timezone(settings.TZ)) + timedelta(seconds=3),
                                    name="AI字幕自动生成")

            # 启动任务
            if self._scheduler.get_jobs():
                self._scheduler.print_jobs()
                self._scheduler.start()

    def _do_autosub(self, path_list: str):
        # 依次处理每个目录
        try:
            self._running = True
            self.success_count = self.skip_count = self.fail_count = self.process_count = 0
            for path in path_list:
                logger.info(f"开始处理目录：{path} ...")
                # 如果目录不存在， 则不处理
                if not os.path.exists(path):
                    logger.warn(f"目录不存在，不进行处理")
                    continue

                # 如果目录不是文件夹， 则不处理
                if not os.path.isdir(path):
                    logger.warn(f"目录不是文件夹，不进行处理")
                    continue

                # 如果目录不是绝对路径， 则不处理
                if not os.path.isabs(path):
                    logger.warn(f"目录不是绝对路径，不进行处理")
                    continue

                # 处理目录
                self.__process_folder_subtitle(path)
        except Exception as e:
            logger.error(f"处理异常: {e}")
        finally:
            logger.info(f"处理完成: "
                        f"成功{self.success_count} / 跳过{self.skip_count} / 失败{self.fail_count} / 共{self.process_count}")
            self._running = False

    def __process_folder_subtitle(self, path):
        """
        处理目录字幕
        :param path:
        :return:
        """
        # 获取目录媒体文件列表
        video_files = list(self.__get_library_files(path))
        if not video_files:
            return

        # 使用线程池处理视频文件
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self.__process_video_subtitle, video_file) for video_file in video_files]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"处理视频文件异常: {e}")

    def __process_video_subtitle(self, video_file):
        """
        处理单个视频文件的字幕
        :param video_file:
        :return:
        """
        if not video_file:
            return

        # 如果文件大小小于指定大小， 则不处理
        if os.path.getsize(video_file) < int(self.file_size):
            return

        self.process_count += 1
        start_time = time.time()
        file_path, file_ext = os.path.splitext(video_file)
        file_name = os.path.basename(video_file)

        try:
            logger.info(f"开始处理文件：{video_file} ...")
            # 判断目的字幕（和内嵌）是否已存在
            if self.__target_subtitle_exists(video_file):
                logger.warn(f"字幕文件已经存在，不进行处理")
                self.skip_count += 1
                return
            # 生成字幕
            if self.send_notify:
                self.post_message(title="自动字幕生成",
                                  text=f" 媒体: {file_name}\n 开始处理文件 ... ")
            ret, lang = self.__generate_subtitle(video_file, file_path, self.translate_only)
            if not ret:
                message = f" 媒体: {file_name}\n "
                if self.translate_only:
                    message += "内嵌&外挂字幕不存在，不进行翻译"
                    self.skip_count += 1
                else:
                    message += "生成字幕失败，跳过后续处理"
                    self.fail_count += 1

                if self.send_notify:
                    self.post_message(title="自动字幕生成", text=message)
                return

            if self.translate_zh:
                # 翻译字幕
                logger.info(f"开始翻译字幕为中文 ...")
                if self.send_notify:
                    self.post_message(title="自动字幕生成",
                                      text=f" 媒体: {file_name}\n 开始翻译字幕为中文 ... ")
                self.__translate_zh_subtitle(lang, f"{file_path}.{lang}.srt", f"{file_path}.zh.srt")
                logger.info(f"翻译字幕完成：{file_name}.zh.srt")

            end_time = time.time()
            message = f" 媒体: {file_name}\n 处理完成\n 字幕原始语言: {lang}\n "
            if self.translate_zh:
                message += f"字幕翻译语言: zh\n "
            message += f"耗时：{round(end_time - start_time, 2)}秒"
            logger.info(f"自动字幕生成 处理完成：{message}")
            if self.send_notify:
                self.post_message(title="自动字幕生成", text=message)
            self.success_count += 1
        except Exception as e:
            logger.error(f"自动字幕生成 处理异常：{e}")
            end_time = time.time()
            message = f" 媒体: {file_name}\n 处理失败\n 耗时：{round(end_time - start_time, 2)}秒"
            if self.send_notify:
                self.post_message(title="自动字幕生成", text=message)
            # 打印调用栈
            traceback.print_exc()
            self.fail_count += 1

    # 其他方法保持不变...
