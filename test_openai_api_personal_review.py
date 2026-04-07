import base64
import difflib
import html
import json
import math
import mimetypes
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

APP_TITLE = "高级形象顾问 AI 专家审校台"
LOG_DIR = Path("logs_style_consultant_tester")
LOG_DIR.mkdir(parents=True, exist_ok=True)
PROMPT_VERSION_DIR = Path("prompt_versions")
INSTRUCTION_VERSION_DIR = PROMPT_VERSION_DIR / "instructions"
MASTER_PROMPT_VERSION_DIR = PROMPT_VERSION_DIR / "master_prompts"
INSTRUCTION_VERSION_DIR.mkdir(parents=True, exist_ok=True)
MASTER_PROMPT_VERSION_DIR.mkdir(parents=True, exist_ok=True)
REVIEW_DIR = Path("logs_style_consultant_reviews")
REVIEW_DIR.mkdir(parents=True, exist_ok=True)
REVIEW_CASES_JSONL = REVIEW_DIR / "review_cases.jsonl"
REVIEW_PATCHES_JSONL = REVIEW_DIR / "review_patches.jsonl"
TEXT_SELECTOR_COMPONENT_DIR = Path(__file__).parent / "streamlit_text_selector"

ERROR_TYPE_OPTIONS = [
    "",
    "风格判断错",
    "色彩判断错",
    "搭配建议错",
    "场合判断错",
    "身材修饰错",
    "措辞不准",
    "其他",
]

text_selector_component = components.declare_component(
    "text_selector_component",
    path=str(TEXT_SELECTOR_COMPONENT_DIR.resolve()),
)

PERSON_ORDERED_KEYS = [
    "person_id",
    "主风格",
    "次风格",
    "面部直曲",
    "面部量感",
    "面部动静",
    "冷暖",
    "明度",
    "彩度",
    "四季色型",
    "体型",
    "穿搭修正重点",
]

REFERENCE_FILE_CONFIGS = [
    {"label": "PCCS 色调图", "path": Path("pccs.png")},
    {"label": "服装风格象限图", "path": Path("Screenshot 2026-02-05 at 15.46.26.png")},
]

TASK_TYPES: Dict[str, Dict[str, Any]] = {
    "auto": {
        "label": "自动判断",
        "description": "根据用户当下的问题和图片，自动判断这次更接近哪类咨询任务。",
        "default_question": "例如：这件衣服适不适合我？ / 这几件哪个更适合我？ / 我今天去见客户怎么穿？",
        "image_hint": "上传与本轮问题相关的单品图、候选图或整身图，模型会自动判断问题类型。",
        "required_images": False,
        "extra_fields": [],
        "output_format": """请按最适合当前问题的方式输出，至少包含：
【任务识别】
- 你判断这次更接近哪类问题：
- 为什么这样判断：

【核心结论】
- 直接回答用户当下问题：

【分析】
- 只保留回答当前问题所必需的分析：

【最终在对话框里输出给用户的话】
- 2-3 句话，口语化，简洁，但要带一点理由和解释；只回答用户此刻最直接的问题，不要额外展开到用户没问的内容：""",
    },
    "single_item_fit_check": {
        "label": "这个衣服适不适合我",
        "description": "判断单品本身适不适合这个用户。",
        "default_question": "这件衣服适不适合我？",
        "image_hint": "上传或填写 1 张主单品图，允许补充细节图。",
        "required_images": False,
        "extra_fields": [],
        "output_format": """请按以下结构输出：
【任务结论】
- 结论：
- 一句话理由：

【单品分析】
- 量感：
- 直曲：
- 动静：
- 风格映射：
- 冷暖：
- 明度：
- 彩度：

【人物匹配】
- 风格匹配：
- 色彩匹配：
- 身材匹配：

【建议】
- 最适合怎么穿：
- 需要避开的点：""",
    },
    "multi_item_compare": {
        "label": "这几件衣服哪个适合我",
        "description": "对多个候选单品做排序和取舍。",
        "default_question": "这几件里哪个更适合我？",
        "image_hint": "上传或填写多张候选单品图，建议按 A/B/C 顺序提供。",
        "required_images": False,
        "extra_fields": [],
        "output_format": """请按以下结构输出：
【排序结论】
- 推荐顺序：
- 最推荐的是：
- 最不推荐的是：

【逐件判断】
- 单品 A：
- 单品 B：
- 单品 C：如无则省略

【为什么这样排】
- 风格原因：
- 色彩原因：
- 身材修饰原因：

【购买或穿着建议】
- 优先留：
- 可穿但需搭配调整：
- 不建议：""",
    },
    "single_item_styling": {
        "label": "这个衣服如何搭配",
        "description": "围绕一件主单品给出整套搭配方案。",
        "default_question": "这件衣服应该怎么搭？",
        "image_hint": "上传或填写 1 张主单品图，允许补充近景图。",
        "required_images": False,
        "extra_fields": ["occasion", "target_feeling"],
        "output_format": """请按以下结构输出：
【主判断】
- 这件单品更适合做主角还是配角：
- 最适合的场合：

【搭配方案 1】
- 上装/下装：
- 鞋：
- 包：
- 配饰：
- 外套：
- 为什么成立：

【搭配方案 2】
- 上装/下装：
- 鞋：
- 包：
- 配饰：
- 外套：
- 为什么成立：

【避雷提醒】
- 不建议怎么搭：
- 最容易翻车的点：""",
    },
    "item_match_check": {
        "label": "这个衣服和这个裤子能搭吗",
        "description": "检查多件单品放在一起是否成立，并给最小改动方案。",
        "default_question": "这几件衣服能不能搭在一起？",
        "image_hint": "上传或填写 2 张以上相关单品图。",
        "required_images": False,
        "extra_fields": ["occasion", "target_feeling"],
        "output_format": """请按以下结构输出：
【能不能搭】
- 结论：可以 / 部分可以 / 不可以

【问题拆解】
- 风格：
- 色彩：
- 比例：
- 量感：
- 场合：

【最小改动方案】
- 改哪一件：
- 怎么改最省：

【更优方案】
- 推荐替换：
- 搭配逻辑：""",
    },
    "outfit_review": {
        "label": "我这身衣服哪里不好看，我要怎么改进",
        "description": "点评整身穿搭并给低成本改法。",
        "default_question": "我这身衣服哪里不好看，应该怎么改？",
        "image_hint": "上传或填写整身照片，最好有正面全身图。",
        "required_images": False,
        "extra_fields": ["occasion", "target_feeling"],
        "output_format": """请按以下结构输出：
【整体结论】
- 适合度：
- 场合适配度：
- 最成功的地方：

【主要问题】
- 问题 1：
- 问题 2：
- 问题 3：如无可省略

【低成本改法】
- 先改什么：
- 为什么最值：

【理想改法】
- 更完整的优化方案：
- 优化后的整体感觉：""",
    },
    "occasion_styling": {
        "label": "我今天要去 xxxx，我应该怎么穿",
        "description": "根据场合和目标感受生成出门方案。",
        "default_question": "我今天去这个场合应该怎么穿？",
        "image_hint": "可选上传衣橱候选图；不上传时按公式输出。",
        "required_images": False,
        "extra_fields": ["time_info", "purpose", "place", "target_feeling"],
        "output_format": """请按以下结构输出：
【场合拆解】
- 时间：
- 目的：
- 地点：
- 想传达的感觉：

【方案 1：更保守】
- 上装/下装：
- 鞋包配饰：
- 适合原因：

【方案 2：更平衡】
- 上装/下装：
- 鞋包配饰：
- 适合原因：

【方案 3：更有风格感】
- 上装/下装：
- 鞋包配饰：
- 适合原因：

【关键提醒】
- 最重要的搭配原则：
- 最不建议犯的错：""",
    },
}

GENERIC_OUTPUT_FORMAT = """请按最适合当前问题的方式输出，至少包含：
【分析】
- 只保留回答当前问题所必需的分析：

【最终在对话框里输出给用户的话】
- 2-3 句话，口语化，简洁，但要带一点理由和解释；只回答用户此刻最直接的问题，不要额外展开到用户没问的内容："""

DEFAULT_MASTER_PROMPT = """请基于用户问题、客户参数和本轮图片进行分析，并给出适合当前问题的回答。"""
DEFAULT_INSTRUCTION_VERSION = "style_rules_v2026_03_29"
DEFAULT_MASTER_PROMPT_VERSION = "main_prompt_v2026_03_29"

DEFAULT_INSTRUCTIONS = """你是一名高级个人穿搭顾问与服装风格分析师。客户已经在线下完成了色彩测试和风格定位。你的任务是帮助用户完成单品分析、场合搭配、人衣匹配、整体优化等等。

你必须严格使用以下体系，不可擅自换标准：
1. 服装风格判断：只使用提供的“服装风格象限图”。
2. 色彩判断：只使用提供的 PCCS 色调图判断冷暖、明度、彩度。
3. 人物匹配：只依据提供的人物参数。
4. 搭配判断：优先看整体统一感、主次、呼应、比例、完成度，需要符合客户场景与想表达的感受。
5. 参考图本身会在每轮作为图片输入提供；你需要按这些参考图执行判断，但不要把参考图复述成用户问题。

【固定认知规则】
一、单品分析规则
对服饰单品必须先判断：
1. 判断量感
量感是视觉体量，不是实际重量。看廓形大小、面料厚薄挺软、图案大小、细节尺度、整体存在感。
输出：小 / 偏小 / 中 / 偏大 / 大

2. 判断直曲
直曲是线条语言。看外轮廓、装饰图案、领口、肩线、腰线、下摆、剪裁是利落规整还是圆润柔和。
输出：曲 / 偏曲 / 中 / 偏直 / 直

3. 判断动静
动静是视觉节奏与冲击感。看色彩饱和度和对比度、款式对称性和设计感和装饰性、面料的光泽感和垂感、图案秩序感。
输出：静 / 偏静 / 中 / 偏动 / 动

再映射风格，只能从以下九类选：
- 少女的
- 优雅的
- 自然的
- 俊秀的
- 前卫的
- 睿智的
- 戏剧的
- 古典的
- 浪漫的
如处于边界，输出“主风格 + 次偏向风格”。

判断优先依据：
- 廓形
- 结构
- 面料
- 线条
- 细节
- 花纹
- 色彩关系
不要把以下因素作为主判断依据：
- 模特长相
- 模特年龄感
- 妆容发型
- 姿势
- 摄影布景
- 品牌文案
- 图片背景色

二、色彩分析规则
用 PCCS 图判断：
- 冷暖：冷 / 暖 / 冷暖结合
- 明度：低 / 中 / 高
- 彩度：无彩色 / 低 / 中 / 高

三、人物适配规则
判断“单品是否适合此人穿”，必须按顺序：
1. 先看风格三维是否匹配：直曲、量感、动静
2. 再看色彩是否匹配：冷暖、明度、彩度
3. 最后看版型是否扬长避短、修正身材比例
4. 特殊场合可适度让步，例如：
   - 严肃职场的颜色需要是低彩度或者低明度的来彰显严肃稳重感
   - 冬季保暖需求可对露肤和轻盈感做让步
   - 功能性场合比如运动装和西服优先满足行动需求

四、搭配总原则
搭配时必须遵循以下逻辑：
1. 先定义场合与目标感受
   - 场合必须结合 TOP 原则：时间（季节）+ 目的 + 地点
   - 同时定义想传达的感觉，例如：成熟稳重、青春活力、松弛自然、精致优雅
2. 先搭上半身
   - 上半身优先匹配人物的风格和色彩
   - 上半身决定第一视觉感受
3. 再搭下半身
   - 下半身优先考虑身材修正、行动便利、与上半身是否协调
   - 下半身不必比上半身更“像本人”，但必须与上半身搭起来成立
4. 最后补鞋、包、饰品
   - 鞋、包、外套是最容易造成割裂、也最容易补救的单品
   - 饰品负责补层次、补亮点、补完成度，不是越多越好

五、搭配审美校验规则
每次给出搭配建议前，都要检查以下 8 点：
1. 感受是否统一：休闲、OL、复古、甜美、酷感等不能互相打架
2. 主次是否清晰：不能颜色太多、图案太多、装饰点太多，导致没有主角
3. 是否有呼应：色彩、材质、线条、量感至少有一到两个呼应点
4. 比例是否顺：优先 2 截，谨慎 3 截，避免把身材切短
5. 鞋子是否合理：鞋子必须匹配场合、风格、色彩、量感
6. 外套是否合理：大面积外套的冷暖、彩度、风格冲突会放大问题
7. 是否缺完成度：太平、太素时，应补层次或配饰，加入对比
8. 是否过度装饰：已有强图案、强颜色、强配件时，不要继续堆元素
搭配的和谐来源于呼应和共素，独特性和美感来源于适当的层次和对比。

六、任务执行要求
你需要先根据用户当下的问题、上传图片、图片中的文字信息和人物参数，自行判断这次更接近哪类任务，再选择最合适的回答方式。
你可在以下任务中自行判断：
- single_item_fit_check：判断单品是否适合此人。
- multi_item_compare：多件候选单品比较并排序。
- single_item_styling：围绕一件衣服给出搭配方案。
- item_match_check：判断多件单品能否搭在一起，并给最小改动方案。
- outfit_review：点评用户当前整身穿搭并给低成本改法。
- occasion_styling：按场合、目标感受给出穿搭方案。
- others:  饰品、化妆品判断选择等等。

七、人物匹配
会输入用户线下测试的结果和参数，判断单品是否合适，依赖以下几个维度：
【风格匹配】
- 直曲是否匹配：
- 量感是否匹配：
- 动静是否匹配：
- 结论：

【色彩匹配】
- 冷暖是否匹配：
- 明度是否匹配：
- 彩度是否匹配：
- 结论：

【身材匹配】
- 是否扬长：
- 是否避短：
- 是否优化比例：
- 结论：

【最终结论】
- 非常适合 / 适合 / 不适合 / 某些场合或者搭配下可以穿

八、输出与表达要求
- 不要泄露思维过程。
- 说人话，具体，明确，不能空泛。
- 如果信息不够，先基于已知信息给最稳妥的判断，再说明不确定点。
- 引用图片时使用“图1 / 图2”这种说法，避免混淆。
- 最后一行必须补上“最终在对话框里输出给用户的话”。
- 这句给用户的话要尽可能口语化、非常简洁，只回答用户当下的问题，不要额外展开。
- 如果用户只问适不适合，就直接回答适不适合和一句理由，不要主动讲避雷点或怎么搭。
- 用户如果没有问搭配问题，只是询问单品是否适合自己时，上半身按照风格色彩是否匹配回答，下半身考虑风格和身型修饰回答。"""

DEFAULT_PERSONS_JSONL = """{"person_id":"包媛媛","主风格":"自然","次风格":"古典","面部直曲":"曲","面部量感":"中","面部动静":"静","冷暖":"暖","明度":"中高","彩度":"低中","四季色型":"柔暖秋","体型":"梨形","穿搭修正重点":"弱化下半身量感，突出腰线，适合A字裙、高腰裤和简洁上衣，避免紧身裤、包臀裙和强调下半身宽度的设计"}
{"person_id":"何丹","主风格":"少女","次风格":"无明显次风格","面部直曲":"曲","面部量感":"小","面部动静":"静","冷暖":"暖","明度":"中高","彩度":"低","四季色型":"浅春","体型":"苹果型","穿搭修正重点":"突出锁骨、手腕、脚腕等纤细部位，强调高腰线和收腰，适合有曲线感且略硬挺的版型，修饰中腹部与臀腿"}
{"person_id":"赵爽","主风格":"俊秀","次风格":"前卫","面部直曲":"直","面部量感":"中","面部动静":"动","冷暖":"暖","明度":"中高","彩度":"低中","四季色型":"浅春","体型":"倒三角偏长方形","穿搭修正重点":"弱化上半身、强调下半身，平衡肩臀比例，上半身避免繁琐设计，适当露出锁骨、手臂、脚踝，增加柔和感"}
{"person_id":"宫静秋","主风格":"古典","次风格":"俊秀","面部直曲":"偏直","面部量感":"中偏大","面部动静":"动","冷暖":"冷","明度":"中高","彩度":"低","四季色型":"浅夏","体型":"倒三角","穿搭修正重点":"弱化上身、强调下半身并凸显腰身，适合V领、直筒裤或阔腿裤，避免高领、紧身裤、包臀裙和过于宽松的外套"}
{"person_id":"管冬梅","主风格":"自然","次风格":"古典","面部直曲":"直","面部量感":"大","面部动静":"动","冷暖":"暖","明度":"中高","彩度":"低中","四季色型":"浅春","体型":"倒三角偏苹果","穿搭修正重点":"适合简洁直线感与硬挺立体版型，弱化腰线并露出锁骨手腕脚腕，修饰中腹部和宽肩，避免过度强调腰腹"}
{"person_id":"曾秋艳","主风格":"优雅","次风格":"浪漫","面部直曲":"曲","面部量感":"中偏大","面部动静":"动","冷暖":"暖","明度":"中高","彩度":"低中","四季色型":"浅春","体型":"倒三角","穿搭修正重点":"弱化上身、强调下半身并凸显腰身，上半身避免繁琐，适合V领、直筒裤或阔腿裤，避免高领、紧身裤、包臀裙和过宽松外套"}
{"person_id":"陈嘉欣","主风格":"俊秀","次风格":"浪漫","面部直曲":"偏直","面部量感":"中","面部动静":"动","冷暖":"暖","明度":"中高","彩度":"中高","四季色型":"净春","体型":"长方形","穿搭修正重点":"适合H型、A版型与短而小的收腰款，露出锁骨手腕脚腕，避免紧身软塌、过紧裤装、上身过长和繁琐设计"}
{"person_id":"韩森森","主风格":"优雅","次风格":"自然","面部直曲":"曲","面部量感":"中","面部动静":"静","冷暖":"冷","明度":"中高","彩度":"低","四季色型":"浅夏","体型":"长方形","穿搭修正重点":"适合硬挺立体且带收腰的款式，塑造腰线并露出锁骨手腕脚腕，整体保持简洁直线感，修饰中腹部微胖和无明显腰身"}"""


def init_session_state() -> None:
    defaults = {
        "model": "gpt-5.4",
        "instruction_version": DEFAULT_INSTRUCTION_VERSION,
        "instructions": DEFAULT_INSTRUCTIONS,
        "master_prompt_version": DEFAULT_MASTER_PROMPT_VERSION,
        "master_prompt": DEFAULT_MASTER_PROMPT,
        "persons_jsonl": DEFAULT_PERSONS_JSONL,
        "reasoning_effort": "low",
        "verbosity": "low",
        "max_output_tokens": 1800,
        "timeout_seconds": 60,
        "temperature": 0.2,
        "top_p": 1.0,
        "user_question": "",
        "occasion": "",
        "time_info": "",
        "purpose": "",
        "place": "",
        "target_feeling": "",
        "additional_constraints": "",
        "closet_notes": "",
        "task_image_urls_text": "",
        "reference_image_urls_text": "",
        "include_local_reference_files": True,
        "run_history": [],
        "reviewer_id": "",
        "review_case_source": "运行记录",
        "review_selected_history_case": "",
        "review_selected_saved_case": "",
        "review_actions_by_case": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_data(show_spinner=False)
def parse_persons_jsonl(raw_text: str) -> List[Dict[str, Any]]:
    persons: List[Dict[str, Any]] = []
    for index, line in enumerate(raw_text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise ValueError(f"第 {index} 行不是 JSON 对象。")
        if not obj.get("person_id"):
            obj["person_id"] = f"person_{index:03d}"
        persons.append(obj)
    return persons


def get_client(timeout_seconds: float) -> OpenAI:
    return OpenAI(api_key=api_key, timeout=timeout_seconds)


def parse_labeled_url_lines(raw_text: str, default_prefix: str) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    for index, raw_line in enumerate(raw_text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        line = line.replace("｜", "|")
        if "|" in line:
            label, url = [part.strip() for part in line.split("|", 1)]
        else:
            label = f"{default_prefix}{index}"
            url = line
        if url:
            items.append({"label": label or f"{default_prefix}{index}", "url": url})
    return items


def guess_mime_type(filename: str, fallback: str = "image/png") -> str:
    guessed, _ = mimetypes.guess_type(filename)
    return guessed or fallback


@st.cache_data(show_spinner=False)
def file_to_data_url(path_str: str) -> str:
    path = Path(path_str)
    mime_type = guess_mime_type(path.name)
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def uploaded_file_to_data_url(uploaded_file) -> str:
    mime_type = uploaded_file.type or guess_mime_type(uploaded_file.name)
    encoded = base64.b64encode(uploaded_file.getvalue()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_reference_assets(
    include_local_reference_files: bool,
    reference_url_text: str,
    uploaded_reference_files: List[Any],
) -> List[Dict[str, Any]]:
    assets: List[Dict[str, Any]] = []

    if include_local_reference_files:
        for config in REFERENCE_FILE_CONFIGS:
            if config["path"].exists():
                assets.append(
                    {
                        "label": config["label"],
                        "image_url": file_to_data_url(str(config["path"].resolve())),
                        "preview_source": str(config["path"].resolve()),
                        "source_type": "local_reference",
                    }
                )

    for item in parse_labeled_url_lines(reference_url_text, "参考图"):
        assets.append(
            {
                "label": item["label"],
                "image_url": item["url"],
                "preview_source": item["url"],
                "source_type": "reference_url",
            }
        )

    for index, uploaded_file in enumerate(uploaded_reference_files, start=1):
        assets.append(
            {
                "label": f"参考上传{index} - {uploaded_file.name}",
                "image_url": uploaded_file_to_data_url(uploaded_file),
                "preview_source": uploaded_file.getvalue(),
                "source_type": "reference_upload",
            }
        )

    return dedupe_assets(assets)


def build_task_assets(task_image_url_text: str, uploaded_task_files: List[Any]) -> List[Dict[str, Any]]:
    assets: List[Dict[str, Any]] = []

    for item in parse_labeled_url_lines(task_image_url_text, "任务图"):
        assets.append(
            {
                "label": item["label"],
                "image_url": item["url"],
                "preview_source": item["url"],
                "source_type": "task_url",
            }
        )

    for index, uploaded_file in enumerate(uploaded_task_files, start=1):
        assets.append(
            {
                "label": f"上传图{index} - {uploaded_file.name}",
                "image_url": uploaded_file_to_data_url(uploaded_file),
                "preview_source": uploaded_file.getvalue(),
                "source_type": "task_upload",
            }
        )

    return dedupe_assets(assets)


def dedupe_assets(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for item in items:
        key = item["image_url"]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def serialize_images_for_log(images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    serialized: List[Dict[str, Any]] = []
    for item in images:
        serialized.append(
            {
                "label": item.get("label"),
                "source_type": item.get("source_type"),
                "image_url": item.get("image_url"),
            }
        )
    return serialized


def deserialize_images_from_log(images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    restored: List[Dict[str, Any]] = []
    for item in images or []:
        image_url = item.get("image_url")
        if not image_url:
            continue
        restored.append(
            {
                "label": item.get("label", "图片"),
                "source_type": item.get("source_type", "logged_image"),
                "image_url": image_url,
                "preview_source": image_url,
            }
        )
    return restored


def build_person_system_block(person: Dict[str, Any]) -> str:
    lines = ["【用户持久化信息】", "以下信息属于该用户的稳定画像，请在整轮判断中持续生效："]
    for key in PERSON_ORDERED_KEYS:
        if key in person:
            lines.append(f"- {key}: {person.get(key, '')}")
    extra_keys = [key for key in person.keys() if key not in PERSON_ORDERED_KEYS]
    for key in extra_keys:
        lines.append(f"- {key}: {person.get(key, '')}")
    return "\n".join(lines)


def build_request_instructions(
    instruction_version: str,
    base_instructions: str,
    person: Dict[str, Any],
) -> str:
    person_block = build_person_system_block(person)
    return f"""【Instructions Version】
- {instruction_version}

{base_instructions.strip()}

{person_block}
"""


def build_context_block(
    task_type: str,
    question: str,
    occasion: str,
    time_info: str,
    purpose: str,
    place: str,
    target_feeling: str,
    additional_constraints: str,
    closet_notes: str,
) -> str:
    _ = task_type
    resolved_question = question.strip() or TASK_TYPES["auto"]["default_question"]
    lines = [
        "【用户问题】",
        resolved_question,
    ]

    if any([occasion, time_info, purpose, place, target_feeling]):
        lines.extend(
            [
                "",
                "【场合与目标】",
                f"- 场合描述: {occasion or '未提供'}",
                f"- 时间/季节: {time_info or '未提供'}",
                f"- 目的: {purpose or '未提供'}",
                f"- 地点: {place or '未提供'}",
                f"- 想传达的感觉: {target_feeling or '未提供'}",
            ]
        )

    if additional_constraints or closet_notes:
        lines.extend(
            [
                "",
                "【补充信息】",
                f"- 限制条件: {additional_constraints or '无'}",
                f"- 衣橱/已有单品说明: {closet_notes or '无'}",
            ]
        )

    return "\n".join(lines)


def build_image_manifest(reference_assets: List[Dict[str, Any]], task_assets: List[Dict[str, Any]]) -> str:
    lines = ["【本轮图片说明】"]
    index = 1

    if reference_assets:
        for item in reference_assets:
            lines.append(f"- 图{index}: 参考资料 / {item['label']}")
            index += 1
    else:
        lines.append("- 本轮没有额外参考图。")

    if task_assets:
        for item in task_assets:
            lines.append(f"- 图{index}: 用户本轮任务图片 / {item['label']}")
            index += 1
    else:
        lines.append("- 本轮没有上传任务图片，请仅根据文字和人物参数回答。")

    return "\n".join(lines)


def build_final_prompt(
    master_prompt: str,
    person: Dict[str, Any],
    task_type: str,
    question: str,
    occasion: str,
    time_info: str,
    purpose: str,
    place: str,
    target_feeling: str,
    additional_constraints: str,
    closet_notes: str,
    reference_assets: List[Dict[str, Any]],
    task_assets: List[Dict[str, Any]],
) -> str:
    context_block = build_context_block(
        task_type=task_type,
        question=question,
        occasion=occasion,
        time_info=time_info,
        purpose=purpose,
        place=place,
        target_feeling=target_feeling,
        additional_constraints=additional_constraints,
        closet_notes=closet_notes,
    )
    image_manifest = build_image_manifest(reference_assets, task_assets)

    return f"""{master_prompt.strip()}

{context_block}

{image_manifest}
"""


def build_input_content(prompt: str, image_assets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "input_text", "text": prompt}]
    for item in image_assets:
        content.append({"type": "input_image", "image_url": item["image_url"]})
    return content


def call_openai(
    client: OpenAI,
    model: str,
    prompt: str,
    image_assets: List[Dict[str, Any]],
    reasoning_effort: str,
    max_output_tokens: int,
    temperature: Optional[float],
    top_p: Optional[float],
    verbosity: str,
    instructions: str,
):
    kwargs: Dict[str, Any] = {
        "model": model,
        "input": [{"role": "user", "content": build_input_content(prompt, image_assets)}],
        "max_output_tokens": max_output_tokens,
        "text": {"verbosity": verbosity},
    }

    if instructions.strip():
        kwargs["instructions"] = instructions.strip()

    if reasoning_effort != "none":
        kwargs["reasoning"] = {"effort": reasoning_effort}
    else:
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p

    return client.responses.create(**kwargs)


def to_plain_text(text: str) -> str:
    if not text:
        return ""
    cleaned = text.replace("**", "").replace("__", "").replace("`", "")
    cleaned = re.sub(r"(?m)^\s{0,3}#+\s*", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def serialize_response(response) -> Dict[str, Any]:
    try:
        raw_dict = response.model_dump()
    except Exception:
        raw_dict = {"raw": str(response)}

    request_id = getattr(response, "_request_id", None) or getattr(response, "request_id", None)
    output_text = getattr(response, "output_text", "")

    return {
        "id": getattr(response, "id", None),
        "model": getattr(response, "model", None),
        "output_text": output_text,
        "output_text_plain": to_plain_text(output_text),
        "usage": getattr(response, "usage", None),
        "request_id": request_id,
        "raw": raw_dict,
    }


def save_json_log(prefix: str, payload: Dict[str, Any]) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{prefix}_{timestamp}_{uuid.uuid4().hex[:8]}.json"
    path = LOG_DIR / filename
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return str(path.resolve())


def normalize_version_name(version_name: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"_", "-", "."} else "_" for char in version_name.strip())
    return cleaned.strip("_") or "untitled_version"


def build_version_file_path(version_name: str, version_type: str) -> Path:
    safe_name = normalize_version_name(version_name)
    base_dir = INSTRUCTION_VERSION_DIR if version_type == "instructions" else MASTER_PROMPT_VERSION_DIR
    return base_dir / f"{safe_name}.txt"


def save_version_file(version_name: str, content: str, version_type: str) -> str:
    path = build_version_file_path(version_name, version_type)
    path.write_text(content, encoding="utf-8")
    return str(path.resolve())


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def summarize_text(text: str, limit: int = 48) -> str:
    compact = " ".join(text.strip().split())
    if len(compact) <= limit:
        return compact or "<empty>"
    return compact[: limit - 1] + "…"


def list_saved_case_logs(limit: int = 80) -> List[Path]:
    patterns = ["style_consult_test_*.json", "style_consult_batch_test_*.json"]
    files = sorted(
        [path for pattern in patterns for path in LOG_DIR.glob(pattern)],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return files[:limit]


def build_case_id(base: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in base)
    return safe.strip("_") or uuid.uuid4().hex[:10]


def build_case_record_from_history(item: Dict[str, Any]) -> Dict[str, Any]:
    response = item.get("response", {})
    return {
        "case_id": item.get("case_id") or build_case_id(
            f"{item.get('person_id', 'case')}_{item.get('timestamp', '')}_{item.get('task_type', '')}"
        ),
        "timestamp": item.get("timestamp"),
        "task_type": item.get("task_type"),
        "person_id": item.get("person_id"),
        "instruction_version": item.get("instruction_version"),
        "master_prompt_version": item.get("master_prompt_version"),
        "question": item.get("question", ""),
        "person": item.get("person", {}),
        "final_prompt": item.get("final_prompt", ""),
        "raw_output": response.get("output_text_plain") or response.get("output_text", "") or "",
        "response": response,
        "log_path": item.get("log_path"),
        "all_images": item.get("all_images", []),
        "source_label": f"运行记录 / {item.get('person_id', '')} / {TASK_TYPES[item.get('task_type', 'single_item_fit_check')]['label']}",
        "source_kind": "run_history",
    }


def load_case_record_from_log(log_path_str: str) -> Dict[str, Any]:
    path = Path(log_path_str)
    payload = json.loads(path.read_text(encoding="utf-8"))
    person = payload.get("person", {}) or {}
    response = payload.get("response", {}) or {}
    task_type = payload.get("task_type", "single_item_fit_check")
    return {
        "case_id": payload.get("case_id") or payload.get("request_trace_id") or build_case_id(path.stem),
        "timestamp": payload.get("timestamp"),
        "task_type": task_type,
        "person_id": person.get("person_id") or payload.get("person_id") or "unknown_person",
        "instruction_version": payload.get("instruction_version"),
        "master_prompt_version": payload.get("master_prompt_version"),
        "question": payload.get("question", ""),
        "person": person,
        "final_prompt": payload.get("final_prompt", ""),
        "request_instructions": payload.get("request_instructions", ""),
        "raw_output": response.get("output_text_plain") or to_plain_text(response.get("output_text", "") or ""),
        "response": response,
        "log_path": str(path.resolve()),
        "all_images": deserialize_images_from_log(payload.get("all_images", [])),
        "source_label": f"日志文件 / {path.name}",
        "source_kind": "saved_log",
    }


def split_output_into_blocks(output_text: str) -> List[Dict[str, str]]:
    chunks = [chunk.strip() for chunk in output_text.strip().split("\n\n") if chunk.strip()]
    if not chunks and output_text.strip():
        chunks = [output_text.strip()]

    blocks: List[Dict[str, str]] = []
    for index, chunk in enumerate(chunks, start=1):
        first_line = chunk.splitlines()[0].strip()
        blocks.append(
            {
                "block_id": f"block_{index:02d}",
                "title": summarize_text(first_line, limit=36),
                "text": chunk,
            }
        )
    return blocks


def apply_patch_to_text(block_text: str, patch: Dict[str, Any]) -> str:
    op = patch["op"]
    old_text = patch.get("old_text", "")
    new_text = patch.get("new_text", "")

    if op == "replace_text":
        return block_text.replace(old_text, new_text, 1)
    if op == "delete_text":
        return block_text.replace(old_text, "", 1)
    if op == "insert_after_text":
        if old_text:
            return block_text.replace(old_text, old_text + new_text, 1)
        separator = "" if block_text.endswith("\n") or not block_text else "\n"
        return block_text + separator + new_text
    if op == "replace_block":
        return new_text
    return block_text


def apply_action_to_text(text: str, action: Dict[str, Any]) -> str:
    op = action["op"]
    start = int(action.get("start", 0))
    end = int(action.get("end", start))
    replacement_text = action.get("replacement_text", "")

    if op == "delete":
        return text[:start] + text[end:]
    if op == "edit":
        return text[:start] + replacement_text + text[end:]
    if op == "add":
        if not replacement_text.strip():
            return text
        insert_text = replacement_text.strip()
        separator = "\n" if end < len(text) and text[end:end + 1] != "\n" else ""
        return text[:end] + separator + insert_text + text[end:]
    return text


def build_review_view(case_record: Dict[str, Any], actions: List[Dict[str, Any]]) -> Dict[str, Any]:
    raw_output = case_record.get("raw_output", "") or ""
    current_text = raw_output
    applied_actions: List[Dict[str, Any]] = []

    for action in actions:
        current_text = apply_action_to_text(current_text, action)
        applied_actions.append(action)

    return {
        "raw_output": raw_output,
        "current_text": current_text,
        "actions": applied_actions,
    }


def upsert_case_actions(case_id: str, actions: List[Dict[str, Any]]) -> None:
    action_map = dict(st.session_state.review_actions_by_case)
    action_map[case_id] = actions
    st.session_state.review_actions_by_case = action_map


def get_case_actions(case_id: str) -> List[Dict[str, Any]]:
    action_map = st.session_state.review_actions_by_case or {}
    return list(action_map.get(case_id, []))


def render_text_selector(text: str, key: str, height: int = 420):
    return text_selector_component(
        text=text,
        height=height,
        key=key,
        default=None,
    )


def build_history_dataframe(history: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for item in history:
        response = item.get("response", {})
        output_text = response.get("output_text_plain") or response.get("output_text", "") or ""
        rows.append(
            {
                "timestamp": item["timestamp"],
                "task_type": item["task_type"],
                "task_label": TASK_TYPES[item["task_type"]]["label"],
                "person_id": item["person_id"],
                "instruction_version": item.get("instruction_version"),
                "master_prompt_version": item.get("master_prompt_version"),
                "images_count": len(item["all_images"]),
                "request_id": response.get("request_id"),
                "response_id": response.get("id"),
                "model": response.get("model"),
                "log_path": item["log_path"],
                "output_preview": output_text[:180],
            }
        )
    return pd.DataFrame(rows)


def render_image_grid(title: str, images: List[Dict[str, Any]]) -> None:
    st.markdown(f"**{title}**")
    if not images:
        st.info("暂无图片。")
        return

    cols = st.columns(2)
    for index, item in enumerate(images):
        with cols[index % 2]:
            st.image(item["preview_source"], caption=item["label"], use_container_width=True)


def render_version_badge(instruction_version: Optional[str], master_prompt_version: Optional[str]) -> None:
    st.info(
        f"版本标记：Instructions = `{instruction_version or '未设置'}` | "
        f"Master Prompt = `{master_prompt_version or '未设置'}`"
    )


def render_result_card(item: Dict[str, Any], index: int) -> None:
    meta = TASK_TYPES[item["task_type"]]
    with st.container(border=True):
        st.subheader(f"{index}. {meta['label']} / {item['person_id']}")
        st.write(
            {
                "timestamp": item["timestamp"],
                "task_type": item["task_type"],
                "instruction_version": item.get("instruction_version"),
                "master_prompt_version": item.get("master_prompt_version"),
                "request_id": item["response"].get("request_id"),
                "response_id": item["response"].get("id"),
                "model": item["response"].get("model"),
                "log_path": item["log_path"],
            }
        )
        render_image_grid("本轮图片", item["all_images"])
        st.markdown("**模型输出**")
        st.write(item["response"].get("output_text_plain") or item["response"].get("output_text") or "<empty>")

        with st.expander("查看本次 Prompt"):
            st.code(item["final_prompt"], language=None)

        if item.get("request_instructions"):
            with st.expander("查看本次 Instructions"):
                st.code(item["request_instructions"], language=None)

        with st.expander("查看原始响应"):
            st.json(item["response"].get("raw"))


def render_patch_summary(patch: Dict[str, Any], index: int) -> None:
    st.write(
        {
            "index": index,
            "op": patch["op"],
            "selected_text": patch.get("selected_text"),
            "replacement_text": patch.get("replacement_text"),
            "start": patch.get("start"),
            "end": patch.get("end"),
            "reviewer_id": patch.get("reviewer_id"),
            "created_at": patch.get("created_at"),
        }
    )


def inject_app_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --page-bg: #f5efe6;
            --card-bg: #fffdfa;
            --ink: #231d18;
            --muted: #70655a;
            --line: #e6d7c6;
            --accent: #b65d3f;
            --accent-soft: #f8e2d6;
            --good: #2c6b56;
            --good-soft: #e2f0ea;
            --warn: #8f5d1f;
            --warn-soft: #f4e8d3;
        }

        .stApp {
            background:
                radial-gradient(circle at top right, rgba(182, 93, 63, 0.09), transparent 28%),
                radial-gradient(circle at top left, rgba(44, 107, 86, 0.08), transparent 22%),
                linear-gradient(180deg, #f6f1ea 0%, #f3ece3 100%);
        }

        .gallery-hero {
            background: linear-gradient(135deg, rgba(255, 253, 250, 0.96), rgba(247, 239, 230, 0.96));
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 1.25rem 1.35rem;
            margin-bottom: 1rem;
            box-shadow: 0 12px 40px rgba(35, 29, 24, 0.06);
        }

        .gallery-kicker {
            color: var(--accent);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }

        .gallery-title {
            color: var(--ink);
            font-size: 1.55rem;
            font-weight: 700;
            line-height: 1.2;
            margin-bottom: 0.4rem;
        }

        .gallery-subtitle {
            color: var(--muted);
            font-size: 0.98rem;
            line-height: 1.55;
        }

        .person-card {
            background: var(--card-bg);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 1rem 1rem 0.9rem 1rem;
            margin-bottom: 0.8rem;
        }

        .person-title {
            color: var(--ink);
            font-size: 1.02rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }

        .person-summary {
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.55;
            margin-bottom: 0.7rem;
        }

        .tag-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.42rem;
        }

        .tag-chip {
            background: #f7efe5;
            border: 1px solid #ead9c5;
            border-radius: 999px;
            color: var(--ink);
            font-size: 0.76rem;
            padding: 0.24rem 0.58rem;
        }

        .status-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            border-radius: 999px;
            font-size: 0.76rem;
            font-weight: 700;
            padding: 0.28rem 0.62rem;
            margin-left: 0.35rem;
        }

        .status-edited {
            background: var(--accent-soft);
            color: var(--accent);
            border: 1px solid #edccb9;
        }

        .status-pass {
            background: var(--good-soft);
            color: var(--good);
            border: 1px solid #cae5d9;
        }

        .status-other {
            background: var(--warn-soft);
            color: var(--warn);
            border: 1px solid #ead9ba;
        }

        .compare-panel {
            background: var(--card-bg);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 1rem 1.08rem;
            min-height: 220px;
            margin-bottom: 0.9rem;
        }

        .compare-label {
            color: var(--ink);
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 0.55rem;
        }

        .compare-text {
            color: var(--ink);
            font-size: 0.96rem;
            line-height: 1.78;
            white-space: pre-wrap;
            word-break: break-word;
        }

        .diff-added {
            background: rgba(44, 107, 86, 0.18);
            border-radius: 0.22rem;
            padding: 0.02rem 0;
        }

        .diff-removed {
            background: rgba(182, 93, 63, 0.2);
            border-radius: 0.22rem;
            padding: 0.02rem 0;
        }

        .case-meta {
            color: var(--muted);
            font-size: 0.82rem;
            line-height: 1.6;
            margin: 0.25rem 0 0.8rem 0;
        }

        .action-pill {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.22rem 0.52rem;
            margin: 0 0.38rem 0.38rem 0;
            font-size: 0.74rem;
            font-weight: 600;
            background: #f7efe5;
            border: 1px solid #ead9c5;
            color: var(--ink);
        }

        .delta-board {
            background: linear-gradient(180deg, rgba(255, 253, 250, 0.98), rgba(248, 241, 232, 0.98));
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 0.95rem 1.05rem;
            margin: 0.2rem 0 0.95rem 0;
        }

        .delta-title {
            color: var(--ink);
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 0.65rem;
        }

        .delta-item {
            border-top: 1px dashed #e4d4c0;
            padding: 0.72rem 0 0.1rem 0;
        }

        .delta-item:first-child {
            border-top: none;
            padding-top: 0;
        }

        .delta-op {
            color: var(--accent);
            font-size: 0.78rem;
            font-weight: 700;
            margin-bottom: 0.42rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .delta-snippet {
            color: var(--ink);
            font-size: 0.9rem;
            line-height: 1.68;
            white-space: pre-wrap;
            word-break: break-word;
            margin-bottom: 0.35rem;
        }

        .delta-before {
            background: rgba(182, 93, 63, 0.08);
            border-left: 3px solid rgba(182, 93, 63, 0.55);
            padding: 0.45rem 0.55rem;
            border-radius: 0.45rem;
        }

        .delta-after {
            background: rgba(44, 107, 86, 0.08);
            border-left: 3px solid rgba(44, 107, 86, 0.55);
            padding: 0.45rem 0.55rem;
            border-radius: 0.45rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def parse_iso_datetime(value: Optional[str]) -> datetime:
    if not value:
        return datetime.min
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return datetime.min


def group_images_by_role(images: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped = {"task": [], "reference": [], "other": []}
    for item in images:
        source_type = str(item.get("source_type", ""))
        if source_type.startswith("task_"):
            grouped["task"].append(item)
        elif "reference" in source_type:
            grouped["reference"].append(item)
        else:
            grouped["other"].append(item)
    return grouped


def load_review_gallery_records() -> List[Dict[str, Any]]:
    if not REVIEW_CASES_JSONL.exists():
        return []

    records: List[Dict[str, Any]] = []
    with REVIEW_CASES_JSONL.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            review_record = json.loads(line)
            saved_at = review_record.get("saved_at")
            actions = review_record.get("actions", []) or []
            action_types = [str(action.get("op", "") or "") for action in actions]

            records.append(
                {
                    **review_record,
                    "saved_at_dt": parse_iso_datetime(saved_at),
                    "task_label": TASK_TYPES.get(review_record.get("task_type"), TASK_TYPES["auto"])["label"],
                    "question_preview": summarize_text(review_record.get("question", ""), limit=52),
                    "action_types": action_types,
                    "action_type_counts": {op: action_types.count(op) for op in sorted(set(action_types)) if op},
                    "source_label": review_record.get("source_label", ""),
                }
            )

    records.sort(key=lambda item: item["saved_at_dt"], reverse=True)
    return records


def hydrate_review_gallery_record(record: Dict[str, Any]) -> Dict[str, Any]:
    log_case: Dict[str, Any] = {}
    log_path = record.get("log_path")
    if log_path and Path(log_path).exists():
        log_case = load_case_record_from_log(log_path)

    all_images = log_case.get("all_images", [])
    image_groups = group_images_by_role(all_images)
    return {
        **record,
        "person": log_case.get("person") or {},
        "all_images": all_images,
        "task_images": image_groups["task"],
        "reference_images": image_groups["reference"],
        "other_images": image_groups["other"],
        "request_instructions": log_case.get("request_instructions", ""),
        "final_prompt": log_case.get("final_prompt", ""),
        "source_label": record.get("source_label") or log_case.get("source_label", ""),
    }


def render_status_chip(review_status: str) -> str:
    status = (review_status or "").strip().lower()
    label_map = {
        "edited": "已修改",
        "pass": "直接通过",
    }
    css_class = {
        "edited": "status-edited",
        "pass": "status-pass",
    }.get(status, "status-other")
    label = label_map.get(status, status or "未标记")
    return f'<span class="status-chip {css_class}">{html.escape(label)}</span>'


def render_person_card(person: Dict[str, Any]) -> None:
    if not person:
        st.info("没有补全到人物参数。")
        return

    summary_parts = [
        f"{person.get('主风格', '未知')} / {person.get('次风格', '未知')}",
        f"{person.get('四季色型', '未知')}",
        f"{person.get('体型', '未知')}",
    ]
    tags = [
        ("面部直曲", person.get("面部直曲")),
        ("面部量感", person.get("面部量感")),
        ("面部动静", person.get("面部动静")),
        ("冷暖", person.get("冷暖")),
        ("明度", person.get("明度")),
        ("彩度", person.get("彩度")),
    ]
    tag_html = "".join(
        f'<span class="tag-chip">{html.escape(label)}: {html.escape(str(value))}</span>'
        for label, value in tags
        if value
    )
    repair_focus = str(person.get("穿搭修正重点", "未提供"))
    markup = f"""
    <div class="person-card">
      <div class="person-title">人物卡 / {html.escape(str(person.get('person_id', '未命名')))}</div>
      <div class="person-summary">{html.escape(" · ".join(summary_parts))}</div>
      <div class="tag-row">{tag_html}</div>
      <div class="person-summary" style="margin-top:0.78rem;"><strong>修正重点：</strong>{html.escape(repair_focus)}</div>
    </div>
    """
    st.markdown(markup, unsafe_allow_html=True)


def build_highlighted_diff_html(source_text: str, target_text: str, mode: str) -> str:
    source_tokens = re.findall(r"\S+|\s+", source_text)
    target_tokens = re.findall(r"\S+|\s+", target_text)
    matcher = difflib.SequenceMatcher(a=source_tokens, b=target_tokens)
    parts: List[str] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if mode == "source":
            snippet = "".join(source_tokens[i1:i2])
            highlight = tag in {"replace", "delete"}
            css_class = "diff-removed"
        else:
            snippet = "".join(target_tokens[j1:j2])
            highlight = tag in {"replace", "insert"}
            css_class = "diff-added"

        if not snippet:
            continue
        escaped = html.escape(snippet)
        parts.append(f'<span class="{css_class}">{escaped}</span>' if highlight else escaped)

    return "".join(parts) or html.escape(source_text if mode == "source" else target_text)


def render_action_pills(action_type_counts: Dict[str, int]) -> None:
    if not action_type_counts:
        return
    pill_html = "".join(
        f'<span class="action-pill">{html.escape(op)} x {count}</span>'
        for op, count in action_type_counts.items()
    )
    st.markdown(pill_html, unsafe_allow_html=True)


def render_delta_summary(actions: List[Dict[str, Any]]) -> None:
    if not actions:
        st.markdown(
            """
            <div class="delta-board">
              <div class="delta-title">变更点</div>
              <div class="delta-snippet">这个案例没有专家修改，AI 输出与最终版本一致。</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    label_map = {
        "edit": "替换",
        "delete": "删除",
        "add": "新增",
        "star": "重点标注",
    }
    item_html: List[str] = []
    for index, action in enumerate(actions, start=1):
        op = str(action.get("op", "") or "")
        before_text = str(action.get("selected_text", "") or "")
        after_text = str(action.get("replacement_text", "") or "")
        blocks = [
            f'<div class="delta-op">{index}. {html.escape(label_map.get(op, op or "变更"))}</div>'
        ]
        if before_text:
            blocks.append(
                f'<div class="delta-snippet delta-before"><strong>原文：</strong>{html.escape(before_text)}</div>'
            )
        if op != "delete" and after_text:
            blocks.append(
                f'<div class="delta-snippet delta-after"><strong>修改后：</strong>{html.escape(after_text)}</div>'
            )
        if op == "delete":
            blocks.append(
                '<div class="delta-snippet delta-after"><strong>修改后：</strong>这段内容被移除。</div>'
            )
        if op == "star":
            blocks.append(
                '<div class="delta-snippet delta-after"><strong>说明：</strong>这段被专家标为重点，正文未改写。</div>'
            )
        item_html.append(f'<div class="delta-item">{"".join(blocks)}</div>')

    st.markdown(
        f"""
        <div class="delta-board">
          <div class="delta-title">变更点</div>
          {''.join(item_html)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_gallery_case(record: Dict[str, Any], index: int) -> None:
    saved_label = record["saved_at_dt"].strftime("%Y-%m-%d %H:%M") if record["saved_at_dt"] != datetime.min else "未知"
    header_html = f"""
    <div class="gallery-title">{index}. {html.escape(str(record.get('person_id', '未命名人物')))}</div>
    <div class="gallery-subtitle">{html.escape(str(record.get('question', '') or '未填写问题'))}</div>
    <div class="case-meta">
      {render_status_chip(str(record.get("review_status", "")))}
      <span class="status-chip status-other">{html.escape(str(record.get('task_label', '未分类')))}</span>
      <span class="status-chip status-other">保存于 {html.escape(saved_label)}</span>
    </div>
    <div class="case-meta">
      case_id = <code>{html.escape(str(record.get("case_id", "")))}</code> |
      版本 = <code>{html.escape(str(record.get("instruction_version") or "未设置"))}</code> /
      <code>{html.escape(str(record.get("master_prompt_version") or "未设置"))}</code>
    </div>
    """

    with st.container(border=True):
        st.markdown(header_html, unsafe_allow_html=True)
        render_action_pills(record.get("action_type_counts", {}))

        top_left, top_right = st.columns([0.82, 1.18], gap="large")
        with top_left:
            render_person_card(record.get("person", {}))
            if record.get("reference_images"):
                with st.expander(f"查看参考图 ({len(record['reference_images'])})", expanded=False):
                    render_image_grid("参考图", record["reference_images"])

        with top_right:
            if record.get("task_images"):
                render_image_grid("任务图", record["task_images"])
            else:
                st.info("这个案例没有可回放的任务图。")

        render_delta_summary(record.get("actions", []))

        raw_html = build_highlighted_diff_html(record.get("raw_output", ""), record.get("final_output", ""), "source")
        st.markdown(
            f"""
            <div class="compare-panel">
              <div class="compare-label">AI 原输出</div>
              <div class="compare-text">{raw_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        final_html = build_highlighted_diff_html(record.get("raw_output", ""), record.get("final_output", ""), "target")
        st.markdown(
            f"""
            <div class="compare-panel">
              <div class="compare-label">专家修改后</div>
              <div class="compare-text">{final_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if record.get("actions"):
            with st.expander(f"查看修改动作明细 ({len(record['actions'])})", expanded=False):
                for patch_index, patch in enumerate(record["actions"], start=1):
                    render_patch_summary(patch, patch_index)

        extra_col1, extra_col2 = st.columns(2)
        with extra_col1:
            with st.expander("查看本次 Prompt", expanded=False):
                st.code(record.get("final_prompt", "") or "无", language=None)
        with extra_col2:
            with st.expander("查看本次 Instructions", expanded=False):
                st.code(record.get("request_instructions", "") or "无", language=None)


def render_review_gallery_tab() -> None:
    records = load_review_gallery_records()

    st.markdown(
        """
        <div class="gallery-hero">
          <div class="gallery-kicker">Review Showcase</div>
          <div class="gallery-title">已审校案例对比展厅</div>
          <div class="gallery-subtitle">
            把所有 review 过的案例集中展示，左侧看人物卡与任务图，右侧直接并排比较 AI 原输出和专家修改后的输出。
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not records:
        st.info("还没有已保存的 review 结果。先在“专家审校”里保存一些案例。")
        return

    total_count = len(records)
    edited_count = sum(1 for item in records if item.get("review_status") == "edited")
    pass_count = sum(1 for item in records if item.get("review_status") == "pass")
    people_count = len({item.get("person_id") for item in records if item.get("person_id")})

    metric_cols = st.columns(4)
    metric_cols[0].metric("已审校案例", total_count)
    metric_cols[1].metric("有专家改动", edited_count)
    metric_cols[2].metric("直接通过", pass_count)
    metric_cols[3].metric("涉及人物", people_count)

    all_people = sorted({str(item.get("person_id", "")) for item in records if item.get("person_id")})
    all_statuses = sorted({str(item.get("review_status", "")) for item in records if item.get("review_status")})
    task_type_options = ["全部"] + sorted(
        {str(item.get("task_type", "")) for item in records if item.get("task_type")}
    )

    filter_col1, filter_col2, filter_col3 = st.columns([1.1, 0.9, 1.0])
    with filter_col1:
        keyword = st.text_input(
            "搜索人物 / 问题 / 输出",
            placeholder="例如：韩森森、裤子、适合",
            key="gallery_keyword",
        ).strip()
    with filter_col2:
        selected_people = st.multiselect(
            "筛选人物",
            all_people,
            default=[],
            key="gallery_people_filter",
        )
    with filter_col3:
        selected_statuses = st.multiselect(
            "筛选状态",
            all_statuses,
            default=all_statuses,
            key="gallery_status_filter",
        )

    option_col1, option_col2, option_col3 = st.columns([0.9, 0.8, 0.8])
    with option_col1:
        selected_task_type = st.selectbox(
            "任务类型",
            task_type_options,
            index=0,
            key="gallery_task_type_filter",
        )
    with option_col2:
        page_size = st.selectbox(
            "每页展示",
            [4, 6, 8, 10],
            index=1,
            key="gallery_page_size",
        )
    with option_col3:
        latest_only = st.checkbox("每个 case 只看最新一次", value=True, key="gallery_latest_only")

    filtered_records = records
    if latest_only:
        latest_map: Dict[str, Dict[str, Any]] = {}
        for item in filtered_records:
            case_id = str(item.get("case_id", ""))
            if case_id not in latest_map:
                latest_map[case_id] = item
        filtered_records = list(latest_map.values())

    if selected_people:
        filtered_records = [item for item in filtered_records if item.get("person_id") in selected_people]

    if selected_statuses:
        filtered_records = [item for item in filtered_records if item.get("review_status") in selected_statuses]

    if selected_task_type != "全部":
        filtered_records = [item for item in filtered_records if item.get("task_type") == selected_task_type]

    if keyword:
        lowered = keyword.lower()
        filtered_records = [
            item
            for item in filtered_records
            if lowered in str(item.get("person_id", "")).lower()
            or lowered in str(item.get("question", "")).lower()
            or lowered in str(item.get("raw_output", "")).lower()
            or lowered in str(item.get("final_output", "")).lower()
        ]

    filtered_records.sort(key=lambda item: item["saved_at_dt"], reverse=True)

    if not filtered_records:
        st.warning("当前筛选条件下没有案例。")
        return

    page_count = max(1, math.ceil(len(filtered_records) / page_size))
    page = st.number_input(
        "页码",
        min_value=1,
        max_value=page_count,
        value=1,
        step=1,
        key="gallery_page",
    )

    start_index = (page - 1) * page_size
    end_index = start_index + page_size
    visible_records = filtered_records[start_index:end_index]

    st.caption(f"当前展示 {len(visible_records)} / {len(filtered_records)} 个案例，第 {page} / {page_count} 页。")

    for index, record in enumerate(visible_records, start=start_index + 1):
        render_gallery_case(hydrate_review_gallery_record(record), index)


def render_review_workbench_tab() -> None:
    st.subheader("专家审校")
    st.caption("从已保存日志中选择一个 case，然后直接在模型输出里选中错的片段做 Delete / Edit / Star。")

    saved_logs = list_saved_case_logs()
    selected_case: Optional[Dict[str, Any]] = None

    if saved_logs:
        log_options = {path.name: str(path.resolve()) for path in saved_logs}
        log_labels = list(log_options.keys())
        if (
            not st.session_state.review_selected_saved_case
            or st.session_state.review_selected_saved_case not in log_options
        ):
            st.session_state.review_selected_saved_case = log_labels[0]

        st.selectbox(
            "选择已保存日志",
            log_labels,
            index=log_labels.index(st.session_state.review_selected_saved_case),
            key="review_selected_saved_case",
        )
        selected_case = load_case_record_from_log(log_options[st.session_state.review_selected_saved_case])
    else:
        st.info("还没有找到已保存的测试日志。请先在 `test_openai_api_personal.py` 里跑测试。")

    if not selected_case:
        return

    render_version_badge(
        selected_case.get("instruction_version"),
        selected_case.get("master_prompt_version"),
    )
    st.write(
        {
            "case_id": selected_case["case_id"],
            "source": selected_case["source_label"],
            "timestamp": selected_case["timestamp"],
            "task_type": selected_case["task_type"],
            "instruction_version": selected_case.get("instruction_version"),
            "master_prompt_version": selected_case.get("master_prompt_version"),
            "person_id": selected_case["person_id"],
            "log_path": selected_case["log_path"],
        }
    )

    if selected_case.get("person"):
        with st.expander("查看人物参数", expanded=False):
            st.json(selected_case["person"])

    if selected_case.get("all_images"):
        render_image_grid("本轮图片", selected_case["all_images"])
    else:
        st.info("这个 case 没有可回放的图片数据。通常是因为它来自较早的日志，当时还没有把图片一起写入日志。请用最新版本重新跑一次该 case。")

    actions = get_case_actions(selected_case["case_id"])
    review_view = build_review_view(selected_case, actions)

    st.info("直接用鼠标选中说错的话。选中后会浮出操作条：`Delete`、`Edit`、`Add`、`Star`。")

    selector_event = render_text_selector(
        review_view["current_text"],
        key=f"text_selector_{selected_case['case_id']}_{len(actions)}",
        height=460,
    )

    if selector_event and selector_event.get("event_id"):
        action_type = selector_event.get("action")
        selected_text_value = selector_event.get("selected_text", "")
        replacement_text = selector_event.get("replacement_text", "")
        start = selector_event.get("start", 0)
        end = selector_event.get("end", 0)

        if action_type in {"edit", "add"} and not replacement_text:
            st.warning(f"{action_type.title()} 操作需要填写内容。")
        elif action_type in {"delete", "edit", "add", "star"} and selected_text_value:
            action_record = {
                "action_id": uuid.uuid4().hex[:10],
                "case_id": selected_case["case_id"],
                "op": action_type,
                "selected_text": selected_text_value,
                "replacement_text": replacement_text,
                "start": start,
                "end": end,
                "context_before": selector_event.get("context_before", ""),
                "context_after": selector_event.get("context_after", ""),
                "reviewer_id": st.session_state.reviewer_id,
                "created_at": datetime.now().isoformat(),
            }
            upsert_case_actions(selected_case["case_id"], actions + [action_record])
            st.rerun()

    st.markdown("**已记录操作**")
    if actions:
        for index, action in enumerate(actions, start=1):
            with st.container(border=True):
                render_patch_summary(action, index)
        action_col1, action_col2 = st.columns(2)
        with action_col1:
            if st.button("撤销最后一步", use_container_width=True):
                upsert_case_actions(selected_case["case_id"], actions[:-1])
                st.rerun()
        with action_col2:
            if st.button("清空本 case 操作", use_container_width=True):
                upsert_case_actions(selected_case["case_id"], [])
                st.rerun()
    else:
        st.info("当前还没有任何操作。完全正确的话可以直接保存。")

    save_col1, save_col2 = st.columns(2)
    with save_col1:
        if st.button("保存审校结果", type="primary", use_container_width=True):
            review_id = uuid.uuid4().hex[:10]
            final_output = review_view["current_text"]
            review_status = "pass" if not actions else "edited"
            review_record = {
                "review_id": review_id,
                "case_id": selected_case["case_id"],
                "review_status": review_status,
                "reviewer_id": st.session_state.reviewer_id,
                "saved_at": datetime.now().isoformat(),
                "source_kind": selected_case["source_kind"],
                "source_label": selected_case["source_label"],
                "task_type": selected_case["task_type"],
                "instruction_version": selected_case.get("instruction_version"),
                "master_prompt_version": selected_case.get("master_prompt_version"),
                "person_id": selected_case["person_id"],
                "question": selected_case["question"],
                "log_path": selected_case["log_path"],
                "raw_output": selected_case["raw_output"],
                "final_output": final_output,
                "action_count": len(actions),
                "actions": actions,
            }
            bundle_path = REVIEW_DIR / (
                f"review_{selected_case['case_id']}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
            )
            bundle_path.write_text(
                json.dumps(review_record, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            append_jsonl(REVIEW_CASES_JSONL, review_record)
            for patch in actions:
                append_jsonl(
                    REVIEW_PATCHES_JSONL,
                    {
                        "review_id": review_id,
                        "case_id": selected_case["case_id"],
                        "task_type": selected_case["task_type"],
                        "instruction_version": selected_case.get("instruction_version"),
                        "master_prompt_version": selected_case.get("master_prompt_version"),
                        "person_id": selected_case["person_id"],
                        **patch,
                    },
                )
            st.success(f"审校结果已保存：{bundle_path.resolve()}")
    with save_col2:
        st.download_button(
            "下载修订后文本",
            data=review_view["current_text"].encode("utf-8"),
            file_name=f"{selected_case['case_id']}_final_output.txt",
            mime="text/plain",
            use_container_width=True,
        )


init_session_state()

st.set_page_config(page_title=APP_TITLE, layout="wide")
inject_app_styles()
st.title(APP_TITLE)
st.caption("现在包含两个入口：一个继续做专家审校，一个把已 review 案例集中做人物卡 + 图片 + 输出对比展示。")

with st.sidebar:
    st.markdown("### 审校设置")
    st.text_input(
        "reviewer_id",
        placeholder="例如：expert_a",
        key="reviewer_id",
    )

    st.divider()
    st.markdown("### 环境状态")
    st.write(
        {
            "logs_dir": str(LOG_DIR.resolve()),
            "review_logs_dir": str(REVIEW_DIR.resolve()),
            "saved_case_count": len(list_saved_case_logs(limit=1000)),
            "review_case_count": len(load_review_gallery_records()),
        }
    )

tab_gallery, tab_review = st.tabs(["案例对比展厅", "专家审校"])

with tab_gallery:
    render_review_gallery_tab()

with tab_review:
    render_review_workbench_tab()

st.divider()

with st.expander("运行方式"):
    st.markdown(
        """
建议环境：
- Python 3.10+
- 已安装 `streamlit openai python-dotenv pandas`
- 本页面不发 API 请求，主要依赖本地日志文件
        """
    )
    st.code(
        "\n".join(
            [
                "python -m venv .venv",
                "source .venv/bin/activate",
                "pip install -U pip",
                "pip install streamlit openai python-dotenv pandas",
                "streamlit run test_openai_api_personal_review.py",
            ]
        ),
        language="bash",
    )

with st.expander("依赖说明"):
    st.code(
        "\n".join(
            [
                "Python 3.10+",
                "streamlit",
                "openai",
                "python-dotenv",
                "pandas",
            ]
        ),
        language=None,
    )

with st.expander("环境变量示例"):
    st.code('OPENAI_API_KEY="sk-..."', language="bash")

with st.expander("使用说明"):
    st.markdown(
        """
1. 先在 `test_openai_api_personal.py` 里跑测试，生成日志。
2. 回到这个页面，在“选择已保存日志”里选一个 case。
3. 如果日志里带了 `all_images`，这里会直接显示本轮图片。
4. 直接选中错误片段，做 Delete / Edit / Star。
5. 保存后会写入 `logs_style_consultant_reviews`。
        """
    )
