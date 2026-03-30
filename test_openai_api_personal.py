import base64
import hashlib
import json
import mimetypes
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

APP_TITLE = "高级形象顾问 AI 测试台"
LOG_DIR = Path("logs_style_consultant_tester")
LOG_DIR.mkdir(parents=True, exist_ok=True)
PROMPT_VERSION_DIR = Path("prompt_versions")
INSTRUCTION_VERSION_DIR = PROMPT_VERSION_DIR / "instructions"
MASTER_PROMPT_VERSION_DIR = PROMPT_VERSION_DIR / "master_prompts"
INSTRUCTION_VERSION_DIR.mkdir(parents=True, exist_ok=True)
MASTER_PROMPT_VERSION_DIR.mkdir(parents=True, exist_ok=True)

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
        "batch_test_rows": [
            {"person_id": "", "images": "", "question": ""},
            {"person_id": "", "images": "", "question": ""},
            {"person_id": "", "images": "", "question": ""},
        ],
        "batch_editor_initialized": False,
        "batch_library_alias_map": {},
        "batch_library_next_index": 1,
        "prompt_load_message": "",
        "prompt_load_error": "",
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


def looks_like_remote_image_ref(value: str) -> bool:
    lowered = value.lower()
    return lowered.startswith(("http://", "https://", "data:"))


def build_assets_from_text(
    raw_text: str,
    default_prefix: str,
    source_prefix: str,
    alias_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    assets: List[Dict[str, Any]] = []

    for item in parse_labeled_url_lines(raw_text, default_prefix):
        raw_ref = item["url"].strip()
        if not raw_ref:
            continue

        alias_asset = (alias_lookup or {}).get(raw_ref)
        if alias_asset:
            resolved_asset = dict(alias_asset)
            if "|" in raw_ref:
                resolved_asset["label"] = item["label"]
            elif item["label"] and item["label"] != raw_ref:
                resolved_asset["label"] = item["label"]
            assets.append(resolved_asset)
            continue

        if looks_like_remote_image_ref(raw_ref):
            assets.append(
                {
                    "label": item["label"],
                    "image_url": raw_ref,
                    "preview_source": raw_ref,
                    "source_type": f"{source_prefix}_url",
                }
            )
            continue

        candidate_path = Path(raw_ref).expanduser()
        if not candidate_path.is_absolute():
            candidate_path = (Path.cwd() / candidate_path).resolve()

        if candidate_path.exists() and candidate_path.is_file():
            assets.append(
                {
                    "label": item["label"],
                    "image_url": file_to_data_url(str(candidate_path)),
                    "preview_source": str(candidate_path),
                    "source_type": f"{source_prefix}_path",
                }
            )
            continue

        assets.append(
            {
                "label": item["label"],
                "image_url": raw_ref,
                "preview_source": raw_ref,
                "source_type": f"{source_prefix}_url",
            }
        )

    return dedupe_assets(assets)


def get_uploaded_file_signature(uploaded_file) -> str:
    content = uploaded_file.getvalue()
    digest = hashlib.md5(content).hexdigest()
    return f"{uploaded_file.name}:{len(content)}:{digest}"


def build_uploaded_library_assets(uploaded_files: List[Any], source_prefix: str) -> List[Dict[str, Any]]:
    assets: List[Dict[str, Any]] = []
    alias_map: Dict[str, str] = dict(st.session_state.batch_library_alias_map)
    next_index = int(st.session_state.batch_library_next_index)

    for index, uploaded_file in enumerate(uploaded_files, start=1):
        signature = get_uploaded_file_signature(uploaded_file)
        alias = alias_map.get(signature)
        if not alias:
            alias = f"IMG{next_index:03d}"
            alias_map[signature] = alias
            next_index += 1
        file_bytes = uploaded_file.getvalue()
        assets.append(
            {
                "alias": alias,
                "file_name": uploaded_file.name,
                "label": f"{alias} - {uploaded_file.name}",
                "image_url": uploaded_file_to_data_url(uploaded_file),
                "preview_source": file_bytes,
                "source_type": f"{source_prefix}_upload_library",
            }
        )

    st.session_state.batch_library_alias_map = alias_map
    st.session_state.batch_library_next_index = next_index
    return assets


def build_uploaded_library_lookup(assets: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for asset in assets:
        asset_copy = dict(asset)
        alias = asset_copy.get("alias")
        file_name = asset_copy.get("file_name")
        if alias:
            lookup[str(alias)] = asset_copy
        if file_name:
            lookup[str(file_name)] = asset_copy
    return lookup


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

    assets.extend(build_assets_from_text(reference_url_text, "参考图", "reference"))

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
    assets = build_assets_from_text(task_image_url_text, "任务图", "task")

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


def load_version_file(version_name: str, version_type: str) -> str:
    path = build_version_file_path(version_name, version_type)
    if not path.exists():
        raise FileNotFoundError(f"未找到版本文件：{path.resolve()}")
    return path.read_text(encoding="utf-8")


def load_instructions_into_editor() -> None:
    try:
        st.session_state["instructions"] = load_version_file(
            version_name=st.session_state.instruction_version,
            version_type="instructions",
        )
        st.session_state["prompt_load_message"] = "Instructions 已从版本文件载入。"
        st.session_state["prompt_load_error"] = ""
    except Exception as exc:
        st.session_state["prompt_load_error"] = str(exc)
        st.session_state["prompt_load_message"] = ""


def load_master_prompt_into_editor() -> None:
    try:
        st.session_state["master_prompt"] = load_version_file(
            version_name=st.session_state.master_prompt_version,
            version_type="master_prompt",
        )
        st.session_state["prompt_load_message"] = "Master Prompt 已从版本文件载入。"
        st.session_state["prompt_load_error"] = ""
    except Exception as exc:
        st.session_state["prompt_load_error"] = str(exc)
        st.session_state["prompt_load_message"] = ""


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


def execute_style_request(
    selected_person: Dict[str, Any],
    question: str,
    reference_assets: List[Dict[str, Any]],
    task_assets: List[Dict[str, Any]],
    log_prefix: str,
    run_mode: str,
    batch_row_index: Optional[int] = None,
) -> Dict[str, Any]:
    all_images = dedupe_assets(reference_assets + task_assets)
    request_instructions = build_request_instructions(
        instruction_version=st.session_state.instruction_version,
        base_instructions=st.session_state.instructions,
        person=selected_person,
    )
    final_prompt = build_final_prompt(
        master_prompt=st.session_state.master_prompt,
        person=selected_person,
        task_type="auto",
        question=question,
        occasion="",
        time_info="",
        purpose="",
        place="",
        target_feeling="",
        additional_constraints="",
        closet_notes="",
        reference_assets=reference_assets,
        task_assets=task_assets,
    )

    client = get_client(st.session_state.timeout_seconds)
    run_timestamp = datetime.now().isoformat()
    request_trace_id = str(uuid.uuid4())

    try:
        response = call_openai(
            client=client,
            model=st.session_state.model,
            prompt=final_prompt,
            image_assets=all_images,
            reasoning_effort=st.session_state.reasoning_effort,
            max_output_tokens=st.session_state.max_output_tokens,
            temperature=st.session_state.temperature if st.session_state.reasoning_effort == "none" else None,
            top_p=st.session_state.top_p if st.session_state.reasoning_effort == "none" else None,
            verbosity=st.session_state.verbosity,
            instructions=request_instructions,
        )
        serialized = serialize_response(response)

        log_payload = {
            "timestamp": run_timestamp,
            "request_trace_id": request_trace_id,
            "run_mode": run_mode,
            "batch_row_index": batch_row_index,
            "instruction_version": st.session_state.instruction_version,
            "master_prompt_version": st.session_state.master_prompt_version,
            "task_type": "auto",
            "task_label": TASK_TYPES["auto"]["label"],
            "person": selected_person,
            "question": question,
            "instructions": st.session_state.instructions,
            "request_instructions": request_instructions,
            "master_prompt": st.session_state.master_prompt,
            "final_prompt": final_prompt,
            "all_images": serialize_images_for_log(all_images),
            "reference_assets": [{"label": x["label"], "source_type": x["source_type"]} for x in reference_assets],
            "task_assets": [{"label": x["label"], "source_type": x["source_type"]} for x in task_assets],
            "image_count": len(all_images),
            "params": {
                "model": st.session_state.model,
                "reasoning_effort": st.session_state.reasoning_effort,
                "verbosity": st.session_state.verbosity,
                "temperature": st.session_state.temperature if st.session_state.reasoning_effort == "none" else None,
                "top_p": st.session_state.top_p if st.session_state.reasoning_effort == "none" else None,
                "max_output_tokens": st.session_state.max_output_tokens,
                "timeout_seconds": st.session_state.timeout_seconds,
            },
            "response": serialized,
        }
        log_path = save_json_log(log_prefix, log_payload)

        history_item = {
            "timestamp": run_timestamp,
            "task_type": "auto",
            "run_mode": run_mode,
            "batch_row_index": batch_row_index,
            "person_id": selected_person["person_id"],
            "instruction_version": st.session_state.instruction_version,
            "master_prompt_version": st.session_state.master_prompt_version,
            "question": question,
            "person": selected_person,
            "all_images": all_images,
            "request_instructions": request_instructions,
            "final_prompt": final_prompt,
            "response": serialized,
            "log_path": log_path,
        }
        return {"ok": True, "history_item": history_item, "log_path": log_path}
    except Exception as exc:
        error_payload = {
            "timestamp": run_timestamp,
            "request_trace_id": request_trace_id,
            "run_mode": run_mode,
            "batch_row_index": batch_row_index,
            "instruction_version": st.session_state.instruction_version,
            "master_prompt_version": st.session_state.master_prompt_version,
            "task_type": "auto",
            "task_label": TASK_TYPES["auto"]["label"],
            "person": selected_person,
            "question": question,
            "request_instructions": request_instructions,
            "final_prompt": final_prompt,
            "error": repr(exc),
        }
        error_log_path = save_json_log("style_consult_error", error_payload)
        return {
            "ok": False,
            "error": str(exc),
            "error_log_path": error_log_path,
            "final_prompt": final_prompt,
            "request_instructions": request_instructions,
        }


def ensure_batch_row_widget_state(person_options: List[str]) -> None:
    rows = st.session_state.batch_test_rows
    default_person = person_options[0] if person_options else ""

    if not st.session_state.batch_editor_initialized:
        for index, row in enumerate(rows):
            st.session_state[f"batch_person_{index}"] = row.get("person_id") or default_person
            st.session_state[f"batch_images_{index}"] = row.get("images", "")
            st.session_state[f"batch_question_{index}"] = row.get("question", "")
        st.session_state.batch_editor_initialized = True
        return

    for index, row in enumerate(rows):
        person_key = f"batch_person_{index}"
        images_key = f"batch_images_{index}"
        question_key = f"batch_question_{index}"
        if person_key not in st.session_state:
            st.session_state[person_key] = row.get("person_id") or default_person
        if images_key not in st.session_state:
            st.session_state[images_key] = row.get("images", "")
        if question_key not in st.session_state:
            st.session_state[question_key] = row.get("question", "")


def sync_batch_rows_from_widgets(person_options: List[str]) -> None:
    rows: List[Dict[str, str]] = []
    default_person = person_options[0] if person_options else ""
    row_count = len(st.session_state.batch_test_rows)

    for index in range(row_count):
        rows.append(
            {
                "person_id": str(st.session_state.get(f"batch_person_{index}", default_person) or default_person),
                "images": str(st.session_state.get(f"batch_images_{index}", "") or ""),
                "question": str(st.session_state.get(f"batch_question_{index}", "") or ""),
            }
        )

    st.session_state.batch_test_rows = rows


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
                "run_mode": item.get("run_mode", "single"),
                "batch_row_index": item.get("batch_row_index"),
                "person_id": item["person_id"],
                "instruction_version": item.get("instruction_version"),
                "master_prompt_version": item.get("master_prompt_version"),
                "images_count": len(item["all_images"]),
                "request_id": response.get("request_id"),
                "response_id": response.get("id"),
                "model": response.get("model"),
                "log_path": item["log_path"],
                "question_preview": (item.get("question") or "")[:80],
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


def render_batch_library_panel(images: List[Dict[str, Any]]) -> None:
    st.markdown("**批量图片库**")
    if not images:
        st.info("先上传批量任务图片库，右侧会显示可滚动的图片 ID、文件名和缩略图。")
        return

    st.dataframe(
        pd.DataFrame([{"id": item["alias"], "file_name": item["file_name"]} for item in images]),
        use_container_width=True,
        hide_index=True,
        height=220,
    )

    with st.container(border=True, height=780):
        for item in images:
            with st.container(border=True):
                st.markdown(f"**`{item['alias']}`**")
                st.caption(item["file_name"])
                st.image(item["preview_source"], use_container_width=True)


def render_version_badge(instruction_version: Optional[str], master_prompt_version: Optional[str]) -> None:
    st.info(
        f"版本标记：Instructions = `{instruction_version or '未设置'}` | "
        f"Master Prompt = `{master_prompt_version or '未设置'}`"
    )


def render_result_card(item: Dict[str, Any], index: int) -> None:
    meta = TASK_TYPES[item["task_type"]]
    with st.container(border=True):
        st.subheader(f"{index}. {meta['label']} / {item['person_id']}")
        render_version_badge(item.get("instruction_version"), item.get("master_prompt_version"))
        st.write(
            {
                "timestamp": item["timestamp"],
                "task_type": item["task_type"],
                "run_mode": item.get("run_mode", "single"),
                "batch_row_index": item.get("batch_row_index"),
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


init_session_state()

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("把线下已测客户参数、风格规则、场合需求和服饰图片组合起来，直接测试不同咨询问题。")

with st.sidebar:
    st.header("模型配置")
    if st.session_state.prompt_load_message:
        st.success(st.session_state.prompt_load_message)
        st.session_state.prompt_load_message = ""
    if st.session_state.prompt_load_error:
        st.error(st.session_state.prompt_load_error)
        st.session_state.prompt_load_error = ""
    st.selectbox(
        "模型",
        options=["gpt-5.4", "gpt-5.4-pro"],
        index=["gpt-5.4", "gpt-5.4-pro"].index(st.session_state.model),
        key="model",
    )
    st.text_area(
        "Instructions",
        height=360,
        key="instructions",
    )
    st.text_input(
        "instruction_version",
        key="instruction_version",
    )
    st.text_input(
        "master_prompt_version",
        key="master_prompt_version",
    )
    instruction_action_col1, instruction_action_col2 = st.columns(2)
    with instruction_action_col1:
        st.button(
            "加载 Instructions 版本",
            use_container_width=True,
            on_click=load_instructions_into_editor,
        )
    with instruction_action_col2:
        if st.button("保存 Instructions 版本", use_container_width=True):
            saved_path = save_version_file(
                version_name=st.session_state.instruction_version,
                content=st.session_state.instructions,
                version_type="instructions",
            )
            st.success(f"已保存到 {saved_path}")
    prompt_action_col1, prompt_action_col2 = st.columns(2)
    with prompt_action_col1:
        st.button(
            "加载 Master Prompt 版本",
            use_container_width=True,
            on_click=load_master_prompt_into_editor,
        )
    with prompt_action_col2:
        if st.button("保存 Master Prompt 版本", use_container_width=True):
            saved_path = save_version_file(
                version_name=st.session_state.master_prompt_version,
                content=st.session_state.master_prompt,
                version_type="master_prompt",
            )
            st.success(f"已保存到 {saved_path}")
    st.selectbox(
        "Reasoning effort",
        options=["none", "low", "medium", "high", "xhigh"],
        index=["none", "low", "medium", "high", "xhigh"].index(st.session_state.reasoning_effort),
        key="reasoning_effort",
    )
    st.selectbox(
        "Text verbosity",
        options=["low", "medium", "high"],
        index=["low", "medium", "high"].index(st.session_state.verbosity),
        key="verbosity",
    )
    st.slider("max_output_tokens", 300, 4000, st.session_state.max_output_tokens, 100, key="max_output_tokens")
    st.slider("timeout_seconds", 10, 300, st.session_state.timeout_seconds, 10, key="timeout_seconds")

    st.markdown("### 采样参数")
    if st.session_state.reasoning_effort == "none":
        st.slider("temperature", 0.0, 2.0, float(st.session_state.temperature), 0.1, key="temperature")
        st.slider("top_p", 0.0, 1.0, float(st.session_state.top_p), 0.05, key="top_p")
    else:
        st.caption("当前使用 reasoning，temperature / top_p 不生效。")

    if st.button("清空运行记录", use_container_width=True):
        st.session_state.run_history = []
        st.rerun()

    st.divider()
    st.markdown("### 环境状态")
    st.write(
        {
            "api_key_loaded": bool(api_key),
            "instruction_version": st.session_state.instruction_version,
            "master_prompt_version": st.session_state.master_prompt_version,
            "instruction_version_path": str(
                build_version_file_path(st.session_state.instruction_version, "instructions").resolve()
            ),
            "master_prompt_version_path": str(
                build_version_file_path(st.session_state.master_prompt_version, "master_prompt").resolve()
            ),
            "local_reference_files": [
                str(config["path"].resolve())
                for config in REFERENCE_FILE_CONFIGS
                if config["path"].exists()
            ],
            "logs_dir": str(LOG_DIR.resolve()),
        }
    )

tab_task, tab_batch, tab_persons, tab_prompt, tab_references, tab_history = st.tabs(
    ["任务测试", "批量测试", "人物库", "Prompt", "参考图", "运行记录"]
)

selected_person: Optional[Dict[str, Any]] = None
reference_assets: List[Dict[str, Any]] = []
task_assets: List[Dict[str, Any]] = []
batch_library_assets: List[Dict[str, Any]] = []
batch_library_lookup: Dict[str, Dict[str, Any]] = {}
final_prompt_preview = ""
run_button = False
batch_run_button = False
batch_add_row_button = False
batch_remove_last_row_button = False

with tab_persons:
    st.subheader("人物参数库")
    st.text_area(
        "人物参数 JSONL",
        height=300,
        help="每行一个 JSON 对象，表示一个线下已经完成测试的客户参数包。",
        key="persons_jsonl",
    )
    try:
        persons = parse_persons_jsonl(st.session_state.persons_jsonl)
        st.success(f"已解析 {len(persons)} 个客户参数")
        st.dataframe(pd.DataFrame(persons), use_container_width=True)
    except Exception as exc:
        persons = []
        st.error(str(exc))

with tab_references:
    st.subheader("参考图设置")
    st.checkbox(
        "自动附带本地参考图（PCCS + 风格象限图）",
        key="include_local_reference_files",
    )
    st.text_area(
        "额外参考图 URL",
        height=140,
        help="每行一张图。可写成“标签 | URL”，例如：色票参考 | https://example.com/a.png",
        key="reference_image_urls_text",
    )
    uploaded_reference_files = st.file_uploader(
        "上传额外参考图",
        accept_multiple_files=True,
        type=["png", "jpg", "jpeg", "webp"],
    )
    reference_assets = build_reference_assets(
        include_local_reference_files=st.session_state.include_local_reference_files,
        reference_url_text=st.session_state.reference_image_urls_text,
        uploaded_reference_files=uploaded_reference_files or [],
    )
    render_image_grid("当前参考图预览", reference_assets)

with tab_task:
    st.subheader("咨询任务测试")

    try:
        persons = parse_persons_jsonl(st.session_state.persons_jsonl)
    except Exception:
        persons = []

    person_options = [person["person_id"] for person in persons]
    if not person_options:
        st.warning("请先在“人物库”中录入至少一个客户参数。")
    else:
        if "selected_person_id" not in st.session_state or st.session_state.selected_person_id not in person_options:
            st.session_state.selected_person_id = person_options[0]
        selected_person_id = st.selectbox("选择客户", person_options, key="selected_person_id")
        selected_person = next(
            (person for person in persons if person["person_id"] == selected_person_id),
            persons[0],
        )
        st.json(selected_person)

    task_meta = TASK_TYPES["auto"]
    st.caption(task_meta["description"])
    st.info(task_meta["image_hint"])

    st.text_area(
        "用户问题",
        height=90,
        help=f"示例：{task_meta['default_question']}",
        key="user_question",
    )
    st.caption("测试入口只保留用户原话和图片。场合、地点、限制条件等信息请直接写在用户问题里。")

    st.session_state.occasion = ""
    st.session_state.time_info = ""
    st.session_state.purpose = ""
    st.session_state.place = ""
    st.session_state.target_feeling = ""
    st.session_state.additional_constraints = ""
    st.session_state.closet_notes = ""

    st.text_area(
        "任务图片 URL",
        height=160,
        help="每行一张图。可写成“标签 | URL”。例如：候选A | https://...png",
        key="task_image_urls_text",
    )
    uploaded_task_files = st.file_uploader(
        "上传任务图片",
        accept_multiple_files=True,
        type=["png", "jpg", "jpeg", "webp"],
    )
    task_assets = build_task_assets(
        task_image_url_text=st.session_state.task_image_urls_text,
        uploaded_task_files=uploaded_task_files or [],
    )
    render_image_grid("任务图片预览", task_assets)

    if selected_person:
        final_prompt_preview = build_final_prompt(
            master_prompt=st.session_state.master_prompt,
            person=selected_person,
            task_type="auto",
            question=st.session_state.user_question,
            occasion=st.session_state.occasion,
            time_info=st.session_state.time_info,
            purpose=st.session_state.purpose,
            place=st.session_state.place,
            target_feeling=st.session_state.target_feeling,
            additional_constraints=st.session_state.additional_constraints,
            closet_notes=st.session_state.closet_notes,
            reference_assets=reference_assets,
            task_assets=task_assets,
        )
        with st.expander("查看即将发送的 Prompt", expanded=False):
            st.code(final_prompt_preview, language=None)
        render_version_badge(st.session_state.instruction_version, st.session_state.master_prompt_version)

    run_button = st.button("发送本轮测试", type="primary", use_container_width=True)

with tab_batch:
    st.subheader("批量测试")
    st.caption("按行配置请求。每一行都会单独发给模型，并各自保存一份 JSON 日志，后面可直接做审校。")
    batch_left_col, batch_right_col = st.columns([1.75, 1.0], gap="large")

    with batch_left_col:
        uploaded_batch_files = st.file_uploader(
            "上传批量任务图片库",
            accept_multiple_files=True,
            type=["png", "jpg", "jpeg", "webp"],
            key="batch_image_library_uploader",
            help="先把这批任务里会反复用到的图片上传到图库，再在下面的“图”列里填 IMG001 / IMG002 这类短 ID。",
        )
        batch_library_assets = build_uploaded_library_assets(uploaded_batch_files or [], "task")
        batch_library_lookup = build_uploaded_library_lookup(batch_library_assets)

        if batch_library_assets:
            st.caption("右侧会固定显示图片库；下方“图”列里可直接写 IMG001、IMG002，也可以混用 URL、本地路径。")

        try:
            persons = parse_persons_jsonl(st.session_state.persons_jsonl)
        except Exception:
            persons = []

        person_options = [person["person_id"] for person in persons]
        if not person_options:
            st.warning("请先在“人物库”中录入至少一个客户参数。")
        else:
            ensure_batch_row_widget_state(person_options)
            sync_batch_rows_from_widgets(person_options)

            st.markdown("**批量请求列表**")
            for index, _row in enumerate(st.session_state.batch_test_rows):
                with st.container(border=True):
                    st.markdown(f"`第 {index + 1} 行`")
                    col_person, col_images, col_question = st.columns([1.05, 1.45, 1.7])
                    with col_person:
                        st.selectbox(
                            "人",
                            person_options,
                            key=f"batch_person_{index}",
                            label_visibility="collapsed",
                        )
                    with col_images:
                        st.text_area(
                            "图",
                            key=f"batch_images_{index}",
                            height=110,
                            help="支持一轮多图。每行一张，可写 IMG001 这类图库 ID，也可写 标签 | URL/本地路径。",
                            label_visibility="collapsed",
                            placeholder="IMG001\nIMG002",
                        )
                    with col_question:
                        st.text_area(
                            "问题",
                            key=f"batch_question_{index}",
                            height=110,
                            label_visibility="collapsed",
                            placeholder="例如：这套衣服适合我吗？",
                        )

            sync_batch_rows_from_widgets(person_options)
            st.caption(
                "图列示例：IMG001，或：候选A | https://example.com/a.jpg，或：裙子正面 | /Users/xxx/Desktop/look1.jpg。多图请换行。"
            )
            render_version_badge(st.session_state.instruction_version, st.session_state.master_prompt_version)
            row_action_col1, row_action_col2 = st.columns(2)
            with row_action_col1:
                batch_add_row_button = st.button("新增一行", use_container_width=True)
            with row_action_col2:
                batch_remove_last_row_button = st.button("删除最后一行", use_container_width=True)
            batch_run_button = st.button("逐行发送批量测试", type="primary", use_container_width=True)

    with batch_right_col:
        render_batch_library_panel(batch_library_assets)

with tab_prompt:
    st.subheader("主 Prompt")
    st.text_area(
        "Master Prompt",
        height=540,
        key="master_prompt",
    )

with tab_history:
    st.subheader("运行记录")
    history = st.session_state.run_history
    if history:
        st.dataframe(build_history_dataframe(history), use_container_width=True)
        for index, item in enumerate(history, start=1):
            render_result_card(item, index)
    else:
        st.info("还没有运行记录。")

if batch_add_row_button:
    current_rows = list(st.session_state.batch_test_rows)
    new_index = len(current_rows)
    default_person = ""
    try:
        parsed_persons = parse_persons_jsonl(st.session_state.persons_jsonl)
        if parsed_persons:
            default_person = parsed_persons[0]["person_id"]
    except Exception:
        default_person = ""

    current_rows.append({"person_id": default_person, "images": "", "question": ""})
    st.session_state.batch_test_rows = current_rows
    st.session_state[f"batch_person_{new_index}"] = default_person
    st.session_state[f"batch_images_{new_index}"] = ""
    st.session_state[f"batch_question_{new_index}"] = ""
    st.rerun()

if batch_remove_last_row_button:
    current_rows = list(st.session_state.batch_test_rows)
    if len(current_rows) > 1:
        remove_index = len(current_rows) - 1
        current_rows.pop()
        st.session_state.batch_test_rows = current_rows
        for key in [
            f"batch_person_{remove_index}",
            f"batch_images_{remove_index}",
            f"batch_question_{remove_index}",
        ]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

if run_button:
    if not api_key:
        st.error("没有读取到 OPENAI_API_KEY，请先检查 .env。")
        st.stop()

    if not selected_person:
        st.error("请先选择客户。")
        st.stop()

    if not st.session_state.master_prompt.strip():
        st.error("Master Prompt 不能为空。")
        st.stop()

    with st.spinner("正在向模型发送请求..."):
        result = execute_style_request(
            selected_person=selected_person,
            question=st.session_state.user_question,
            reference_assets=reference_assets,
            task_assets=task_assets,
            log_prefix="style_consult_test",
            run_mode="single",
        )
        if result["ok"]:
            history_item = result["history_item"]
            st.session_state.run_history = [history_item] + st.session_state.run_history
            st.success("请求完成，结果已写入运行记录。")
            render_result_card(history_item, 1)
        else:
            st.error(f"请求失败：{result['error']}")
            st.info(f"错误日志已保存：{result['error_log_path']}")

if batch_run_button:
    if not api_key:
        st.error("没有读取到 OPENAI_API_KEY，请先检查 .env。")
        st.stop()

    if not st.session_state.master_prompt.strip():
        st.error("Master Prompt 不能为空。")
        st.stop()

    try:
        persons = parse_persons_jsonl(st.session_state.persons_jsonl)
    except Exception as exc:
        st.error(f"人物库解析失败：{exc}")
        st.stop()

    person_map = {person["person_id"]: person for person in persons}
    raw_rows = st.session_state.batch_test_rows
    batch_rows: List[Dict[str, Any]] = []
    validation_errors: List[str] = []

    for row_index, row in enumerate(raw_rows, start=1):
        person_id = str(row.get("person_id", "") or "").strip()
        question = str(row.get("question", "") or "").strip()
        images_text = str(row.get("images", "") or "").strip()
        if not any([person_id, question, images_text]):
            continue
        if not person_id:
            validation_errors.append(f"第 {row_index} 行缺少人物。")
            continue
        if person_id not in person_map:
            validation_errors.append(f"第 {row_index} 行的人物 `{person_id}` 不在人物库中。")
            continue
        if not question:
            validation_errors.append(f"第 {row_index} 行缺少问题。")
            continue
        batch_rows.append(
            {
                "row_index": row_index,
                "person": person_map[person_id],
                "question": question,
                "images_text": images_text,
            }
        )

    if validation_errors:
        for message in validation_errors:
            st.error(message)
        st.stop()

    if not batch_rows:
        st.error("没有可发送的批量行。请至少填写一行人物和问题。")
        st.stop()

    progress = st.progress(0.0)
    status = st.empty()
    batch_success_items: List[Dict[str, Any]] = []
    batch_failures: List[str] = []

    for index, row in enumerate(batch_rows, start=1):
        status.write(f"正在发送第 {row['row_index']} 行（{index}/{len(batch_rows)}）...")
        batch_task_assets = build_assets_from_text(
            row["images_text"],
            "任务图",
            "task",
            alias_lookup=batch_library_lookup,
        )
        result = execute_style_request(
            selected_person=row["person"],
            question=row["question"],
            reference_assets=reference_assets,
            task_assets=batch_task_assets,
            log_prefix="style_consult_batch_test",
            run_mode="batch",
            batch_row_index=row["row_index"],
        )
        if result["ok"]:
            batch_success_items.append(result["history_item"])
        else:
            batch_failures.append(
                f"第 {row['row_index']} 行失败：{result['error']}（错误日志：{result['error_log_path']}）"
            )
        progress.progress(index / len(batch_rows))

    if batch_success_items:
        st.session_state.run_history = batch_success_items[::-1] + st.session_state.run_history

    status.empty()
    progress.empty()

    if batch_success_items:
        st.success(f"批量测试完成，成功 {len(batch_success_items)} 条。")
    if batch_failures:
        for message in batch_failures:
            st.error(message)

st.divider()

with st.expander("运行方式"):
    st.markdown(
        """
建议环境：
- Python 3.10+
- 已安装 `streamlit openai python-dotenv pandas`
- 项目根目录存在 `.env`
- `.env` 中至少包含 `OPENAI_API_KEY=你的key`
        """
    )
    st.code(
        "\n".join(
            [
                "python -m venv .venv",
                "source .venv/bin/activate",
                "pip install -U pip",
                "pip install streamlit openai python-dotenv pandas",
                "streamlit run test_openai_api_personal.py",
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

with st.expander("建议测试方式"):
    st.markdown(
        """
1. 先在“人物库”里确认客户参数。
2. 在“参考图”里勾选本地 PCCS 和风格象限图。
3. 在“任务测试”里直接填用户原话，模型会自动判断问题类型。
4. 上传任务图片或填写图片 URL。
5. 先看 Prompt 是否符合你的业务逻辑，再发起测试。
        """
    )
