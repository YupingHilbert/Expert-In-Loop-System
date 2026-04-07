import base64
import hashlib
import io
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
import httpx
from dotenv import load_dotenv
from openai import APITimeoutError, OpenAI, PermissionDeniedError

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

APP_TITLE = "高级形象顾问 AI 多轮对话测试台"
PROMPT_VERSION_DIR = Path("prompt_versions")
INSTRUCTION_VERSION_DIR = PROMPT_VERSION_DIR / "instructions"
MASTER_PROMPT_VERSION_DIR = PROMPT_VERSION_DIR / "master_prompts"
LOG_DIR = Path("logs_style_consultant_multiturn")
ERROR_LOG_DIR = LOG_DIR / "errors"
GENERATED_IMAGE_DIR = LOG_DIR / "generated_images"
INSTRUCTION_VERSION_DIR.mkdir(parents=True, exist_ok=True)
MASTER_PROMPT_VERSION_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
ERROR_LOG_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

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

DEFAULT_MASTER_PROMPT = "请基于用户问题、客户参数和本轮图片进行分析，并给出适合当前问题的回答。"
DEFAULT_INSTRUCTION_VERSION = "instruction_v0"
DEFAULT_MASTER_PROMPT_VERSION = "prompt_v0"

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


def normalize_version_name(version_name: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"_", "-", "."} else "_" for char in version_name.strip())
    return cleaned.strip("_") or "untitled_version"


def build_version_file_path(version_name: str, version_type: str) -> Path:
    safe_name = normalize_version_name(version_name)
    base_dir = INSTRUCTION_VERSION_DIR if version_type == "instructions" else MASTER_PROMPT_VERSION_DIR
    return base_dir / f"{safe_name}.txt"


def load_version_file(version_name: str, version_type: str) -> str:
    path = build_version_file_path(version_name, version_type)
    if not path.exists():
        raise FileNotFoundError(f"未找到版本文件：{path.resolve()}")
    return path.read_text(encoding="utf-8")


def save_version_file(version_name: str, content: str, version_type: str) -> str:
    path = build_version_file_path(version_name, version_type)
    path.write_text(content, encoding="utf-8")
    return str(path.resolve())


def load_initial_prompt_content(version_name: str, version_type: str, fallback: str) -> str:
    try:
        return load_version_file(version_name, version_type)
    except Exception:
        return fallback


def init_session_state() -> None:
    defaults = {
        "model": "gpt-5.4",
        "instruction_version": DEFAULT_INSTRUCTION_VERSION,
        "instructions": load_initial_prompt_content(
            DEFAULT_INSTRUCTION_VERSION,
            "instructions",
            DEFAULT_INSTRUCTIONS,
        ),
        "master_prompt_version": DEFAULT_MASTER_PROMPT_VERSION,
        "master_prompt": load_initial_prompt_content(
            DEFAULT_MASTER_PROMPT_VERSION,
            "master_prompt",
            DEFAULT_MASTER_PROMPT,
        ),
        "persons_jsonl": DEFAULT_PERSONS_JSONL,
        "reasoning_effort": "low",
        "verbosity": "low",
        "max_output_tokens": 1800,
        "timeout_seconds": 120,
        "temperature": 0.2,
        "top_p": 1.0,
        "include_local_reference_files": True,
        "reference_image_urls_text": "",
        "prompt_load_message": "",
        "prompt_load_error": "",
        "conversation_session_id": "",
        "conversation_started_at": "",
        "conversation_updated_at": "",
        "conversation_turns": [],
        "conversation_last_response_id": "",
        "conversation_log_path": "",
        "conversation_title": "",
        "conversation_title_input": "",
        "conversation_person_id": "",
        "conversation_person_snapshot": {},
        "turn_library_alias_map": {},
        "turn_library_next_index": 1,
        "turn_library_assets": [],
        "auto_image_generation_enabled": True,
        "auto_image_generation_model": "gpt-image-1.5",
        "auto_image_generation_quality": "high",
        "auto_image_generation_size": "1024x1536",
        "auto_image_generation_template": "真人上身",
        "planned_turn_count": 6,
        "selected_person_id": "",
        "current_user_message": "",
        "current_turn_image_urls_text": "",
        "pending_clear_turn_inputs": False,
        "turn_input_nonce": 0,
        "history_viewer_selected_file": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def process_pending_widget_resets() -> None:
    if st.session_state.pending_clear_turn_inputs:
        st.session_state.current_user_message = ""
        st.session_state.current_turn_image_urls_text = ""
        st.session_state.pending_clear_turn_inputs = False


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
            if item["label"] and item["label"] != raw_ref:
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
    existing_assets: List[Dict[str, Any]] = [dict(item) for item in st.session_state.turn_library_assets]
    assets_by_signature: Dict[str, Dict[str, Any]] = {
        str(item.get("signature")): dict(item)
        for item in existing_assets
        if item.get("signature")
    }
    alias_map: Dict[str, str] = dict(st.session_state.turn_library_alias_map)
    next_index = int(st.session_state.turn_library_next_index)

    for uploaded_file in uploaded_files:
        signature = get_uploaded_file_signature(uploaded_file)
        existing_asset = assets_by_signature.get(signature)
        if existing_asset:
            continue
        alias = alias_map.get(signature)
        if not alias:
            alias = f"IMG{next_index:03d}"
            alias_map[signature] = alias
            next_index += 1
        assets_by_signature[signature] = {
            "signature": signature,
            "alias": alias,
            "file_name": uploaded_file.name,
            "label": f"{alias} - {uploaded_file.name}",
            "image_url": uploaded_file_to_data_url(uploaded_file),
            "preview_source": uploaded_file.getvalue(),
            "source_type": f"{source_prefix}_upload_library",
        }

    assets = sorted(
        assets_by_signature.values(),
        key=lambda item: str(item.get("alias", "")),
    )

    st.session_state.turn_library_assets = assets
    st.session_state.turn_library_alias_map = alias_map
    st.session_state.turn_library_next_index = next_index
    return assets


def serialize_library_assets(images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    serialized: List[Dict[str, Any]] = []
    for item in images:
        serialized.append(
            {
                "signature": item.get("signature"),
                "alias": item.get("alias"),
                "file_name": item.get("file_name"),
                "label": item.get("label"),
                "image_url": item.get("image_url"),
                "source_type": item.get("source_type"),
            }
        )
    return serialized


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


def build_task_assets(
    task_image_url_text: str,
    uploaded_task_files: List[Any],
    alias_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    assets = build_assets_from_text(task_image_url_text, "任务图", "task", alias_lookup=alias_lookup)

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


def build_person_system_block(person: Dict[str, Any]) -> str:
    lines = ["【用户持久化信息】", "以下信息属于该用户的稳定画像，请在整轮判断中持续生效："]
    for key in PERSON_ORDERED_KEYS:
        if key in person:
            lines.append(f"- {key}: {person.get(key, '')}")
    extra_keys = [key for key in person.keys() if key not in PERSON_ORDERED_KEYS]
    for key in extra_keys:
        lines.append(f"- {key}: {person.get(key, '')}")
    return "\n".join(lines)


def build_request_instructions(instruction_version: str, base_instructions: str, person: Dict[str, Any]) -> str:
    return f"""【Instructions Version】
- {instruction_version}

{base_instructions.strip()}

{build_person_system_block(person)}
"""


def build_image_generation_policy_block(enabled: bool) -> str:
    if not enabled:
        return ""
    return """

【示意图策略】
系统后续可能根据当前对话与图片决定是否额外生成示意图。
你当前这一步不需要自己输出技术参数，也不需要描述调用工具过程。
你的任务仍然是先把分析与结论说清楚。

【高保真保留规则】
如果本轮已经提供了服饰实拍图或商品图，并且系统后续决定生成示意图：
- 优先基于现有服饰图做编辑，不要从零重画
- 尽量保留原服饰的版型、轮廓、长度、领口、门襟、纽扣数量、织纹、针织机理、褶裥、腰带宽度与主要细节
- 不要把真实服饰改成插画感、卡通感、3D 渲染感
- 不要擅自替换主面料、主颜色、关键结构和明显的小设计
- 如果某些细节无法完全保留，要优先保留服饰本体，而不是追求夸张氛围
""".strip()


def combine_request_images(task_assets: List[Dict[str, Any]], reference_assets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return dedupe_assets(task_assets + reference_assets)


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
        session_library_count = len(st.session_state.get("turn_library_assets", []) or [])
        if session_library_count > 0:
            lines.append(f"- 本轮没有新上传任务图片；但当前会话图片库中已有 {session_library_count} 张历史图片，可在需要时沿用。")
        else:
            lines.append("- 本轮没有上传任务图片，请仅根据文字和人物参数回答。")

    return "\n".join(lines)


def build_session_candidate_prompt_block(candidate_assets: List[Dict[str, Any]]) -> str:
    if not candidate_assets:
        return ""
    lines = ["【会话可复用历史图片】"]
    for item in candidate_assets:
        lines.append(
            f"- ref={item.get('selector_ref')} / source={item.get('source_scope')} / label={item.get('label')}"
        )
    return "\n".join(lines)


def build_final_prompt(
    master_prompt: str,
    question: str,
    reference_assets: List[Dict[str, Any]],
    task_assets: List[Dict[str, Any]],
    session_candidate_assets: Optional[List[Dict[str, Any]]] = None,
) -> str:
    resolved_question = question.strip()
    return f"""{master_prompt.strip()}

【用户问题】
{resolved_question}

{build_image_manifest(reference_assets, task_assets)}

{build_session_candidate_prompt_block(session_candidate_assets or [])}
"""


def asset_to_input_image_url(asset: Dict[str, Any]) -> str:
    image_url = str(asset.get("image_url") or "").strip()
    if image_url:
        return image_url

    preview_source = asset.get("preview_source")
    file_name = str(asset.get("file_name") or asset.get("label") or "image.png")
    mime_type = guess_mime_type(file_name)

    if isinstance(preview_source, bytes):
        encoded = base64.b64encode(preview_source).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"
    if isinstance(preview_source, str) and preview_source.startswith(("http://", "https://")):
        return preview_source
    if isinstance(preview_source, str) and Path(preview_source).exists():
        encoded = base64.b64encode(Path(preview_source).read_bytes()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    raise ValueError(f"素材缺少可用于模型输入的图片地址：{asset.get('label')}")


def build_input_content(prompt: str, image_assets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "input_text", "text": prompt}]
    for item in image_assets:
        content.append({"type": "input_image", "image_url": asset_to_input_image_url(item)})
    return content


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


def save_generated_image_file(base64_payload: str, session_id: str, turn_index: int, image_index: int) -> str:
    session_dir = GENERATED_IMAGE_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    file_path = session_dir / f"turn_{turn_index:03d}_generated_{image_index:02d}.png"
    file_path.write_bytes(base64.b64decode(base64_payload))
    return str(file_path.resolve())


def extract_generated_images(response, session_id: str, turn_index: int) -> List[Dict[str, Any]]:
    output_items = getattr(response, "output", None) or []
    generated_images: List[Dict[str, Any]] = []

    for item in output_items:
        item_type = getattr(item, "type", None)
        if item_type != "image_generation_call":
            continue

        result_payload = getattr(item, "result", None)
        if not result_payload:
            continue

        if isinstance(result_payload, str):
            base64_images = [result_payload]
        elif isinstance(result_payload, list):
            base64_images = [value for value in result_payload if isinstance(value, str)]
        else:
            base64_images = []

        for image_index, base64_image in enumerate(base64_images, start=1):
            file_path = save_generated_image_file(base64_image, session_id, turn_index, image_index)
            generated_images.append(
                {
                    "label": f"第 {turn_index} 轮示意图 {image_index}",
                    "file_path": file_path,
                    "preview_source": file_path,
                    "image_generation_call_id": getattr(item, "id", None),
                    "revised_prompt": getattr(item, "revised_prompt", None),
                }
            )

    return generated_images


def serialize_generated_images(images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    serialized: List[Dict[str, Any]] = []
    for item in images:
        serialized.append(
            {
                "label": item.get("label"),
                "file_path": item.get("file_path"),
                "image_generation_call_id": item.get("image_generation_call_id"),
                "revised_prompt": item.get("revised_prompt"),
            }
        )
    return serialized


def extract_json_object(text: str) -> Dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
        candidate = re.sub(r"\s*```$", "", candidate)
    try:
        return json.loads(candidate)
    except Exception:
        match = re.search(r"\{.*\}", candidate, re.DOTALL)
        if not match:
            raise ValueError("未找到可解析的 JSON 对象。")
        return json.loads(match.group(0))


def ensure_string_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def compact_join(items: List[str], default: str = "未明确") -> str:
    cleaned = [str(item).strip() for item in items if str(item).strip()]
    return "；".join(cleaned) if cleaned else default


def build_generation_decision_prompt(
    user_message: str,
    assistant_reply: str,
) -> str:
    return f"""你是一个严格控制成本的图像生成决策器。
请判断当前这一轮是否真的需要生成“高保真商品示意图”。

生成条件：
- 用户明确要求看示意图 / 效果图 / 上身图 / 生成图
- 纯文字不足以表达最终建议
- 本轮已经有清晰的商品图，可用于高保真编辑

不生成条件：
- 用户只是问适不适合、为什么、怎么选
- 文字已经足够
- 没有商品图，不适合做高保真编辑

如果要生成，目标是：
- 保持商品本身尽量不变
- 保留版型、长度、颜色、材质、纹理、纽扣、褶裥、腰带和关键结构
- 生成真人上身或穿搭示意，不要卡通化

只返回 JSON，不要输出别的内容：
{{
  "should_generate": true or false,
  "reason": "一句话",
  "edit_prompt": "如果 should_generate=true，这里给出高保真编辑提示词；否则给空字符串"
}}

【用户本轮消息】
{user_message}

【本轮助手回答】
{assistant_reply}
"""


def build_simple_edit_prompt(
    user_message: str,
    assistant_reply: str,
    planner_prompt: str,
    template_name: str,
    task_assets: List[Dict[str, Any]],
) -> str:
    template_map = {
        "真人上身": """基于输入商品图生成真实摄影风格的真人上身图。
保持商品款式、结构、颜色、比例和材质不变，不要改成新款。
不要卡通化，不要插画化，不要过度柔焦，保留原图的关键设计和面料纹理。
色彩必须与原图尽量完全一致，不要自动校色，不要把颜色变深、变灰、变脏、变冷或变暖。""",
        "商品保真棚拍": """基于输入商品图生成高保真品牌棚拍效果图。
保持商品本身不变，只优化为更真实的服装呈现与陈列效果。
不要新增设计，不要改变结构，不要卡通化。
色彩必须与原图尽量完全一致，不要自动校色，不要改变亮度、饱和度和冷暖。""",
        "场景化穿搭图": """基于输入商品图生成真实摄影风格的穿搭场景图。
商品必须保持原样，只改变模特穿着与画面场景。
不要改版型、不要改材质、不要卡通化，主体始终是原商品。
色彩必须与原图尽量完全一致，不要自动校色，不要改变亮度、饱和度和冷暖。""",
    }
    base_prompt = template_map.get(template_name, template_map["真人上身"])
    if len(task_assets) >= 2:
        merge_instruction = """多图编辑规则：
- 第 1 张图是基底图，请把它当成主要被编辑的原图
- 优先保留第 1 张图里的主体服饰、颜色、面料观感、宽松程度、垂坠感、光线和整体气质
- 后续图片只作为补充服饰参考，用来把缺少的单品接入整体造型
- 不要因为引入后续图片而改深第 1 张图的颜色，不要把宽松版型改得更紧
- 如果第 1 张图已经有上衣，就尽量像在第 1 张图上继续编辑，把后续下装或配饰自然接上去
- 颜色、饱和度、明暗关系、毛绒感或挺括感，默认以第 1 张图为准；后续图片不要覆盖这些观感
- 第 1 张图里的主色、辅色、条纹顺序、蓝黄比例、明度和饱和度都要保持一致，禁止自动美化成更深或更艳的版本
"""
    else:
        merge_instruction = """单图编辑规则：
- 把输入图当成唯一基底图，尽量像在原图上直接做编辑，而不是重画一套新造型
- 优先保留原图颜色、材质、版型和宽松程度
- 颜色、亮度、饱和度和冷暖关系要与原图保持一致，禁止自动校色或风格化调色
"""
    return f"""{base_prompt}

{merge_instruction}

用户诉求：{user_message.strip() or "生成示意图"}
助手结论：{assistant_reply.strip() or "基于商品图生成高保真结果"}
生图规划：{planner_prompt.strip() or "保持商品不变，基于原图生成"}
"""

def build_generation_candidate_assets(
    current_task_assets: List[Dict[str, Any]],
    max_history_turns: int = 3,
    max_candidates: int = 10,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    seen_urls = set()

    def add_candidate(asset: Dict[str, Any], selector_ref: str, source_scope: str) -> None:
        if len(candidates) >= max_candidates:
            return
        image_url = str(asset.get("image_url") or "").strip()
        if not image_url or image_url in seen_urls:
            return
        seen_urls.add(image_url)
        candidate = dict(asset)
        candidate["selector_ref"] = selector_ref
        candidate["source_scope"] = source_scope
        candidates.append(candidate)

    for index, asset in enumerate(current_task_assets, start=1):
        selector_ref = str(asset.get("alias") or f"CUR{index:02d}")
        add_candidate(asset, selector_ref, "current_turn")

    for index, asset in enumerate(st.session_state.get("turn_library_assets", []) or [], start=1):
        selector_ref = str(asset.get("alias") or f"LIB{index:02d}")
        add_candidate(asset, selector_ref, "session_library")

    recent_turns = list(st.session_state.get("conversation_turns", []) or [])[-max_history_turns:]
    for turn in reversed(recent_turns):
        turn_index = int(turn.get("turn_index") or 0)
        for image_index, asset in enumerate(turn.get("task_assets") or [], start=1):
            rebuilt_asset = {
                "label": asset.get("label") or f"历史任务图{turn_index}_{image_index}",
                "image_url": asset.get("image_url"),
                "preview_source": asset.get("image_url"),
                "source_type": asset.get("source_type") or "history_task",
            }
            add_candidate(rebuilt_asset, f"T{turn_index:02d}I{image_index:02d}", "recent_turn")

    return candidates


def build_generation_candidate_manifest(candidates: List[Dict[str, Any]]) -> str:
    lines = ["【可用于生图的候选图片】"]
    for index, item in enumerate(candidates, start=1):
        lines.append(
            f"- 图{index}: ref={item['selector_ref']} / source={item.get('source_scope')} / label={item.get('label')}"
        )
    return "\n".join(lines)


def build_generation_selection_prompt(
    user_message: str,
    assistant_reply: str,
    candidates: List[Dict[str, Any]],
) -> str:
    return f"""你是一个会话级图片选图器。当前会话里已经有一些可复用图片，请判断是否需要生成高保真示意图，并选出最合适的基底图和补充图。

选图规则：
- 如果用户要求“看看效果”“生成效果图”“上身图”“搭配图”，优先考虑生成
- `base_image_ref` 必须是最应该被直接编辑的那张图，通常是用户最想保留质感和颜色的商品图
- `support_image_refs` 只放补充单品图，不要超过 3 张
- 如果当前轮没有新图，但候选池里有历史图，也可以直接选历史图

只返回 JSON：
{{
  "should_generate": true or false,
  "reason": "一句话",
  "edit_prompt": "一句简短规划",
  "base_image_ref": "例如 IMG001",
  "support_image_refs": ["例如 IMG002"]
}}

{build_generation_candidate_manifest(candidates)}

【用户本轮消息】
{user_message}

【本轮助手回答】
{assistant_reply}
"""


def resolve_generation_assets(
    generation_plan: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    current_task_assets: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    candidate_lookup = {str(item.get("selector_ref")): item for item in candidates}
    selected_assets: List[Dict[str, Any]] = []
    seen_urls = set()

    def append_asset(asset: Optional[Dict[str, Any]]) -> None:
        if not asset:
            return
        image_url = str(asset.get("image_url") or "").strip()
        if not image_url or image_url in seen_urls:
            return
        seen_urls.add(image_url)
        selected_assets.append(dict(asset))

    append_asset(candidate_lookup.get(str(generation_plan.get("base_image_ref") or "").strip()))
    for ref in ensure_string_list(generation_plan.get("support_image_refs"))[:3]:
        append_asset(candidate_lookup.get(ref))

    if selected_assets:
        return selected_assets
    if current_task_assets:
        return dedupe_assets([dict(item) for item in current_task_assets])
    return []


def plan_image_generation(
    client: OpenAI,
    model: str,
    user_message: str,
    assistant_reply: str,
    candidate_assets: List[Dict[str, Any]],
    timeout_seconds: float,
) -> Dict[str, Any]:
    if not candidate_assets:
        return {
            "should_generate": False,
            "reason": "会话里没有可用图片，不执行高保真编辑。",
            "edit_prompt": "",
            "base_image_ref": "",
            "support_image_refs": [],
        }

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": build_input_content(
                    build_generation_selection_prompt(user_message, assistant_reply, candidate_assets),
                    candidate_assets,
                ),
            }
        ],
        max_output_tokens=700,
        timeout=timeout_seconds,
    )
    output_text = getattr(response, "output_text", "") or ""
    parsed = extract_json_object(output_text)
    return {
        "should_generate": bool(parsed.get("should_generate")),
        "reason": str(parsed.get("reason") or ""),
        "edit_prompt": str(parsed.get("edit_prompt") or ""),
        "base_image_ref": str(parsed.get("base_image_ref") or "").strip(),
        "support_image_refs": ensure_string_list(parsed.get("support_image_refs")),
        "raw_text": output_text,
    }


def asset_to_filelike(asset: Dict[str, Any], fallback_name: str) -> io.BytesIO:
    preview_source = asset.get("preview_source")
    image_url = str(asset.get("image_url") or "")

    if isinstance(preview_source, bytes):
        file_bytes = preview_source
    elif isinstance(preview_source, str) and Path(preview_source).exists():
        file_bytes = Path(preview_source).read_bytes()
    elif image_url.startswith("data:"):
        payload = image_url.split(",", 1)[1]
        file_bytes = base64.b64decode(payload)
    elif isinstance(preview_source, str) and preview_source.startswith(("http://", "https://")):
        file_bytes = httpx.get(preview_source, timeout=30.0).content
    else:
        raise ValueError(f"无法把素材转换为可编辑图片：{asset.get('label')}")

    buffer = io.BytesIO(file_bytes)
    suffix = ".png"
    label = str(asset.get("label") or fallback_name)
    safe_label = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_") or fallback_name
    buffer.name = f"{safe_label}{suffix}"
    buffer.seek(0)
    return buffer


def call_product_consistent_image_edit(
    client: OpenAI,
    task_assets: List[Dict[str, Any]],
    edit_prompt: str,
    image_model: str,
    quality: str,
    size: str,
) -> List[Dict[str, Any]]:
    images = [asset_to_filelike(asset, f"task_{index:02d}") for index, asset in enumerate(task_assets, start=1)]

    kwargs: Dict[str, Any] = {
        "model": image_model,
        "image": images,
        "prompt": edit_prompt,
        "quality": quality,
    }
    if size != "auto":
        kwargs["size"] = size

    try:
        response = client.images.edit(**kwargs)
    except PermissionDeniedError as exc:
        if image_model == "chatgpt-image-latest" and "must be verified" in str(exc):
            fallback_kwargs = dict(kwargs)
            fallback_kwargs["model"] = "gpt-image-1.5"
            response = client.images.edit(**fallback_kwargs)
            edit_prompt = f"{edit_prompt}\n\n[system note] chatgpt-image-latest unavailable for this org; fell back to gpt-image-1.5."
        else:
            raise
    generated_images: List[Dict[str, Any]] = []
    for index, item in enumerate(getattr(response, "data", []) or [], start=1):
        base64_payload = getattr(item, "b64_json", None)
        if not base64_payload:
            continue
        file_path = save_generated_image_file(base64_payload, st.session_state.conversation_session_id, len(st.session_state.conversation_turns) + 1, index)
        generated_images.append(
            {
                "label": f"第 {len(st.session_state.conversation_turns) + 1} 轮示意图 {index}",
                "file_path": file_path,
                "preview_source": file_path,
                "revised_prompt": edit_prompt,
                "image_generation_call_id": None,
            }
        )
    return generated_images


def call_openai_turn(
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
    previous_response_id: Optional[str] = None,
    timeout_seconds: Optional[float] = None,
):
    kwargs: Dict[str, Any] = {
        "model": model,
        "input": [{"role": "user", "content": build_input_content(prompt, image_assets)}],
        "max_output_tokens": max_output_tokens,
        "text": {"verbosity": verbosity},
    }

    if instructions.strip():
        kwargs["instructions"] = instructions.strip()

    if previous_response_id:
        kwargs["previous_response_id"] = previous_response_id

    if reasoning_effort != "none":
        kwargs["reasoning"] = {"effort": reasoning_effort}
    else:
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p

    if timeout_seconds is not None:
        kwargs["timeout"] = timeout_seconds

    return client.responses.create(**kwargs)


def build_conversation_log_path(session_id: str) -> Path:
    return LOG_DIR / f"style_consult_chat_session_{session_id}.json"


def save_error_log(payload: Dict[str, Any]) -> str:
    filename = f"multiturn_error_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{uuid.uuid4().hex[:8]}.json"
    path = ERROR_LOG_DIR / filename
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return str(path.resolve())


def start_new_conversation() -> None:
    session_id = uuid.uuid4().hex[:10]
    now = datetime.now().isoformat()
    st.session_state.conversation_session_id = session_id
    st.session_state.conversation_started_at = now
    st.session_state.conversation_updated_at = now
    st.session_state.conversation_turns = []
    st.session_state.conversation_last_response_id = ""
    st.session_state.conversation_log_path = str(build_conversation_log_path(session_id).resolve())
    st.session_state.conversation_person_id = ""
    st.session_state.conversation_person_snapshot = {}
    st.session_state.conversation_title = ""
    st.session_state.conversation_title_input = ""
    st.session_state.turn_library_alias_map = {}
    st.session_state.turn_library_next_index = 1
    st.session_state.turn_library_assets = []
    st.session_state.current_user_message = ""
    st.session_state.current_turn_image_urls_text = ""
    st.session_state.pending_clear_turn_inputs = False
    st.session_state.turn_input_nonce = 0


def ensure_conversation_session() -> None:
    if not st.session_state.conversation_session_id:
        start_new_conversation()


def bind_conversation_person(selected_person: Dict[str, Any]) -> Optional[str]:
    locked_person_id = st.session_state.conversation_person_id
    current_person_id = str(selected_person.get("person_id", "") or "")

    if locked_person_id and locked_person_id != current_person_id:
        return f"当前会话已绑定客户 `{locked_person_id}`。如需切换人物，请点击“新建会话”。"

    if not locked_person_id:
        st.session_state.conversation_person_id = current_person_id
        st.session_state.conversation_person_snapshot = dict(selected_person)
        if not st.session_state.conversation_title.strip():
            draft_title = str(st.session_state.get("conversation_title_input", "") or "").strip()
            st.session_state.conversation_title = draft_title or f"{current_person_id}_{st.session_state.conversation_session_id}"

    return None


def build_conversation_payload(
    selected_person: Dict[str, Any],
    reference_assets: List[Dict[str, Any]],
) -> Dict[str, Any]:
    person_snapshot = st.session_state.conversation_person_snapshot or dict(selected_person)
    turns = st.session_state.conversation_turns
    status = "completed" if len(turns) >= int(st.session_state.planned_turn_count) else "active"
    return {
        "session_id": st.session_state.conversation_session_id,
        "title": (
            str(st.session_state.get("conversation_title_input", "") or "").strip()
            or st.session_state.conversation_title.strip()
            or st.session_state.conversation_session_id
        ),
        "status": status,
        "started_at": st.session_state.conversation_started_at,
        "updated_at": st.session_state.conversation_updated_at,
        "planned_turn_count": int(st.session_state.planned_turn_count),
        "completed_turn_count": len(turns),
        "instruction_version": st.session_state.instruction_version,
        "master_prompt_version": st.session_state.master_prompt_version,
        "instructions": st.session_state.instructions,
        "master_prompt": st.session_state.master_prompt,
        "person": person_snapshot,
        "reference_assets": serialize_images_for_log(reference_assets),
        "turn_library_assets": serialize_library_assets(st.session_state.turn_library_assets),
        "params": {
            "model": st.session_state.model,
            "reasoning_effort": st.session_state.reasoning_effort,
            "verbosity": st.session_state.verbosity,
            "auto_image_generation_enabled": st.session_state.auto_image_generation_enabled,
            "auto_image_generation_model": st.session_state.auto_image_generation_model,
            "auto_image_generation_quality": st.session_state.auto_image_generation_quality,
            "auto_image_generation_size": st.session_state.auto_image_generation_size,
            "auto_image_generation_template": st.session_state.auto_image_generation_template,
            "temperature": st.session_state.temperature if st.session_state.reasoning_effort == "none" else None,
            "top_p": st.session_state.top_p if st.session_state.reasoning_effort == "none" else None,
            "max_output_tokens": st.session_state.max_output_tokens,
            "timeout_seconds": st.session_state.timeout_seconds,
        },
        "last_response_id": st.session_state.conversation_last_response_id or None,
        "turns": turns,
    }


def save_conversation_payload(selected_person: Dict[str, Any], reference_assets: List[Dict[str, Any]]) -> str:
    bind_error = bind_conversation_person(selected_person)
    if bind_error:
        raise ValueError(bind_error)
    payload = build_conversation_payload(selected_person, reference_assets)
    path = build_conversation_log_path(st.session_state.conversation_session_id)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    resolved = str(path.resolve())
    st.session_state.conversation_log_path = resolved
    return resolved


def send_current_turn(
    selected_person: Dict[str, Any],
    reference_assets: List[Dict[str, Any]],
    task_assets: List[Dict[str, Any]],
) -> Dict[str, Any]:
    bind_error = bind_conversation_person(selected_person)
    if bind_error:
        return {"ok": False, "error": bind_error, "error_log_path": ""}
    bound_person = st.session_state.conversation_person_snapshot or dict(selected_person)
    question = st.session_state.current_user_message.strip()
    all_images = combine_request_images(task_assets, reference_assets)
    generation_candidate_assets = build_generation_candidate_assets(task_assets)
    request_instructions = build_request_instructions(
        instruction_version=st.session_state.instruction_version,
        base_instructions=st.session_state.instructions,
        person=bound_person,
    )
    generation_policy = build_image_generation_policy_block(st.session_state.auto_image_generation_enabled)
    if generation_policy:
        request_instructions = f"{request_instructions.rstrip()}\n\n{generation_policy}\n"
    final_prompt = build_final_prompt(
        master_prompt=st.session_state.master_prompt,
        question=question,
        reference_assets=reference_assets,
        task_assets=task_assets,
        session_candidate_assets=generation_candidate_assets,
    )

    run_timestamp = datetime.now().isoformat()
    request_trace_id = str(uuid.uuid4())
    previous_response_id = st.session_state.conversation_last_response_id or None
    request_timeout = float(st.session_state.timeout_seconds)
    request_max_output_tokens = int(st.session_state.max_output_tokens)
    request_attempts: List[Dict[str, Any]] = [
        {
            "attempt": 1,
            "timeout_seconds": request_timeout,
            "max_output_tokens": request_max_output_tokens,
            "previous_response_id": previous_response_id,
        }
    ]
    client = get_client(request_timeout)
    try:
        try:
            response = call_openai_turn(
                client=client,
                model=st.session_state.model,
                prompt=final_prompt,
                image_assets=all_images,
                reasoning_effort=st.session_state.reasoning_effort,
                max_output_tokens=request_max_output_tokens,
                temperature=st.session_state.temperature if st.session_state.reasoning_effort == "none" else None,
                top_p=st.session_state.top_p if st.session_state.reasoning_effort == "none" else None,
                verbosity=st.session_state.verbosity,
                instructions=request_instructions,
                previous_response_id=previous_response_id,
                timeout_seconds=request_timeout,
            )
        except APITimeoutError:
            retry_timeout = max(request_timeout * 2, 180.0)
            retry_max_output_tokens = min(request_max_output_tokens, 1200)
            request_attempts.append(
                {
                    "attempt": 2,
                    "timeout_seconds": retry_timeout,
                    "max_output_tokens": retry_max_output_tokens,
                    "previous_response_id": previous_response_id,
                    "reason": "retry_after_timeout",
                }
            )
            response = call_openai_turn(
                client=get_client(retry_timeout),
                model=st.session_state.model,
                prompt=final_prompt,
                image_assets=all_images,
                reasoning_effort=st.session_state.reasoning_effort,
                max_output_tokens=retry_max_output_tokens,
                temperature=st.session_state.temperature if st.session_state.reasoning_effort == "none" else None,
                top_p=st.session_state.top_p if st.session_state.reasoning_effort == "none" else None,
                verbosity=st.session_state.verbosity,
                instructions=request_instructions,
                previous_response_id=previous_response_id,
                timeout_seconds=retry_timeout,
            )
        serialized = serialize_response(response)
        turn_index = len(st.session_state.conversation_turns) + 1
        generation_plan: Dict[str, Any] = {
            "should_generate": False,
            "reason": "",
            "edit_prompt": "",
        }
        generated_images: List[Dict[str, Any]] = []
        if st.session_state.auto_image_generation_enabled:
            generation_plan = plan_image_generation(
                client=client,
                model=st.session_state.model,
                user_message=question,
                assistant_reply=serialized.get("output_text_plain") or serialized.get("output_text") or "",
                candidate_assets=generation_candidate_assets,
                timeout_seconds=float(st.session_state.timeout_seconds),
            )
            if generation_plan.get("should_generate") and generation_plan.get("edit_prompt"):
                selected_generation_assets = resolve_generation_assets(
                    generation_plan=generation_plan,
                    candidates=generation_candidate_assets,
                    current_task_assets=task_assets,
                )
                generation_plan["selected_image_refs"] = [
                    str(item.get("selector_ref") or item.get("alias") or item.get("label") or "")
                    for item in selected_generation_assets
                ]
                generation_plan["edit_prompt"] = build_simple_edit_prompt(
                    user_message=question,
                    assistant_reply=serialized.get("output_text_plain") or serialized.get("output_text") or "",
                    planner_prompt=str(generation_plan["edit_prompt"]),
                    template_name=str(st.session_state.auto_image_generation_template),
                    task_assets=selected_generation_assets,
                )
                generated_images = call_product_consistent_image_edit(
                    client=client,
                    task_assets=selected_generation_assets,
                    edit_prompt=str(generation_plan["edit_prompt"]),
                    image_model=str(st.session_state.auto_image_generation_model),
                    quality=str(st.session_state.auto_image_generation_quality),
                    size=str(st.session_state.auto_image_generation_size),
                )
        turn_record = {
            "turn_index": turn_index,
            "timestamp": run_timestamp,
            "request_trace_id": request_trace_id,
            "instruction_version": st.session_state.instruction_version,
            "master_prompt_version": st.session_state.master_prompt_version,
            "user_message": question,
            "request_instructions": request_instructions,
            "final_prompt": final_prompt,
            "previous_response_id": previous_response_id,
            "request_attempts": request_attempts,
            "all_images": serialize_images_for_log(all_images),
            "reference_assets": serialize_images_for_log(reference_assets),
            "task_assets": serialize_images_for_log(task_assets),
            "generation_candidate_assets": serialize_images_for_log(generation_candidate_assets),
            "generation_plan": generation_plan,
            "generated_images": serialize_generated_images(generated_images),
            "params": {
                "model": st.session_state.model,
                "reasoning_effort": st.session_state.reasoning_effort,
                "verbosity": st.session_state.verbosity,
                "auto_image_generation_enabled": st.session_state.auto_image_generation_enabled,
                "auto_image_generation_model": st.session_state.auto_image_generation_model,
                "auto_image_generation_quality": st.session_state.auto_image_generation_quality,
                "auto_image_generation_size": st.session_state.auto_image_generation_size,
                "auto_image_generation_template": st.session_state.auto_image_generation_template,
                "temperature": st.session_state.temperature if st.session_state.reasoning_effort == "none" else None,
                "top_p": st.session_state.top_p if st.session_state.reasoning_effort == "none" else None,
                "max_output_tokens": st.session_state.max_output_tokens,
                "timeout_seconds": st.session_state.timeout_seconds,
            },
            "response": serialized,
        }
        st.session_state.conversation_turns = st.session_state.conversation_turns + [turn_record]
        st.session_state.conversation_last_response_id = serialized.get("id") or ""
        st.session_state.conversation_updated_at = run_timestamp
        log_path = save_conversation_payload(selected_person, reference_assets)
        st.session_state.pending_clear_turn_inputs = True
        st.session_state.turn_input_nonce = int(st.session_state.turn_input_nonce) + 1
        return {"ok": True, "turn_record": turn_record, "log_path": log_path}
    except Exception as exc:
        error_payload = {
            "timestamp": run_timestamp,
            "session_id": st.session_state.conversation_session_id,
            "request_trace_id": request_trace_id,
            "person": bound_person,
            "instruction_version": st.session_state.instruction_version,
            "master_prompt_version": st.session_state.master_prompt_version,
            "question": question,
            "request_instructions": request_instructions,
            "final_prompt": final_prompt,
            "previous_response_id": previous_response_id,
            "request_attempts": request_attempts,
            "error": repr(exc),
        }
        error_log_path = save_error_log(error_payload)
        return {"ok": False, "error": str(exc), "error_log_path": error_log_path}


def load_instructions_into_editor() -> None:
    try:
        st.session_state.instructions = load_version_file(st.session_state.instruction_version, "instructions")
        st.session_state.prompt_load_message = "Instructions 已从版本文件载入。"
        st.session_state.prompt_load_error = ""
    except Exception as exc:
        st.session_state.prompt_load_error = str(exc)
        st.session_state.prompt_load_message = ""


def load_master_prompt_into_editor() -> None:
    try:
        st.session_state.master_prompt = load_version_file(st.session_state.master_prompt_version, "master_prompt")
        st.session_state.prompt_load_message = "Master Prompt 已从版本文件载入。"
        st.session_state.prompt_load_error = ""
    except Exception as exc:
        st.session_state.prompt_load_error = str(exc)
        st.session_state.prompt_load_message = ""


def render_version_badge(instruction_version: Optional[str], master_prompt_version: Optional[str]) -> None:
    st.caption(
        f"版本标记：Instructions = `{instruction_version or '未设置'}` | "
        f"Master Prompt = `{master_prompt_version or '未设置'}`"
    )


def render_image_grid(title: str, images: List[Dict[str, Any]]) -> None:
    st.markdown(f"**{title}**")
    if not images:
        st.caption("暂无图片。")
        return

    st.image(
        [item["preview_source"] for item in images],
        caption=[item["label"] for item in images],
        use_container_width=True,
    )


def render_batch_library_panel(images: List[Dict[str, Any]]) -> None:
    st.markdown("**会话图片库**")
    if not images:
        st.info("先上传一批会话图片，右侧会显示可滚动的图片 ID、文件名和缩略图。")
        return

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "序号": index,
                    "id": item["alias"],
                    "file_name": item["file_name"],
                }
                for index, item in enumerate(images, start=1)
            ]
        ),
        use_container_width=True,
        hide_index=True,
        height=220,
    )

    with st.container(border=True, height=640):
        for index, item in enumerate(images, start=1):
            with st.container(border=True):
                st.markdown(f"**{index}. `{item['alias']}`**")
                st.caption(item["file_name"])
                st.image(item["preview_source"], use_container_width=True)


def render_generated_image_block(images: List[Dict[str, Any]]) -> None:
    if not images:
        return

    st.markdown("**模型生成的示意图**")
    st.image(
        [item["file_path"] for item in images if item.get("file_path")],
        caption=[item["label"] for item in images if item.get("file_path")],
        use_container_width=True,
    )

    for item in images:
        if item.get("revised_prompt"):
            with st.expander(f"查看 {item['label']} 的生成提示词", expanded=False):
                st.code(item["revised_prompt"], language=None)

def render_chat_turn(turn: Dict[str, Any]) -> None:
    with st.chat_message("user"):
        st.markdown(turn["user_message"] or "<empty>")
        if turn.get("task_assets"):
            task_assets = [
                {
                    "label": item.get("label"),
                    "preview_source": item.get("image_url"),
                }
                for item in turn["task_assets"]
            ]
            render_image_grid("本轮用户图片", task_assets)

    with st.chat_message("assistant"):
        st.markdown(turn["response"].get("output_text_plain") or turn["response"].get("output_text") or "<empty>")
        render_generated_image_block(turn.get("generated_images") or [])
        generation_plan = turn.get("generation_plan") or {}
        with st.expander(f"查看第 {turn['turn_index']} 轮底层记录", expanded=False):
            st.write(
                {
                    "turn_index": turn["turn_index"],
                    "timestamp": turn["timestamp"],
                    "request_id": turn["response"].get("request_id"),
                    "response_id": turn["response"].get("id"),
                    "model": turn["response"].get("model"),
                    "previous_response_id": turn.get("previous_response_id"),
                    "generation_plan": generation_plan,
                }
            )
            st.code(turn["final_prompt"], language=None)
            st.code(turn["request_instructions"], language=None)


def list_saved_session_files(limit: int = 50) -> List[Path]:
    files = sorted(LOG_DIR.glob("style_consult_chat_session_*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    return files[:limit]


def build_history_dataframe(files: List[Path]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows.append(
            {
                "session_id": payload.get("session_id"),
                "title": payload.get("title"),
                "status": payload.get("status"),
                "person_id": (payload.get("person") or {}).get("person_id"),
                "planned_turn_count": payload.get("planned_turn_count"),
                "completed_turn_count": payload.get("completed_turn_count"),
                "updated_at": payload.get("updated_at"),
                "file_path": str(path.resolve()),
            }
        )
    return pd.DataFrame(rows)


init_session_state()
ensure_conversation_session()
process_pending_widget_resets()

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("用接近线上聊天的方式，围绕固定用户画像做长窗口多轮对话测试，并把整段会话持续保存为 JSON。")

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
    st.text_area("Instructions", height=320, key="instructions")
    st.text_input("instruction_version", key="instruction_version")
    st.text_input("master_prompt_version", key="master_prompt_version")

    instruction_action_col1, instruction_action_col2 = st.columns(2)
    with instruction_action_col1:
        st.button("加载 Instructions", use_container_width=True, on_click=load_instructions_into_editor)
    with instruction_action_col2:
        if st.button("保存 Instructions", use_container_width=True):
            saved_path = save_version_file(st.session_state.instruction_version, st.session_state.instructions, "instructions")
            st.success(f"已保存到 {saved_path}")

    prompt_action_col1, prompt_action_col2 = st.columns(2)
    with prompt_action_col1:
        st.button("加载 Master Prompt", use_container_width=True, on_click=load_master_prompt_into_editor)
    with prompt_action_col2:
        if st.button("保存 Master Prompt", use_container_width=True):
            saved_path = save_version_file(st.session_state.master_prompt_version, st.session_state.master_prompt, "master_prompt")
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
    st.checkbox("允许模型自动判断是否生图", key="auto_image_generation_enabled")
    st.selectbox(
        "生图模型",
        options=["gpt-image-1.5", "chatgpt-image-latest", "gpt-image-1"],
        index=["gpt-image-1.5", "chatgpt-image-latest", "gpt-image-1"].index(st.session_state.auto_image_generation_model),
        key="auto_image_generation_model",
        disabled=not st.session_state.auto_image_generation_enabled,
    )
    st.selectbox(
        "生图模式",
        options=["真人上身", "商品保真棚拍", "场景化穿搭图"],
        index=["真人上身", "商品保真棚拍", "场景化穿搭图"].index(st.session_state.auto_image_generation_template),
        key="auto_image_generation_template",
        disabled=not st.session_state.auto_image_generation_enabled,
    )
    st.selectbox(
        "生图质量",
        options=["auto", "low", "medium", "high"],
        index=["auto", "low", "medium", "high"].index(st.session_state.auto_image_generation_quality),
        key="auto_image_generation_quality",
        disabled=not st.session_state.auto_image_generation_enabled,
    )
    st.selectbox(
        "生图尺寸",
        options=["auto", "1024x1024", "1024x1536", "1536x1024"],
        index=["auto", "1024x1024", "1024x1536", "1536x1024"].index(st.session_state.auto_image_generation_size),
        key="auto_image_generation_size",
        disabled=not st.session_state.auto_image_generation_enabled,
    )
    st.slider("max_output_tokens", 300, 4000, int(st.session_state.max_output_tokens), 100, key="max_output_tokens")
    st.slider("timeout_seconds", 10, 300, int(st.session_state.timeout_seconds), 10, key="timeout_seconds")

    st.markdown("### 采样参数")
    if st.session_state.reasoning_effort == "none":
        st.slider("temperature", 0.0, 2.0, float(st.session_state.temperature), 0.1, key="temperature")
        st.slider("top_p", 0.0, 1.0, float(st.session_state.top_p), 0.05, key="top_p")
    else:
        st.caption("当前使用 reasoning，temperature / top_p 不生效。")

    st.divider()
    st.markdown("### 当前会话")
    st.write(
        {
            "session_id": st.session_state.conversation_session_id,
            "planned_turn_count": int(st.session_state.planned_turn_count),
            "completed_turn_count": len(st.session_state.conversation_turns),
            "log_path": st.session_state.conversation_log_path,
            "logs_dir": str(LOG_DIR.resolve()),
            "auto_image_generation_enabled": bool(st.session_state.auto_image_generation_enabled),
            "api_key_loaded": bool(api_key),
        }
    )

tab_chat, tab_persons, tab_prompt, tab_references, tab_history = st.tabs(
    ["多轮对话", "人物库", "Prompt", "参考图", "会话记录"]
)

with tab_persons:
    st.subheader("人物参数库")
    st.text_area(
        "人物参数 JSONL",
        height=320,
        help="每行一个 JSON 对象，表示一个线下已完成测试的客户参数包。",
        key="persons_jsonl",
    )
    try:
        persons = parse_persons_jsonl(st.session_state.persons_jsonl)
        st.success(f"已解析 {len(persons)} 个客户参数。")
        st.dataframe(pd.DataFrame(persons), use_container_width=True)
    except Exception as exc:
        persons = []
        st.error(str(exc))

with tab_references:
    st.subheader("参考图设置")
    st.checkbox("自动附带本地参考图（PCCS + 风格象限图）", key="include_local_reference_files")
    st.text_area(
        "额外参考图 URL / 本地路径",
        height=140,
        help="每行一张图。可写成“标签 | URL/本地路径”。",
        key="reference_image_urls_text",
    )
    uploaded_reference_files = st.file_uploader(
        "上传额外参考图",
        accept_multiple_files=True,
        type=["png", "jpg", "jpeg", "webp"],
        key="reference_files_uploader",
    )
    reference_assets = build_reference_assets(
        include_local_reference_files=st.session_state.include_local_reference_files,
        reference_url_text=st.session_state.reference_image_urls_text,
        uploaded_reference_files=uploaded_reference_files or [],
    )
    render_image_grid("当前参考图预览", reference_assets)

try:
    persons = parse_persons_jsonl(st.session_state.persons_jsonl)
except Exception:
    persons = []

person_options = [person["person_id"] for person in persons]
selected_person: Optional[Dict[str, Any]] = None

if person_options:
    if st.session_state.selected_person_id not in person_options:
        st.session_state.selected_person_id = person_options[0]
    selected_person = next(
        (person for person in persons if person["person_id"] == st.session_state.selected_person_id),
        persons[0],
    )

with tab_chat:
    st.subheader("多轮对话测试")
    turn_library_assets: List[Dict[str, Any]] = []
    turn_library_lookup: Dict[str, Dict[str, Any]] = {}

    if not person_options:
        st.warning("请先在“人物库”里录入至少一个客户参数。")
    else:
        top_col1, top_col2 = st.columns([1.15, 1.0], gap="large")

        with top_col1:
            st.selectbox("选择客户", person_options, key="selected_person_id")
            selected_person = next(
                (person for person in persons if person["person_id"] == st.session_state.selected_person_id),
                persons[0],
            )
            st.number_input("计划对话轮数", min_value=1, max_value=30, step=1, key="planned_turn_count")
            st.text_input("会话标题", key="conversation_title_input", placeholder="例如：包媛媛春季衣橱长对话")
            render_version_badge(st.session_state.instruction_version, st.session_state.master_prompt_version)

            action_col1, action_col2 = st.columns(2)
            with action_col1:
                st.button("新建会话", use_container_width=True, on_click=start_new_conversation)
            with action_col2:
                if st.button("立即保存会话", use_container_width=True):
                    try:
                        save_conversation_payload(selected_person, reference_assets)
                        st.success(f"已保存到 {st.session_state.conversation_log_path}")
                    except Exception as exc:
                        st.error(str(exc))

            st.caption(
                "对话状态会在每轮模型返回后自动写入一个 JSON 文件；同一个 session 会持续覆盖更新，便于后续专家审阅整段上下文。"
            )

            if st.session_state.conversation_person_id:
                st.info(f"当前会话已绑定客户：`{st.session_state.conversation_person_id}`")

            if st.session_state.conversation_turns:
                with st.expander("当前会话概览", expanded=False):
                    st.write(
                        {
                            "session_id": st.session_state.conversation_session_id,
                            "planned_turn_count": int(st.session_state.planned_turn_count),
                            "completed_turn_count": len(st.session_state.conversation_turns),
                            "started_at": st.session_state.conversation_started_at,
                            "updated_at": st.session_state.conversation_updated_at,
                            "log_path": st.session_state.conversation_log_path,
                        }
                    )

        with top_col2:
            st.json(selected_person)
            turn_library_uploader_key = f"turn_image_library_uploader_{st.session_state.conversation_session_id}"
            uploaded_turn_library_files = st.file_uploader(
                "上传本会话图片库",
                accept_multiple_files=True,
                type=["png", "jpg", "jpeg", "webp"],
                key=turn_library_uploader_key,
                help="先把这段对话会反复使用的图片批量上传，再在右侧本轮输入里写 IMG001 / IMG002 这类短 ID。",
            )
            turn_library_assets = build_uploaded_library_assets(uploaded_turn_library_files or [], "task")
            turn_library_lookup = build_uploaded_library_lookup(turn_library_assets)
            if turn_library_assets:
                st.caption("下方聊天输入里可以直接写 IMG001、IMG002，也可以混用 URL、本地路径。")

        st.divider()

        chat_col, library_col = st.columns([1.7, 1.0], gap="large")

        with chat_col:
            for turn in st.session_state.conversation_turns:
                render_chat_turn(turn)

            current_turn_index = len(st.session_state.conversation_turns) + 1
            reached_limit = len(st.session_state.conversation_turns) >= int(st.session_state.planned_turn_count)

            if reached_limit:
                st.warning("已达到当前设置的对话轮数上限。你可以调大轮数继续，或点击“新建会话”。")
            else:
                st.markdown(f"**第 {current_turn_index} 轮用户输入**")
                st.text_area(
                    "用户本轮消息",
                    height=120,
                    key="current_user_message",
                    placeholder="例如：那如果我想明天见客户穿得更利落一点，你建议我改哪件？",
                )
                st.text_area(
                    "本轮图片 URL / 本地路径 / 图片库 ID",
                    height=120,
                    key="current_turn_image_urls_text",
                    help="每行一张图。可写成“标签 | URL/本地路径”，也可直接写 IMG001 这类图库 ID。",
                    placeholder="IMG001\n候选B | IMG002\n正面图 | /Users/xxx/Desktop/look.jpg",
                )
                turn_uploader_key = (
                    f"turn_uploader_{st.session_state.conversation_session_id}_"
                    f"{current_turn_index}_{int(st.session_state.turn_input_nonce)}"
                )
                uploaded_turn_files = st.file_uploader(
                    "上传本轮任务图片",
                    accept_multiple_files=True,
                    type=["png", "jpg", "jpeg", "webp"],
                    key=turn_uploader_key,
                )
                task_assets = build_task_assets(
                    task_image_url_text=st.session_state.current_turn_image_urls_text,
                    uploaded_task_files=uploaded_turn_files or [],
                    alias_lookup=turn_library_lookup,
                )
                render_image_grid("本轮任务图片预览", task_assets)

                if st.session_state.current_user_message.strip():
                    preview_prompt = build_final_prompt(
                        master_prompt=st.session_state.master_prompt,
                        question=st.session_state.current_user_message,
                        reference_assets=reference_assets,
                        task_assets=task_assets,
                    )
                    with st.expander("查看本轮即将发送的 Prompt", expanded=False):
                        st.code(preview_prompt, language=None)

                send_disabled = not st.session_state.current_user_message.strip()
                if st.button("发送本轮对话", type="primary", use_container_width=True, disabled=send_disabled):
                    if not api_key:
                        st.error("没有读取到 OPENAI_API_KEY，请先检查 .env。")
                        st.stop()
                    if not st.session_state.master_prompt.strip():
                        st.error("Master Prompt 不能为空。")
                        st.stop()

                    with st.spinner(f"正在发送第 {current_turn_index} 轮..."):
                        result = send_current_turn(
                            selected_person=selected_person,
                            reference_assets=reference_assets,
                            task_assets=task_assets,
                        )
                    if result["ok"]:
                        st.success(f"第 {current_turn_index} 轮已完成，并已写入 {result['log_path']}")
                        st.rerun()
                    else:
                        st.error(f"请求失败：{result['error']}")
                        if result.get("error_log_path"):
                            st.info(f"错误日志已保存：{result['error_log_path']}")

        with library_col:
            render_batch_library_panel(turn_library_assets)

with tab_prompt:
    st.subheader("主 Prompt")
    st.text_area("Master Prompt", height=420, key="master_prompt")
    if selected_person:
        request_instructions_preview = build_request_instructions(
            instruction_version=st.session_state.instruction_version,
            base_instructions=st.session_state.instructions,
            person=selected_person,
        )
        with st.expander("查看当前 Instructions 拼装结果", expanded=False):
            st.code(request_instructions_preview, language=None)

with tab_history:
    st.subheader("已保存会话")
    saved_files = list_saved_session_files()
    if not saved_files:
        st.info("还没有保存过多轮会话。先在“多轮对话”里跑至少一轮。")
    else:
        history_df = build_history_dataframe(saved_files)
        if not history_df.empty:
            st.dataframe(history_df, use_container_width=True)

        file_options = [str(path.resolve()) for path in saved_files]
        if st.session_state.history_viewer_selected_file not in file_options:
            st.session_state.history_viewer_selected_file = file_options[0]
        selected_history_file = st.selectbox(
            "选择一个会话 JSON",
            file_options,
            key="history_viewer_selected_file",
        )
        payload = json.loads(Path(selected_history_file).read_text(encoding="utf-8"))
        st.write(
            {
                "session_id": payload.get("session_id"),
                "title": payload.get("title"),
                "status": payload.get("status"),
                "person_id": (payload.get("person") or {}).get("person_id"),
                "planned_turn_count": payload.get("planned_turn_count"),
                "completed_turn_count": payload.get("completed_turn_count"),
                "updated_at": payload.get("updated_at"),
            }
        )
        with st.expander("查看完整 JSON", expanded=False):
            st.json(payload)

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
                "streamlit run test_openai_api_personal_multiturn.py",
            ]
        ),
        language="bash",
    )

with st.expander("环境变量示例"):
    st.code('OPENAI_API_KEY="sk-..."', language="bash")
