# -*- coding: utf-8 -*-
"""
Simple_Agent.ipynb 반영 통합본 (GitHub Actions 실행용)

1) 정책브리핑 OpenAPI 수집 (3일 초과 호출 방지: 3일 단위 분할)
2) DataContents 정제 (ContentsType == 'H'인 경우 HTML -> text)
3) Mistral로 레코드별 2줄 요약 + 키워드(JSON 출력 강제, 재시도 포함)
4) Mistral로 보고서 초안 생성 (2단계: 이슈 구조화 JSON -> Markdown 보고서)
5) Markdown -> HTML 변환 후 Gmail(SMTP) 발송
   - 변환 실패 시: 발송하지 않고 로그/파일 저장만 수행

필수 환경변수(Secrets 권장):
- SERVICE_KEY
- MISTRAL_API_KEY
- GMAIL_ADDRESS
- GMAIL_APP_PASSWORD
- TO_EMAIL

선택:
- POLICY_BASE_URL (default: https://apis.data.go.kr/1371000/policyNewsService/policyNewsList)
- START_DATE / END_DATE (YYYYMMDD, 둘 다 있어야 함)
- DAYS_RANGE (default: 1)
- POLICY_KEYWORD (default: '')
- MAX_ITEMS_FOR_ANALYSIS (default: 80)
- MAX_BODY_CHARS_PER_ITEM (default: 1400)
- MISTRAL_MODEL (default: mistral-small-latest)
- OUT_DIR (default: out)
- SEND_EMAIL (default: 1)
"""

from __future__ import annotations

import os
import re
import json
import time
import smtplib
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

import requests
import pandas as pd
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from markdown import markdown as md_to_html

from mistralai import Mistral


# ----------------------------
# Config / Logging
# ----------------------------
@dataclass
class Config:
    service_key: str
    base_url: str

    mistral_api_key: str
    mistral_model: str

    gmail_address: str
    gmail_app_password: str
    to_email: str

    policy_keyword: str
    start_date: Optional[str]
    end_date: Optional[str]
    days_range: int

    max_items_for_analysis: int
    max_body_chars_per_item: int

    out_dir: Path
    send_email: bool


def _env(name: str, default: Optional[str] = None, required: bool = False) -> str:
    v = os.getenv(name, default)
    if required and (v is None or str(v).strip() == ""):
        raise RuntimeError(f"Missing required env var: {name}")
    return str(v) if v is not None else ""


def load_config() -> Config:
    start_date = _env("START_DATE", "").strip() or None
    end_date = _env("END_DATE", "").strip() or None
    if (start_date and not end_date) or (end_date and not start_date):
        raise RuntimeError("START_DATE와 END_DATE는 함께 지정해야 합니다.")

    return Config(
        service_key=_env("SERVICE_KEY", required=True),
        base_url=_env("POLICY_BASE_URL", "https://apis.data.go.kr/1371000/policyNewsService/policyNewsList"),

        mistral_api_key=_env("MISTRAL_API_KEY", required=True),
        mistral_model=_env("MISTRAL_MODEL", "mistral-small-latest"),

        gmail_address=_env("GMAIL_ADDRESS", required=True),
        gmail_app_password=_env("GMAIL_APP_PASSWORD", required=True),
        to_email=_env("TO_EMAIL", required=True),

        policy_keyword=_env("POLICY_KEYWORD", "").strip(),
        start_date=start_date,
        end_date=end_date,
        days_range=int(_env("DAYS_RANGE", "1")),

        max_items_for_analysis=int(_env("MAX_ITEMS_FOR_ANALYSIS", "80")),
        max_body_chars_per_item=int(_env("MAX_BODY_CHARS_PER_ITEM", "1400")),

        out_dir=Path(_env("OUT_DIR", "out")),
        send_email=(_env("SEND_EMAIL", "1").strip() != "0"),
    )


def setup_logging(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = out_dir / f"run_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )
    return log_path


# ----------------------------
# Utils (ipynb 반영)
# ----------------------------
def normalize_service_key(service_key: str) -> str:
    """
    노트북과 동일:
    - 이미 % 인코딩이 있으면 그대로
    - 없으면 quote로 인코딩
    """
    k = (service_key or "").strip()
    return k if "%" in k else quote(k, safe="")


def mask_url(u: str) -> str:
    return re.sub(r"(serviceKey=)[^&]+", r"\1***", u or "")


def strip_html(html: str) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text("\n")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def strip_namespaces(root: ET.Element) -> ET.Element:
    for el in root.iter():
        if isinstance(el.tag, str) and el.tag.startswith("{"):
            el.tag = el.tag.split("}", 1)[1]
    return root


def daterange_chunks(start_yyyymmdd: str, end_yyyymmdd: str, chunk_days: int = 3) -> List[Tuple[str, str]]:
    """
    노트북 반영:
    - API가 3일 초과 조회 시 에러를 반환하므로 3일 단위로 쪼개 호출
    - inclusive 구간
    """
    start = datetime.strptime(start_yyyymmdd, "%Y%m%d").date()
    end = datetime.strptime(end_yyyymmdd, "%Y%m%d").date()

    ranges = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=chunk_days - 1), end)
        ranges.append((cur.strftime("%Y%m%d"), chunk_end.strftime("%Y%m%d")))
        cur = chunk_end + timedelta(days=1)
    return ranges


def build_date_range(cfg: Config) -> Tuple[str, str]:
    if cfg.start_date and cfg.end_date:
        return cfg.start_date, cfg.end_date

    today = date.today()
    start = today - timedelta(days=cfg.days_range - 1)
    return start.strftime("%Y%m%d"), today.strftime("%Y%m%d")


# ----------------------------
# Policy News Fetch
# ----------------------------
def call_api(cfg: Config, start_date: str, end_date: str, timeout: int = 30) -> Tuple[List[Dict[str, str]], Dict[str, str], str]:
    sk = normalize_service_key(cfg.service_key)
    url = f"{cfg.base_url}?serviceKey={sk}&startDate={start_date}&endDate={end_date}"

    logging.info(f"[CALL] {start_date} ~ {end_date} | {mask_url(url)}")
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()

    rows, meta = parse_items_from_xml(resp.text)
    # 노트북과 동일하게 resultCode == "0"만 정상 취급
    if meta.get("resultCode") != "0":
        return [], meta, resp.text
    return rows, meta, resp.text


def parse_items_from_xml(xml_text: str) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
    root = ET.fromstring(xml_text)

    result_code = (root.findtext(".//resultCode") or "").strip()
    result_msg = (root.findtext(".//resultMsg") or "").strip()

    root = strip_namespaces(root)

    items = root.findall(".//item")
    if len(items) == 0:
        items = [el for el in root.iter() if isinstance(el.tag, str) and el.tag.lower().endswith("item")]

    rows: List[Dict[str, str]] = []

    for it in items:
        def t(tag: str) -> str:
            x = it.find(tag)
            return x.text.strip() if (x is not None and x.text) else ""

        contents_type = t("ContentsType")
        data_contents = t("DataContents")

        rows.append({
            "NewsItemId": t("NewsItemId"),
            "GroupingCode": t("GroupingCode"),
            "MinisterCode": t("MinisterCode"),
            "ApproveDate": t("ApproveDate"),
            "ModifyDate": t("ModifyDate"),
            "Title": t("Title"),
            "SubTitle1": t("SubTitle1"),
            "SubTitle2": t("SubTitle2"),
            "SubTitle3": t("SubTitle3"),
            "ContentsType": contents_type,
            "OriginalUrl": t("OriginalUrl"),
            "ThumbnailUrl": t("ThumbnailUrl"),
            "OriginalimgUrl": t("OriginalimgUrl"),
            # 노트북 반영: ContentsType == 'H'면 HTML 제거한 텍스트 저장
            "DataContents_text": strip_html(data_contents) if contents_type.upper() == "H" else data_contents,
            "DataContents_raw": data_contents,
        })

    meta = {"resultCode": result_code, "resultMsg": result_msg}
    return rows, meta


def fetch_policy_news(cfg: Config, start_date: str, end_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_rows: List[Dict[str, str]] = []
    errors: List[Dict[str, str]] = []

    for s, e in daterange_chunks(start_date, end_date, chunk_days=3):
        rows, meta, _ = call_api(cfg, s, e)
        if meta.get("resultCode") == "0":
            for r in rows:
                r["req_startDate"] = s
                r["req_endDate"] = e
            all_rows.extend(rows)
        else:
            errors.append({"startDate": s, "endDate": e, **meta})

    df = pd.DataFrame(all_rows)
    err_df = pd.DataFrame(errors)

    if not df.empty and "NewsItemId" in df.columns:
        df = df.drop_duplicates(subset=["NewsItemId"]).reset_index(drop=True)

    return df, err_df


def apply_keyword_filter(df: pd.DataFrame, keyword: str) -> pd.DataFrame:
    kw = (keyword or "").strip()
    if not kw or df.empty:
        return df

    cols = [c for c in ["Title", "SubTitle1", "SubTitle2", "SubTitle3", "DataContents_text"] if c in df.columns]
    blob = df[cols].fillna("").astype(str).agg(" ".join, axis=1)
    return df[blob.str.contains(re.escape(kw), na=False)].reset_index(drop=True)


# ----------------------------
# Mistral: record summary/keywords (ipynb 반영)
# ----------------------------
def truncate_text(text: str, max_chars: int) -> str:
    text = "" if text is None else str(text)
    return text if len(text) <= max_chars else text[:max_chars] + "\n...(truncated)"


def safe_json_parse(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text or "", re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group())
        except Exception:
            return None


def mistral_record_enrich(client: Mistral, model: str, title: str, body: str, max_body_chars: int,
                         sleep_sec: float = 0.3, max_retries: int = 5) -> Dict[str, str]:
    body_t = truncate_text(body, max_body_chars)

    prompt = f"""
아래 뉴스(제목+본문)를 읽고, 반드시 JSON만 출력하세요(코드블록/설명 금지).

출력 JSON 스키마:
{{
  "summary_line1": "요약 1줄",
  "summary_line2": "요약 1줄",
  "keywords": ["키워드1", "키워드2", "..."]
}}

요구사항:
- summary_line1/2는 각각 1줄, 총 2줄
- keywords는 5~12개
- 한국어로 작성

[제목]
{title}

[본문]
{body_t}
""".strip()

    for attempt in range(max_retries):
        try:
            resp = client.chat.complete(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a Korean policy/news analyst. Output JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            content = resp.choices[0].message.content or ""
            data = safe_json_parse(content)
            if not data:
                raise ValueError("JSON parse failed")

            kw_list = data.get("keywords", [])
            if isinstance(kw_list, list):
                # 중복 제거(순서 유지)
                kw_list = list(dict.fromkeys([str(x).strip() for x in kw_list if str(x).strip()]))
            else:
                kw_list = []

            return {
                "summary_line1": str(data.get("summary_line1", "")).strip(),
                "summary_line2": str(data.get("summary_line2", "")).strip(),
                "keywords": ", ".join(kw_list),
                "mistral_error": "",
            }

        except Exception as e:
            if attempt == max_retries - 1:
                return {
                    "summary_line1": "",
                    "summary_line2": "",
                    "keywords": "",
                    "mistral_error": f"{type(e).__name__}: {e}",
                }
            time.sleep(sleep_sec * (attempt + 1))

    # unreachable
    return {"summary_line1": "", "summary_line2": "", "keywords": "", "mistral_error": "unknown"}


# ----------------------------
# Mistral: report draft (ipynb 2단계 방식 반영)
# ----------------------------
def extract_json(text: str) -> dict:
    t = (text or "").strip()
    m = re.search(r"(\{.*\})", t, flags=re.DOTALL)
    if not m:
        raise ValueError("모델 출력에서 JSON을 찾지 못했습니다.")
    return json.loads(m.group(1))


def mistral_chat(client: Mistral, model: str, system: str, user: str, temperature: float = 0.2) -> str:
    resp = client.chat.complete(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()


def build_analysis_items(df: pd.DataFrame, max_items: int, max_body_chars: int) -> List[dict]:
    items = []
    for i, row in df.head(max_items).iterrows():
        title = str(row.get("Title", "")).strip()
        date_ = str(row.get("ApproveDate", "")).strip()
        url = str(row.get("OriginalUrl", "")).strip()
        body = str(row.get("DataContents_text", "")).strip()
        s1 = str(row.get("summary_line1", "")).strip()
        s2 = str(row.get("summary_line2", "")).strip()
        kw = str(row.get("keywords", "")).strip()

        summary = (s1 + "\n" + s2).strip() if (s1 or s2) else ""
        items.append({
            "idx": i + 1,
            "title": re.sub(r"\s+", " ", title),
            "date": re.sub(r"\s+", " ", date_),
            "url": url,
            "summary": summary,
            "keywords": kw,
            "body": re.sub(r"\s+", " ", body)[:max_body_chars],
        })
    return items


def generate_report_markdown(client: Mistral, model: str, analysis_items: List[dict]) -> Tuple[dict, str]:
    payload = {"items": analysis_items}

    # (1단계) 이슈 구조화 JSON
    system_1 = (
        "당신은 공공기관 보고서 작성 보조자입니다. "
        "아래 뉴스 목록을 바탕으로 핵심 이슈를 구조화해 주세요. "
        "출력은 반드시 JSON 하나만 반환하세요(코드블록/설명 금지)."
    )

    user_1 = f"""
[요청]
다음 뉴스 목록을 분석하여, 보고서에 바로 쓸 수 있도록 이슈를 구조화해 주세요.

[출력 JSON 스키마 예시]
{{
  "period": "분석 기간(가능하면 추정)",
  "top_topics": [
    {{
      "topic": "핵심 주제",
      "why_it_matters": "중요성(2~4문장)",
      "key_points": ["핵심 포인트1", "2", "3"],
      "implications_for_public_org": ["공공기관 시사점1", "2"],
      "related_items": [1, 3, 7]
    }}
  ],
  "risks_and_watchpoints": ["리스크/모니터링 포인트..."],
  "recommended_actions": ["권고 조치..."]
}}

규칙:
- related_items는 입력 목록의 idx를 참조
- 가능하면 keywords/summary를 우선 활용하고, 부족할 때 body를 참고

[뉴스 목록 JSON]
{json.dumps(payload, ensure_ascii=False)}
""".strip()

    issue_json_text = mistral_chat(client, model, system_1, user_1, temperature=0.2)
    issue_struct = extract_json(issue_json_text)

    # (2단계) Markdown 보고서 생성
    source_lines = []
    for it in analysis_items:
        d = f" ({it['date']})" if it.get("date") else ""
        u = f" | {it['url']}" if it.get("url") else ""
        source_lines.append(f"- [{it['idx']}] {it['title']}{d}{u}")

    system_2 = (
        "당신은 공공기관 보고서 작성자입니다. "
        "주어진 이슈 구조(JSON)와 출처 목록을 근거로 한국어 Markdown 보고서 초안을 작성하세요."
    )

    user_2 = f"""
[요청]
아래 이슈 구조(JSON)와 출처 목록을 기반으로 '보고서 초안'을 Markdown으로 작성하세요.
구성은 반드시 다음 순서를 지키세요:

1) 목적
2) 개요
3) 본문
4) 출처

추가 규칙:
- 본문은 top_topics를 중심으로 섹션을 구성하고, 각 섹션에 related_items 출처를 각주/참조 형태로 연결하세요.
- 과장 금지, 근거 중심, 문장 길이는 실무 보고서 수준.
- Markdown 표/목록을 적절히 활용.

[이슈 구조 JSON]
{json.dumps(issue_struct, ensure_ascii=False)}

[출처 목록]
{chr(10).join(source_lines)}
""".strip()

    report_md = mistral_chat(client, model, system_2, user_2, temperature=0.2)
    return issue_struct, report_md


# ----------------------------
# Email
# ----------------------------
def split_emails(s: str) -> List[str]:
    parts = re.split(r"[;,]", s or "")
    return [p.strip() for p in parts if p.strip()]


def send_gmail_html(subject: str, html_body: str, from_addr: str, to_addrs: List[str], app_password: str) -> None:
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = ", ".join(to_addrs)
    msg.attach(MIMEText(html_body, "html", _charset="utf-8"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=60) as server:
        server.login(from_addr, app_password)
        server.sendmail(from_addr, to_addrs, msg.as_string())


# ----------------------------
# Main
# ----------------------------
def main() -> int:
    cfg = load_config()
    log_path = setup_logging(cfg.out_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    logging.info("=== START ===")
    logging.info(f"OUT_DIR={cfg.out_dir}")
    logging.info(f"MISTRAL_MODEL={cfg.mistral_model}")
    logging.info(f"DAYS_RANGE={cfg.days_range}")
    logging.info(f"POLICY_KEYWORD='{cfg.policy_keyword}' (len={len(cfg.policy_keyword)})")
    logging.info(f"SEND_EMAIL={1 if cfg.send_email else 0}")
    logging.info(f"LOG={log_path}")

    start_date, end_date = build_date_range(cfg)
    logging.info(f"DATE_RANGE={start_date}~{end_date}")

    # 1) Fetch
    df, err_df = fetch_policy_news(cfg, start_date, end_date)
    logging.info(f"[FETCH] rows={len(df)} errors={len(err_df)}")

    if not err_df.empty:
        err_path = cfg.out_dir / f"api_errors_{ts}.csv"
        err_df.to_csv(err_path, index=False, encoding="utf-8-sig")
        logging.warning(f"[WRITE] {err_path}")

    if df.empty:
        empty_path = cfg.out_dir / f"empty_{ts}.txt"
        empty_path.write_text("No rows fetched.\n", encoding="utf-8")
        logging.warning("No rows. Exit.")
        return 0

    # 2) Keyword filter
    df2 = apply_keyword_filter(df, cfg.policy_keyword)
    logging.info(f"[FILTER] rows={len(df2)}")
    if df2.empty:
        empty_path = cfg.out_dir / f"empty_after_filter_{ts}.txt"
        empty_path.write_text("No rows after keyword filtering.\n", encoding="utf-8")
        logging.warning("No rows after filter. Exit.")
        return 0

    df2["content_length"] = df2["DataContents_text"].fillna("").astype(str).map(len)

    raw_csv = cfg.out_dir / f"policy_news_raw_{ts}.csv"
    df2.to_csv(raw_csv, index=False, encoding="utf-8-sig")
    logging.info(f"[WRITE] {raw_csv}")

    # 3) Per-record Mistral summary/keywords
    client = Mistral(api_key=cfg.mistral_api_key)

    s1_list, s2_list, kw_list, err_list = [], [], [], []
    for _, row in df2.iterrows():
        title = str(row.get("Title", "")).strip()
        body = str(row.get("DataContents_text", "")).strip()
        out = mistral_record_enrich(
            client=client,
            model=cfg.mistral_model,
            title=title,
            body=body,
            max_body_chars=cfg.max_body_chars_per_item,
            sleep_sec=0.3,
            max_retries=5,
        )
        s1_list.append(out["summary_line1"])
        s2_list.append(out["summary_line2"])
        kw_list.append(out["keywords"])
        err_list.append(out["mistral_error"])
        time.sleep(0.2)

    df2["summary_line1"] = s1_list
    df2["summary_line2"] = s2_list
    df2["keywords"] = kw_list
    df2["mistral_error"] = err_list

    out_csv = cfg.out_dir / f"policy_news_with_summary_{ts}.csv"
    df2.to_csv(out_csv, index=False, encoding="utf-8-sig")
    logging.info(f"[WRITE] {out_csv}")

    # 4) Report draft (2-step)
    analysis_items = build_analysis_items(
        df=df2,
        max_items=cfg.max_items_for_analysis,
        max_body_chars=cfg.max_body_chars_per_item,
    )

    issue_struct, report_md = generate_report_markdown(client, cfg.mistral_model, analysis_items)

    issue_json_path = cfg.out_dir / f"issue_struct_{ts}.json"
    issue_json_path.write_text(json.dumps(issue_struct, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info(f"[WRITE] {issue_json_path}")

    md_path = cfg.out_dir / f"report_draft_{ts}.md"
    md_path.write_text(report_md, encoding="utf-8")
    logging.info(f"[WRITE] {md_path}")

    # 5) Markdown -> HTML (실패 시 발송 중단)
    try:
        html = md_to_html(report_md, extensions=["extra", "tables", "sane_lists"])
        if not html or "<" not in html:
            raise ValueError("Markdown->HTML 변환 결과가 비정상")
    except Exception as e:
        logging.error("[EMAIL] Markdown->HTML 변환 실패. 이메일 발송 중단.")
        logging.error(str(e))
        skip_path = cfg.out_dir / f"email_skipped_{ts}.txt"
        skip_path.write_text(f"Email skipped due to markdown->html failure: {e}\n", encoding="utf-8")
        return 0

    html_path = cfg.out_dir / f"report_draft_{ts}.html"
    html_path.write_text(html, encoding="utf-8")
    logging.info(f"[WRITE] {html_path}")

    # 6) Send email
    if cfg.send_email:
        to_addrs = split_emails(cfg.to_email)
        subject = f"[정책뉴스 보고서] {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        send_gmail_html(subject, html, cfg.gmail_address, to_addrs, cfg.gmail_app_password)
        logging.info("[EMAIL] sent")
    else:
        logging.info("[EMAIL] skipped (SEND_EMAIL=0)")

    logging.info("=== DONE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
