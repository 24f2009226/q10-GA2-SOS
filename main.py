import json
import os
import re

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

# Optional LLM fallback (e.g. your Task 9 Ollama tunnel). Leave unset to use rules only.
LLM_URL = os.getenv("OLLAMA_URL", "")  # e.g. https://xxx.trycloudflare.com/v1/chat/completions
LLM_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

app = FastAPI()


class ExtractRequest(BaseModel):
    text: str = ""


class Invoice(BaseModel):
    vendor: str
    amount: float
    currency: str
    date: str


DATE_RE = r"\b(\d{4}-\d{2}-\d{2})\b"
CUR = r"(?:USD|EUR|GBP|\$|€|£)"
NUM = r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
SUFFIX = (
    r"(?:Ltd|Limited|Inc|LLC|LLP|Corp|Corporation|GmbH|PLC|Co|Company|Industries|Group|"
    r"Holdings|Enterprises|Solutions|Technologies|Systems|Partners|Associates|Services|"
    r"Trading|AG|SA|BV|Pty)"
)
COMPANY_RE = re.compile(
    rf"((?:(?:[A-Z][\w&'\-]*|&)\s+)*?[A-Z][\w&'\-]*(?:\s+{SUFFIX}\b\.?)+)"
)
VENDOR_HINT = re.compile(
    r"(from|vendor|supplier|issued by|billed by|payee|pay to|payable to|seller|remit)", re.I
)
CUSTOMER_HINT = re.compile(r"(bill to|billed to|customer|client|invoice to)", re.I)


def find_vendor(t: str) -> str:
    matches = list(COMPANY_RE.finditer(t))
    if matches:
        for m in matches:
            before = t[max(0, m.start() - 25) : m.start()]
            if VENDOR_HINT.search(before) and not CUSTOMER_HINT.search(before):
                return m.group(1).strip()
        for m in matches:
            before = t[max(0, m.start() - 25) : m.start()]
            if not CUSTOMER_HINT.search(before):
                return m.group(1).strip()
        return matches[0].group(1).strip()
    m = re.search(
        r"(?:vendor|supplier|from|issued by|billed by|payee|pay to|payable to)\s*[:\-]?\s*"
        r"([^,;\n]+?)(?=\s+(?:for|total|amount|due|dated|on|is|invoice)\b|[,;\n]|$)",
        t,
        re.I,
    )
    return m.group(1).strip() if m else ""


def find_date(t: str) -> str:
    m = re.search(
        r"(?:due|pay(?:able)?\s+(?:by|on)|payment\s+date|deadline)[^0-9]{0,40}" + DATE_RE, t, re.I
    )
    if m:
        return m.group(1)
    dates = re.findall(DATE_RE, t)
    return dates[-1] if dates else ""


def find_currency(t: str) -> str:
    m = re.search(r"\b(USD|EUR|GBP)\b", t, re.I)
    if m:
        return m.group(1).upper()
    if "€" in t:
        return "EUR"
    if "£" in t:
        return "GBP"
    if "$" in t:
        return "USD"
    return "USD"


def to_float(s: str):
    try:
        return float(s.replace(",", ""))
    except Exception:
        return None


def find_amount(t: str):
    keyword_groups = [
        r"total",
        r"amount\s+due|balance\s+due|amount|balance",
        r"due|owed|payable|pay",
    ]
    for kw in keyword_groups:
        ms = list(re.finditer(rf"(?:{kw})\b[^0-9\n]{{0,40}}?{NUM}", t, re.I))
        if ms:
            v = to_float(ms[-1].group(1))
            if v is not None:
                return v
    ms = list(re.finditer(rf"{CUR}\s*{NUM}|{NUM}\s*{CUR}", t, re.I))
    if ms:
        g = [x for x in ms[-1].groups() if x]
        v = to_float(g[0]) if g else None
        if v is not None:
            return v
    nums = [to_float(n) for n in re.findall(NUM, t)]
    nums = [n for n in nums if n is not None and 50 <= n <= 9050]
    decimals = [n for n in nums if n != int(n)]
    if decimals:
        return decimals[0]
    return nums[0] if nums else None


async def llm_extract(text: str) -> dict:
    if not LLM_URL:
        return {}
    try:
        async with httpx.AsyncClient(timeout=25) as client:
            r = await client.post(
                LLM_URL,
                json={
                    "model": LLM_MODEL,
                    "stream": False,
                    "messages": [
                        {
                            "role": "system",
                            "content": "Extract invoice fields. Reply with ONLY JSON: "
                            '{"vendor": str, "amount": number, "currency": "USD|EUR|GBP", '
                            '"date": "YYYY-MM-DD"} where date is the payment due date.',
                        },
                        {"role": "user", "content": text},
                    ],
                },
            )
        content = r.json()["choices"][0]["message"]["content"]
        m = re.search(r"\{.*\}", content, re.S)
        return json.loads(m.group(0)) if m else {}
    except Exception:
        return {}


@app.post("/extract", response_model=Invoice)
async def extract(req: ExtractRequest):
    text = req.text or ""
    vendor, amount, currency, date = "", None, "USD", ""
    try:
        if text.strip():
            vendor = find_vendor(text)
            date = find_date(text)
            currency = find_currency(text)
            rest = re.sub(DATE_RE, " ", text)
            if vendor:
                rest = rest.replace(vendor, " ")
            amount = find_amount(rest)

            if not vendor or amount is None or not date:
                data = await llm_extract(text)
                vendor = vendor or str(data.get("vendor") or "")
                if amount is None:
                    try:
                        amount = float(data.get("amount"))
                    except Exception:
                        amount = None
                date = date or str(data.get("date") or "")
    except Exception:
        pass

    currency = currency.upper() if re.fullmatch(r"[A-Za-z]{3}", currency or "") else "USD"
    return Invoice(vendor=vendor, amount=amount if amount is not None else 0.0, currency=currency, date=date)


@app.get("/")
async def root():
    return {"status": "ok"}
