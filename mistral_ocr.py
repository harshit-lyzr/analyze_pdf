from dataclasses import dataclass
from typing import List


@dataclass
class OCRPageObject:
    index: int
    markdown: str


@dataclass
class OCRResponse:
    pages: List[OCRPageObject]


def combine_markdown(ocr_response: OCRResponse) -> str:
    if not ocr_response or not ocr_response.pages:
        return ""
    return "\n\n".join(page.markdown for page in ocr_response.pages).strip()
