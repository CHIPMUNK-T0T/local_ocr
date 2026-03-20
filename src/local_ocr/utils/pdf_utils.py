import pypdfium2 as pdfium
from PIL import Image
from pathlib import Path
from typing import Generator


def pdf_page_generator(pdf_path: Path, dpi: int = 250) -> Generator[Image.Image, None, None]:
    """
    PDFの各ページを1枚ずつPIL Imageとして生成します（メモリ節約用）。
    """
    print(f"Opening PDF: {pdf_path.name} at {dpi} DPI...")
    pdf = pdfium.PdfDocument(str(pdf_path))
    scale = dpi / 72

    try:
        for i in range(len(pdf)):
            page = pdf.get_page(i)
            bitmap = page.render(scale=scale)
            pil_image = bitmap.to_pil()

            try:
                rgb_image = pil_image.convert("RGB")
                try:
                    # Return a detached PIL image so pdfium-backed buffers can be released immediately.
                    yield rgb_image.copy()
                finally:
                    rgb_image.close()
            finally:
                pil_image.close()
                bitmap.close()
                page.close()
    finally:
        pdf.close()
