"""Decrypt permission-restricted PDFs (empty user password) into reference/unlocked/."""
import os

from pypdf import PdfReader, PdfWriter

SRC_DIR = r"D:\BAMER-RL\reference"
OUT_DIR = os.path.join(SRC_DIR, "unlocked")
os.makedirs(OUT_DIR, exist_ok=True)

for name in os.listdir(SRC_DIR):
    if not name.lower().endswith(".pdf"):
        continue
    src = os.path.join(SRC_DIR, name)
    out = os.path.join(OUT_DIR, name)
    reader = PdfReader(src)
    if reader.is_encrypted:
        result = reader.decrypt("")  # empty user password
        print(name, "encrypted, decrypt result:", result)
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    with open(out, "wb") as f:
        writer.write(f)
    print(name, "->", out, "pages:", len(reader.pages))
