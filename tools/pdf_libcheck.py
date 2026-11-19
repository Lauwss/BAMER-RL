import importlib

for m in ["pypdf", "PyPDF2", "fitz", "pdfplumber", "pikepdf"]:
    try:
        importlib.import_module(m)
        print(m, "OK")
    except Exception:
        print(m, "NO")
