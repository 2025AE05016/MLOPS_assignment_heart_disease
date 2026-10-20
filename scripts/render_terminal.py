"""Render a captured command-output text file as a terminal-style PNG (headless Chrome)."""
import html
import subprocess
import sys
from datetime import datetime
from pathlib import Path

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def render(source, target, title, width=1500):
    text = Path(source).read_text()
    lines = text.count("\n") + 1
    page = Path(target).with_suffix(".html")
    page.write_text(
        "<html><body style='margin:0;background:#1e1e1e'>"
        "<div style='background:#3c3c3c;color:#ddd;font:13px -apple-system;padding:8px 14px'>"
        f"● ● ●&nbsp;&nbsp; {html.escape(title)} — captured "
        f"{datetime.now():%Y-%m-%d %H:%M}</div>"
        "<pre style='color:#d4d4d4;font:13px Menlo,monospace;padding:12px 16px;margin:0;"
        f"white-space:pre-wrap'>{html.escape(text)}</pre></body></html>")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                    f"--window-size={width},{lines * 17 + 70}",
                    f"--screenshot={Path(target).resolve()}", page.resolve().as_uri()],
                   check=True, capture_output=True)
    page.unlink()


if __name__ == "__main__":
    render(*sys.argv[1:4])
