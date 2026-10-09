"""Exercise the generated offline dashboard in Chromium, without external requests."""

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/end_to_end")
    parser.add_argument("--chromium", default="/usr/bin/chromium")
    args = parser.parse_args()
    folder = (Path(args.results) / "dashboard").resolve()
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(folder)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    errors, external = [], []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=args.chromium, args=["--no-sandbox"])
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.on("pageerror", lambda error: errors.append(str(error)))

            def route(request):
                if urlparse(request.request.url).hostname != "127.0.0.1":
                    external.append(request.request.url)
                    request.abort()
                else:
                    request.continue_()

            page.route("**/*", route)
            page.goto(f"http://127.0.0.1:{server.server_port}/index.html", wait_until="networkidle")
            page.wait_for_function("document.getElementById('forecast').data?.length >= 3")
            assert page.locator("#dataset").input_value() == "oscillator"
            assert (
                page.evaluate("document.getElementById('heatmap').layout.xaxis.type") == "category"
            )
            page.locator("#origin").fill("1")
            page.locator("#origin").dispatch_event("input")
            assert "Origin" in page.locator("#origin-label").inner_text()
            page.locator("#runtime").click()
            assert (
                page.evaluate("document.getElementById('convergence').layout.xaxis.title.text")
                == "Training seconds"
            )
            page.screenshot(path=str(folder / "preview.png"), full_page=True)
            page.select_option("#dataset", "weather")
            assert (
                page.evaluate("document.getElementById('forecast').layout.xaxis.title.text")
                == "Calendar time"
            )
            sindy = page.evaluate(
                "study.runs.find(r=>r.dataset==='oscillator' && r.model==='sindy').id"
            )
            page.select_option("#dataset", "oscillator")
            page.select_option("#run", sindy)
            assert page.locator("#equations table").count() == 1
            page.set_viewport_size({"width": 390, "height": 844})
            page.wait_for_timeout(500)
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")
            page.screenshot(path=str(folder / "preview-mobile.png"), full_page=True)
            assert not errors, errors
            assert not external, external
            print(
                "Dashboard verified: forecast slider, time curves, date axes, sparse equations, mobile layout; no external requests or JavaScript errors."
            )
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    main()
