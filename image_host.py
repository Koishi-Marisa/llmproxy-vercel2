"""Tiny stand-in for a remote image host, so the proxy's URL-fetch path can be
tested without leaving the machine.

    /pixel.png        a real 1x1 PNG (content-type: image/png)
    /not-an-image     text/html, to prove non-image responses are rejected
"""
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, Response

app = FastAPI()

PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d4944415478da63f8cf0000030001fc02fed70000000049454e44ae426082"
)


@app.get("/pixel.png")
async def pixel():
    return Response(content=PNG, media_type="image/png")


@app.get("/not-an-image")
async def not_an_image():
    return HTMLResponse("<html><body>nope</body></html>")
