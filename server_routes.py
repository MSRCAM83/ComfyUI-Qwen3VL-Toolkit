"""
Server-side file browser API endpoints for QVL_ServerImage node.
Provides folder listing, thumbnails, and image serving.
"""

import os
import io
from aiohttp import web
from server import PromptServer
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
THUMB_SIZE = 192

@PromptServer.instance.routes.get("/qvl/browse")
async def qvl_browse(request):
    """List files and folders in a directory."""
    path = request.query.get("path", "/workspace")

    if not os.path.isdir(path):
        return web.json_response({"error": f"Not a directory: {path}"}, status=400)

    dirs = []
    files = []

    try:
        for entry in sorted(os.scandir(path), key=lambda e: e.name.lower()):
            if entry.name.startswith("."):
                continue
            if entry.is_dir(follow_symlinks=True):
                dirs.append({"name": entry.name, "type": "dir"})
            elif entry.is_file(follow_symlinks=True):
                ext = os.path.splitext(entry.name)[1].lower()
                if ext in IMG_EXTS:
                    try:
                        stat = entry.stat()
                        files.append({
                            "name": entry.name,
                            "type": "image",
                            "size": stat.st_size,
                        })
                    except OSError:
                        pass
    except PermissionError:
        return web.json_response({"error": f"Permission denied: {path}"}, status=403)

    parent = os.path.dirname(path) if path != "/" else None

    return web.json_response({
        "path": path,
        "parent": parent,
        "dirs": dirs,
        "files": files,
        "total_images": len(files),
    })


@PromptServer.instance.routes.get("/qvl/thumbnail")
async def qvl_thumbnail(request):
    """Serve a thumbnail of an image file."""
    filepath = request.query.get("path", "")

    if not filepath or not os.path.isfile(filepath):
        return web.Response(status=404, text="File not found")

    ext = os.path.splitext(filepath)[1].lower()
    if ext not in IMG_EXTS:
        return web.Response(status=400, text="Not an image")

    try:
        img = Image.open(filepath)
        img.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75)
        buf.seek(0)
        return web.Response(
            body=buf.read(),
            content_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=3600"},
        )
    except Exception as e:
        return web.Response(status=500, text=str(e))


@PromptServer.instance.routes.get("/qvl/image")
async def qvl_image(request):
    """Serve a full image file."""
    filepath = request.query.get("path", "")

    if not filepath or not os.path.isfile(filepath):
        return web.Response(status=404, text="File not found")

    ext = os.path.splitext(filepath)[1].lower()
    content_types = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".webp": "image/webp",
        ".bmp": "image/bmp", ".gif": "image/gif",
    }
    ct = content_types.get(ext, "application/octet-stream")

    return web.FileResponse(filepath, headers={"Content-Type": ct})


print("[QVL] Server browser routes registered: /qvl/browse, /qvl/thumbnail, /qvl/image")
