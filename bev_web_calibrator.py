#!/usr/bin/env python3
"""
Headless, browser-based BEV calibration + ROI tool.

Runs a small HTTP server (Python stdlib only — no Flask/FastAPI) so it works
inside a Docker container with no display: start it, forward the port, open
the page in your browser, and calibrate interactively.

    python bev_web_calibrator.py --video AD_intersection01.mp4.mp4 --port 8000
    # then, from your host:  ssh -L 8000:localhost:8000 <docker-host>
    # or docker run -p 8000:8000 ...  and open http://localhost:8000

Workflow in the browser:
  1. Load a frame from the video.
  2. Click >=4 correspondences: a point on the camera frame (src), then its
     matching point on the BEV canvas (dst). Compute the homography (cv2,
     server-side) and verify the warped preview + reprojection error.
  3. Optionally set metric scale: click two points of known real distance on
     the preview and enter the distance -> pixels_per_meter.
  4. Define an ROI polygon on the frame, OR tick "Full frame" to analyse the
     whole image.
  5. Save -> writes bev_config.json (merging, so calibration_confidence,
     conflict_zones, etc. are preserved).

The pipeline (utils/site.py + main.py) reads roi_enabled / roi_polygon_img and
processes detections only inside the ROI when enabled.
"""

import os
import cv2
import json
import base64
import argparse
import logging
import numpy as np
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("bev_web_calibrator")

# Set by main(); read by the request handler.
VIDEO_SOURCE = ""
CONFIG_PATH = "bev_config.json"


# ── frame + homography helpers (server-side, cv2) ─────────────────────────

def read_frame(video_source: str, index: int):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {video_source}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = max(0, int(index))
    if total > 0 and idx >= total:
        idx = total - 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Failed to read frame {idx}")
    return frame, idx


def compute_homography(src, dst):
    src = np.asarray(src, dtype=np.float32)
    dst = np.asarray(dst, dtype=np.float32)
    n = len(src)
    if n < 4 or n != len(dst):
        raise ValueError("need >= 4 matched point pairs")
    if n == 4:
        H = cv2.getPerspectiveTransform(src, dst)
    else:
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None:
            raise ValueError("findHomography failed (degenerate points?)")
    src_h = np.hstack([src, np.ones((n, 1), dtype=np.float32)])
    proj = (H @ src_h.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    err = np.linalg.norm(proj - dst, axis=1)
    return H, float(err.mean()), float(err.max())


def jpeg_b64(img) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def load_config() -> dict:
    if os.path.isfile(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Could not read %s: %s", CONFIG_PATH, e)
    return {}


# ── HTTP handler ──────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        logger.info("%s - %s", self.address_string(), fmt % args)

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n) if n else b"{}"
        return json.loads(raw or b"{}")

    # -- GET --
    def do_GET(self):
        u = urlparse(self.path)
        try:
            if u.path in ("/", "/index.html"):
                return self._send(200, HTML_PAGE, "text/html; charset=utf-8")

            if u.path == "/api/meta":
                cfg = load_config()
                frame, idx = read_frame(VIDEO_SOURCE, cfg.get("frame_index", 0))
                h, w = frame.shape[:2]
                return self._send(200, {
                    "video_source": VIDEO_SOURCE,
                    "config_path": os.path.abspath(CONFIG_PATH),
                    "frame_index": idx,
                    "width": w, "height": h,
                    "src_points": cfg.get("src_points", []),
                    "dst_points": cfg.get("dst_points", []),
                    "pixels_per_meter": cfg.get("pixels_per_meter"),
                    "roi_enabled": bool(cfg.get("roi_enabled", False)),
                    "roi_polygon_img": cfg.get("roi_polygon_img", []),
                })

            if u.path == "/api/frame":
                q = parse_qs(u.query)
                idx = int(q.get("index", ["0"])[0])
                frame, idx = read_frame(VIDEO_SOURCE, idx)
                return self._send(200, {
                    "index": idx,
                    "width": frame.shape[1], "height": frame.shape[0],
                    "jpeg_b64": jpeg_b64(frame),
                })

            return self._send(404, {"error": "not found"})
        except Exception as e:
            logger.exception("GET %s failed", u.path)
            return self._send(500, {"error": str(e)})

    # -- POST --
    def do_POST(self):
        u = urlparse(self.path)
        try:
            body = self._read_json()

            if u.path == "/api/compute":
                H, mean_e, max_e = compute_homography(body["src_points"], body["dst_points"])
                frame, _ = read_frame(VIDEO_SOURCE, int(body.get("frame_index", 0)))
                h, w = frame.shape[:2]
                warp = cv2.warpPerspective(frame, H, (w, h))
                return self._send(200, {
                    "ok": True,
                    "homography": H.astype(float).tolist(),
                    "reproj_mean_px": round(mean_e, 2),
                    "reproj_max_px": round(max_e, 2),
                    "warp_jpeg_b64": jpeg_b64(warp),
                    "width": w, "height": h,
                })

            if u.path == "/api/save":
                cfg = load_config()
                frame, idx = read_frame(VIDEO_SOURCE, int(body.get("frame_index", 0)))
                h, w = frame.shape[:2]
                cfg.update({
                    "video_source": VIDEO_SOURCE,
                    "frame_index": idx,
                    "frame_width": w, "frame_height": h,
                    "bev_width_px": w, "bev_height_px": h,
                    "src_points": [[int(x), int(y)] for x, y in body.get("src_points", [])],
                    "dst_points": [[int(x), int(y)] for x, y in body.get("dst_points", [])],
                    "roi_enabled": bool(body.get("roi_enabled", False)),
                    "roi_polygon_img": [[int(x), int(y)] for x, y in body.get("roi_polygon_img", [])],
                })
                if body.get("homography"):
                    cfg["homography_matrix"] = body["homography"]
                if body.get("pixels_per_meter"):
                    cfg["pixels_per_meter"] = round(float(body["pixels_per_meter"]), 4)
                cfg.setdefault("calibration_confidence", "bev_only")
                cfg.setdefault("conflict_zones", [])
                with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                    json.dump(cfg, f, indent=2)
                logger.info("Saved config -> %s", os.path.abspath(CONFIG_PATH))
                return self._send(200, {"ok": True, "path": os.path.abspath(CONFIG_PATH)})

            return self._send(404, {"error": "not found"})
        except Exception as e:
            logger.exception("POST %s failed", u.path)
            return self._send(500, {"error": str(e)})


HTML_PAGE = r"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>BEV Calibration &amp; ROI</title>
<style>
  :root{--bg:#12141c;--panel:#1b1e2b;--acc:#4c8bf5;--ok:#37c871;--warn:#ffb020;--txt:#e7e9f0;--mut:#8b90a3}
  *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--txt);font:14px/1.5 system-ui,Segoe UI,sans-serif}
  h1{font-size:18px;margin:0} h3{margin:14px 0 6px;font-size:14px;color:var(--mut)}
  header{padding:12px 18px;background:var(--panel);display:flex;gap:16px;align-items:center;flex-wrap:wrap;position:sticky;top:0;z-index:5}
  main{padding:18px;max-width:1400px;margin:0 auto}
  .panel{background:var(--panel);border-radius:10px;padding:14px 16px;margin-bottom:16px}
  .row{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:8px 0}
  button{background:var(--acc);color:#fff;border:0;border-radius:7px;padding:7px 13px;font-weight:600;cursor:pointer}
  button.sec{background:#2a2e40} button:disabled{opacity:.4;cursor:not-allowed}
  input[type=number]{background:#0f1119;border:1px solid #2a2e40;color:var(--txt);border-radius:6px;padding:6px 8px;width:90px}
  label{color:var(--mut)} .canvs{display:flex;gap:16px;flex-wrap:wrap}
  .canvbox{flex:1;min-width:360px} canvas{max-width:100%;border:1px solid #2a2e40;border-radius:8px;background:#000;cursor:crosshair;display:block}
  table{border-collapse:collapse;width:100%;margin-top:8px;font-size:13px} td,th{border-bottom:1px solid #2a2e40;padding:4px 8px;text-align:left}
  .pill{padding:2px 8px;border-radius:20px;font-size:12px;font-weight:600}
  .mut{color:var(--mut)} .ok{color:var(--ok)} .warn{color:var(--warn)}
  #status{margin-left:auto}
</style></head><body>
<header>
  <h1>🛰️ BEV Calibration &amp; ROI</h1>
  <label>Frame # <input id=frameIdx type=number value=0 min=0></label>
  <button id=loadBtn>Load frame</button>
  <span id=dims class=mut></span>
  <span id=status class=mut></span>
</header>
<main>
  <div class=panel>
    <div class=row>
      <b>Click mode:</b>
      <label><input type=radio name=mode value=corr checked> Correspondences (src&rarr;dst)</label>
      <label><input type=radio name=mode value=roi> ROI polygon</label>
      <button id=undoBtn class=sec>Undo last</button>
      <button id=resetBtn class=sec>Reset points</button>
    </div>
    <div class=canvs>
      <div class=canvbox><h3>Camera frame — click SRC (or ROI vertices)</h3><canvas id=frame></canvas></div>
      <div class=canvbox><h3>BEV canvas — click matching DST</h3><canvas id=bev></canvas></div>
    </div>
    <table id=pts><thead><tr><th>#</th><th>src (px)</th><th>dst (bev px)</th></tr></thead><tbody></tbody></table>
    <div class=row><button id=computeBtn>Compute homography</button><span id=reproj class=mut></span></div>
  </div>

  <div class=panel>
    <h3>BEV preview &amp; metric scale (optional)</h3>
    <div class=row><span class=mut>Click 2 points of known real distance on the preview, enter metres:</span>
      <input id=realdist type=number step=any placeholder="m">
      <button id=scaleBtn class=sec>Set scale</button>
      <span id=ppmLabel class=mut></span></div>
    <canvas id=warp></canvas>
  </div>

  <div class=panel>
    <h3>Region of Interest</h3>
    <div class=row><label><input type=checkbox id=fullframe> <b>Full frame</b> — process the entire image (no ROI)</label></div>
    <div id=roiInfo class=mut>ROI mode: pick &ge;3 vertices on the camera frame above.</div>
  </div>

  <div class=panel>
    <div class=row><button id=saveBtn>💾 Save config</button><span id=saveMsg class=mut></span></div>
  </div>
</main>
<script>
const $=s=>document.querySelector(s);
const st={W:0,H:0,idx:0,frameImg:null,warpImg:null,
  src:[],dst:[],roi:[],homography:null,ppm:null,scalePts:[],fullframe:false};
const fc=$('#frame'),bc=$('#bev'),wc=$('#warp');
const fx=fc.getContext('2d'),bx=bc.getContext('2d'),wx=wc.getContext('2d');
const PC=['#ff4d4d','#37c871','#4c8bf5','#ffb020','#c774f5','#27d3d3','#ff8a3d','#9acd32'];

function mode(){return document.querySelector('input[name=mode]:checked').value;}
function setStatus(m,c){const s=$('#status');s.textContent=m;s.className=c||'mut';}

function evtXY(cv,e){const r=cv.getBoundingClientRect();
  return [Math.round((e.clientX-r.left)*(cv.width/r.width)),
          Math.round((e.clientY-r.top)*(cv.height/r.height))];}

function drawFrame(){
  if(!st.frameImg)return; fx.drawImage(st.frameImg,0,0);
  // ROI polygon
  if(st.roi.length){fx.strokeStyle='#ffb020';fx.fillStyle='rgba(255,176,32,.15)';fx.lineWidth=2;
    fx.beginPath();st.roi.forEach((p,i)=>i?fx.lineTo(p[0],p[1]):fx.moveTo(p[0],p[1]));
    if(st.roi.length>2)fx.closePath();fx.stroke();if(st.roi.length>2)fx.fill();
    st.roi.forEach((p,i)=>{dot(fx,p,'#ffb020');tag(fx,p,'R'+(i+1),'#ffb020');});}
  // src points
  st.src.forEach((p,i)=>{const c=PC[i%PC.length];dot(fx,p,c);tag(fx,p,'S'+(i+1),c);});
}
function drawBev(){
  bx.fillStyle='#0b1020';bx.fillRect(0,0,bc.width,bc.height);
  bx.strokeStyle='#20263c';bx.lineWidth=1;
  for(let x=0;x<bc.width;x+=50){bx.beginPath();bx.moveTo(x,0);bx.lineTo(x,bc.height);bx.stroke();}
  for(let y=0;y<bc.height;y+=50){bx.beginPath();bx.moveTo(0,y);bx.lineTo(bc.width,y);bx.stroke();}
  st.dst.forEach((p,i)=>{const c=PC[i%PC.length];dot(bx,p,c);tag(bx,p,'D'+(i+1),c);});
}
function drawWarp(){
  if(!st.warpImg)return; wx.drawImage(st.warpImg,0,0);
  st.scalePts.forEach((p,i)=>{dot(wx,p,'#37c871');tag(wx,p,'P'+(i+1),'#37c871');});
  if(st.scalePts.length==2){wx.strokeStyle='#37c871';wx.lineWidth=2;wx.beginPath();
    wx.moveTo(...st.scalePts[0]);wx.lineTo(...st.scalePts[1]);wx.stroke();}
}
function dot(c,p,col){c.fillStyle=col;c.beginPath();c.arc(p[0],p[1],5,0,7);c.fill();}
function tag(c,p,t,col){c.fillStyle=col;c.font='bold 14px sans-serif';c.fillText(t,p[0]+7,p[1]-7);}

function renderTable(){
  const tb=$('#pts tbody');tb.innerHTML='';
  const n=Math.max(st.src.length,st.dst.length);
  for(let i=0;i<n;i++){const tr=document.createElement('tr');
    tr.innerHTML=`<td>${i+1}</td><td>${st.src[i]?st.src[i].join(', '):'—'}</td><td>${st.dst[i]?st.dst[i].join(', '):'—'}</td>`;
    tb.appendChild(tr);}
}

fc.addEventListener('click',e=>{const p=evtXY(fc,e);
  if(mode()==='roi'){if(st.fullframe)return;st.roi.push(p);updateRoiInfo();drawFrame();}
  else{if(st.src.length>st.dst.length){setStatus('Click the matching DST on the BEV canvas first.','warn');return;}
    st.src.push(p);drawFrame();renderTable();setStatus('SRC '+st.src.length+' set — now click DST.','mut');}});
bc.addEventListener('click',e=>{const p=evtXY(bc,e);
  if(mode()!=='corr')return;
  if(st.dst.length>=st.src.length){setStatus('Click a SRC point on the frame first.','warn');return;}
  st.dst.push(p);drawBev();renderTable();setStatus(st.dst.length+' pair(s) set.','ok');});
wc.addEventListener('click',e=>{if(!st.warpImg)return;const p=evtXY(wc,e);
  if(st.scalePts.length>=2)st.scalePts=[];st.scalePts.push(p);drawWarp();});

$('#undoBtn').onclick=()=>{if(mode()==='roi'){st.roi.pop();updateRoiInfo();}
  else{if(st.dst.length===st.src.length)st.dst.pop();else st.src.pop();renderTable();}
  drawFrame();drawBev();};
$('#resetBtn').onclick=()=>{st.src=[];st.dst=[];st.roi=[];st.homography=null;st.scalePts=[];st.ppm=null;
  $('#reproj').textContent='';$('#ppmLabel').textContent='';renderTable();drawFrame();drawBev();
  wx.clearRect(0,0,wc.width,wc.height);st.warpImg=null;updateRoiInfo();};

$('#fullframe').onchange=e=>{st.fullframe=e.target.checked;if(st.fullframe)st.roi=[];drawFrame();updateRoiInfo();};
function updateRoiInfo(){$('#roiInfo').innerHTML=st.fullframe
  ?'<span class=ok>Full frame</span> — ROI disabled, entire image processed.'
  :(st.roi.length?('ROI vertices: '+st.roi.length+(st.roi.length<3?' <span class=warn>(need &ge;3)</span>':' <span class=ok>ok</span>')):'ROI mode: pick &ge;3 vertices on the camera frame.');}

async function loadFrame(){
  const idx=+$('#frameIdx').value||0; setStatus('loading frame…');
  const r=await fetch('/api/frame?index='+idx); const j=await r.json();
  if(j.error){setStatus(j.error,'warn');return;}
  st.idx=j.index;st.W=j.width;st.H=j.height;
  fc.width=bc.width=wc.width=j.width; fc.height=bc.height=wc.height=j.height;
  const img=new Image();img.onload=()=>{st.frameImg=img;drawFrame();drawBev();};
  img.src='data:image/jpeg;base64,'+j.jpeg_b64;
  $('#dims').textContent=`frame ${j.index} · ${j.width}×${j.height}`;setStatus('frame loaded','ok');
}
$('#loadBtn').onclick=loadFrame;

$('#computeBtn').onclick=async()=>{
  if(st.src.length<4||st.src.length!==st.dst.length){setStatus('Need ≥4 complete src↔dst pairs.','warn');return;}
  setStatus('computing homography…');
  const r=await fetch('/api/compute',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({src_points:st.src,dst_points:st.dst,frame_index:st.idx})});
  const j=await r.json(); if(j.error){setStatus(j.error,'warn');return;}
  st.homography=j.homography;
  $('#reproj').innerHTML=`reproj err: mean ${j.reproj_mean_px}px, max ${j.reproj_max_px}px `+
    (j.reproj_max_px<8?'<span class=ok>✔</span>':'<span class=warn>(refine)</span>');
  const img=new Image();img.onload=()=>{st.warpImg=img;st.scalePts=[];drawWarp();};
  img.src='data:image/jpeg;base64,'+j.warp_jpeg_b64; setStatus('homography computed','ok');
};

$('#scaleBtn').onclick=()=>{
  if(st.scalePts.length!==2){setStatus('Click exactly 2 points on the preview.','warn');return;}
  const d=+$('#realdist').value; if(!(d>0)){setStatus('Enter a positive distance in metres.','warn');return;}
  const px=Math.hypot(st.scalePts[0][0]-st.scalePts[1][0],st.scalePts[0][1]-st.scalePts[1][1]);
  st.ppm=px/d; $('#ppmLabel').innerHTML=`pixels_per_meter = <b>${st.ppm.toFixed(3)}</b> (${px.toFixed(1)}px / ${d}m)`;
};

$('#saveBtn').onclick=async()=>{
  if(!st.homography){setStatus('Compute the homography before saving.','warn');return;}
  if(!st.fullframe&&st.roi.length>0&&st.roi.length<3){setStatus('ROI needs ≥3 vertices (or tick Full frame).','warn');return;}
  const payload={frame_index:st.idx,src_points:st.src,dst_points:st.dst,homography:st.homography,
    pixels_per_meter:st.ppm,roi_enabled:(!st.fullframe&&st.roi.length>=3),roi_polygon_img:st.roi};
  const r=await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  const j=await r.json(); if(j.error){setStatus(j.error,'warn');return;}
  $('#saveMsg').innerHTML='<span class=ok>Saved → '+j.path+'</span>';setStatus('saved','ok');
};

// bootstrap: load meta + first frame, restore existing points
(async()=>{
  try{const m=await(await fetch('/api/meta')).json();
    $('#frameIdx').value=m.frame_index||0;
    st.src=(m.src_points||[]).map(p=>[p[0],p[1]]);
    st.dst=(m.dst_points||[]).map(p=>[p[0],p[1]]);
    st.roi=(m.roi_polygon_img||[]).map(p=>[p[0],p[1]]);
    st.fullframe=!m.roi_enabled&&st.roi.length===0;
    $('#fullframe').checked=st.fullframe;
    if(m.pixels_per_meter){st.ppm=m.pixels_per_meter;$('#ppmLabel').innerHTML='pixels_per_meter = <b>'+m.pixels_per_meter+'</b> (loaded)';}
  }catch(e){}
  await loadFrame(); renderTable(); updateRoiInfo();
})();
</script></body></html>"""


def main():
    global VIDEO_SOURCE, CONFIG_PATH
    ap = argparse.ArgumentParser(description="Headless web BEV calibration + ROI tool.")
    ap.add_argument("--video", default=None, help="Video/stream source (default: config.py source_stream).")
    ap.add_argument("--config", default="bev_config.json", help="Output config JSON (default: bev_config.json).")
    ap.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0 for port-forwarding).")
    ap.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000).")
    args = ap.parse_args()

    video = args.video
    if not video:
        try:
            from config import AppConfig
            video = AppConfig().source_stream
        except Exception:
            video = ""
    if not video:
        raise SystemExit("No --video given and could not read config.py source_stream.")

    VIDEO_SOURCE = video
    CONFIG_PATH = args.config

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    logger.info("BEV calibrator serving on http://%s:%d  (video=%s, config=%s)",
                args.host, args.port, VIDEO_SOURCE, os.path.abspath(CONFIG_PATH))
    logger.info("Port-forward it, then open the page in your browser. Ctrl-C to stop.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down.")
        srv.shutdown()


if __name__ == "__main__":
    main()
