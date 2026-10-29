"""OpenArm WebRTC 實時串流與延遲監控模組 (WebRTC Streamer & Latency Monitor)

功能：
1. 啟動背景 aiohttp Web 伺服器與 WebRTC 伺服器 (aiortc)。
2. 即時打包機器人攝影機影像 (多相機拼貼合成) 並以超低延遲 (<50ms) 串流至其他電腦瀏覽器。
3. 透過 WebRTC DataChannel 即時傳輸雙臂姿態 (J1~J7)、夾爪、目標角度、力矩與錄製狀態。
4. 測量並計算：
   - 網路往返延遲 (Network RTT) 與單向延遲 (Network One-Way)
   - 主機採集與控制管線處理延遲 (Pipeline Latency)
   - 端到端總延遲 (End-to-End Latency)
   - 影像畫面幀率 (Video FPS) 與主迴圈控制頻率 (Loop Hz)
5. 內建現代化暗黑模式繁體中文 Web UI，無需安裝任何外部依賴或聯網。
"""

from __future__ import annotations

import asyncio
from fractions import Fraction
import json
import logging
import os
import socket
import threading
import time
from typing import Any

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
import av
import cv2
import numpy as np

logger = logging.getLogger("openarm_gc.webrtc")


def get_local_ip() -> str:
    """取得當前主機在區域網路的主要 IP 位址。"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# 內嵌前端 Web UI (HTML + CSS + JS) - 100% 自給自足，無須外部網路或 CDN
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OpenArm WebRTC 實時監控控制台</title>
<style>
  :root {
    --bg-dark: #0f172a;
    --card-bg: #1e293b;
    --card-border: #334155;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --accent-blue: #38bdf8;
    --accent-green: #22c55e;
    --accent-yellow: #eab308;
    --accent-red: #ef4444;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body {
    height: 100vh;
    max-height: 100vh;
    overflow: hidden; /* 單頁無捲軸 */
    background-color: var(--bg-dark);
    color: var(--text-primary);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    padding: 6px 10px;
    display: flex;
    flex-direction: column;
  }
  header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 14px;
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 8px;
    gap: 12px;
    flex-shrink: 0;
  }
  .header-left { display: flex; align-items: center; gap: 8px; }
  .logo { font-size: 1.05rem; font-weight: 700; color: var(--accent-blue); white-space: nowrap; }
  .badge {
    padding: 2px 8px;
    border-radius: 9999px;
    font-size: 0.72rem;
    font-weight: 600;
    white-space: nowrap;
  }
  .badge-connected { background: rgba(34, 197, 94, 0.15); color: var(--accent-green); border: 1px solid var(--accent-green); }
  .badge-disconnected { background: rgba(239, 68, 68, 0.15); color: var(--accent-red); border: 1px solid var(--accent-red); }
  .badge-recording { background: rgba(239, 68, 68, 0.2); color: #fca5a5; border: 1px solid var(--accent-red); animation: pulse 1.5s infinite; }
  .badge-standby { background: rgba(148, 163, 184, 0.15); color: var(--text-secondary); border: 1px solid var(--card-border); }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }

  /* 頂部整合指標區 */
  .metrics-bar {
    display: flex;
    align-items: center;
    gap: 14px;
    flex: 1;
    justify-content: center;
  }
  .metric-item {
    display: flex;
    align-items: baseline;
    gap: 5px;
    padding: 3px 10px;
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid var(--card-border);
    border-radius: 6px;
  }
  .m-label { font-size: 0.7rem; color: var(--text-secondary); white-space: nowrap; }
  .m-val { font-size: 1.05rem; font-weight: 700; font-family: monospace; color: var(--text-primary); }
  .m-sub { font-size: 0.68rem; color: var(--text-secondary); }
  .val-good { color: var(--accent-green) !important; }
  .val-warn { color: var(--accent-yellow) !important; }
  .val-bad { color: var(--accent-red) !important; }

  .btn-reconnect {
    background: #2563eb;
    color: #fff;
    border: none;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 0.72rem;
    font-weight: 600;
    cursor: pointer;
    white-space: nowrap;
    transition: background 0.2s;
  }
  .btn-reconnect:hover { background: #1d4ed8; }

  /* 主內容雙欄佈局 (100% 填滿視窗，永不超出) */
  .main-layout {
    flex: 1;
    min-height: 0; /* 防止子元素撐破 flex 容器 */
    display: grid;
    grid-template-columns: 1fr 330px;
    gap: 8px;
    margin-top: 6px;
  }
  @media (max-width: 900px) {
    .main-layout { grid-template-columns: 1fr; }
  }

  .left-panel {
    display: flex;
    flex-direction: column;
    height: 100%;
    min-height: 0;
  }

  .video-container {
    flex: 1;
    min-height: 0;
    background: #000;
    border: 1px solid var(--card-border);
    border-radius: 8px;
    overflow: hidden;
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  video {
    width: 100%;
    height: 100%;
    object-fit: contain;
    background: #000;
  }
  .video-hud {
    position: absolute;
    top: 8px;
    left: 8px;
    background: rgba(15, 23, 42, 0.75);
    backdrop-filter: blur(4px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 6px;
    padding: 3px 8px;
    font-size: 0.72rem;
    font-family: monospace;
    display: flex;
    gap: 8px;
    pointer-events: none;
  }
  .hud-badge { color: var(--accent-blue); font-weight: 600; }

  .chart-bar {
    height: 46px;
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 6px;
    padding: 3px 8px;
    margin-top: 6px;
    flex-shrink: 0;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }
  .chart-header {
    display: flex;
    justify-content: space-between;
    font-size: 0.68rem;
    color: var(--text-secondary);
    margin-bottom: 2px;
  }
  canvas#latencyChart {
    width: 100%;
    height: 24px;
    display: block;
  }

  /* 右側手臂監控面板 */
  .telemetry-sidebar {
    height: 100%;
    min-height: 0;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .arm-card {
    flex: 1;
    min-height: 0;
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 8px;
    padding: 8px 12px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
  }
  .arm-title {
    font-size: 0.8rem;
    font-weight: 700;
    color: var(--accent-blue);
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid var(--card-border);
    padding-bottom: 3px;
    margin-bottom: 3px;
  }
  .joint-table {
    width: 100%;
    font-size: 0.72rem;
    font-family: monospace;
    border-collapse: collapse;
  }
  .joint-table th {
    text-align: left;
    color: var(--text-secondary);
    font-weight: 500;
    padding-bottom: 2px;
  }
  .joint-table td {
    padding: 1.5px 0;
  }
  .val-tgt { color: var(--text-secondary); font-size: 0.68rem; }
  .gripper-section { margin-top: 4px; }
  .gripper-bar-container {
    background: #0f172a;
    border-radius: 4px;
    height: 8px;
    overflow: hidden;
    position: relative;
    border: 1px solid var(--card-border);
    margin-top: 2px;
  }
  .gripper-bar {
    height: 100%;
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-green));
    width: 0%;
    transition: width 0.08s ease-out;
  }
  .gripper-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.68rem;
    color: var(--text-secondary);
    font-family: monospace;
  }
</style>
</head>
<body>

<header>
  <div class="header-left">
    <span class="logo">🤖 OpenArm WebRTC 實時監控</span>
    <span id="connBadge" class="badge badge-disconnected">● 連線中...</span>
    <span id="recBadge" class="badge badge-standby">● 待命 (STANDBY)</span>
  </div>
  
  <div class="metrics-bar">
    <div class="metric-item">
      <span class="m-label">網路 RTT</span>
      <span class="m-val" id="valRtt">-- ms</span>
      <span class="m-sub" id="valNetOneWay">(單向 -- ms)</span>
    </div>
    <div class="metric-item">
      <span class="m-label">管線處理</span>
      <span class="m-val" id="valPipeline">-- ms</span>
    </div>
    <div class="metric-item">
      <span class="m-label">端到端總延遲</span>
      <span class="m-val val-good" id="valTotal">-- ms</span>
    </div>
    <div class="metric-item">
      <span class="m-label">串流 FPS / 頻率</span>
      <span class="m-val" id="valFps">-- FPS</span>
      <span class="m-sub" id="valLoopHz">(-- Hz)</span>
    </div>
  </div>

  <button class="btn-reconnect" onclick="reconnect()">重新連線</button>
</header>

<div class="main-layout">
  <div class="left-panel">
    <div class="video-container">
      <video id="remoteVideo" autoplay playsinline muted></video>
      <div class="video-hud">
        <span class="hud-badge" id="hudStatus">WebRTC</span>
        <span id="hudResolution">-- x --</span>
        <span id="hudFps">-- FPS</span>
      </div>
    </div>

    <div class="chart-bar">
      <div class="chart-header">
        <span>實時總延遲歷史趨勢 (最近 60 秒)</span>
        <span id="chartStats">平均: -- ms | 最大: -- ms</span>
      </div>
      <canvas id="latencyChart"></canvas>
    </div>
  </div>

  <div class="telemetry-sidebar">
    <div class="arm-card">
      <div class="arm-title">
        <span>左臂 (Left Arm)</span>
        <span id="leftArmStatus" style="font-size:0.7rem; color:var(--text-secondary);">正常</span>
      </div>
      <table class="joint-table">
        <thead>
          <tr><th>關節</th><th>當前</th><th>目標</th><th>誤差</th></tr>
        </thead>
        <tbody id="leftJointBody">
          <tr><td colspan="4" style="color:var(--text-secondary)">等候數據中...</td></tr>
        </tbody>
      </table>
      <div class="gripper-section">
        <div class="gripper-label">
          <span>夾爪</span>
          <span id="leftGripperVal">--°</span>
        </div>
        <div class="gripper-bar-container">
          <div id="leftGripperBar" class="gripper-bar"></div>
        </div>
      </div>
    </div>

    <div class="arm-card">
      <div class="arm-title">
        <span>右臂 (Right Arm)</span>
        <span id="rightArmStatus" style="font-size:0.7rem; color:var(--text-secondary);">正常</span>
      </div>
      <table class="joint-table">
        <thead>
          <tr><th>關節</th><th>當前</th><th>目標</th><th>誤差</th></tr>
        </thead>
        <tbody id="rightJointBody">
          <tr><td colspan="4" style="color:var(--text-secondary)">等候數據中...</td></tr>
        </tbody>
      </table>
      <div class="gripper-section">
        <div class="gripper-label">
          <span>夾爪</span>
          <span id="rightGripperVal">--°</span>
        </div>
        <div class="gripper-bar-container">
          <div id="rightGripperBar" class="gripper-bar"></div>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
  let pc = null;
  let dc = null;
  let pingInterval = null;
  let currentRtt = 0;
  let currentPipeline = 0;
  let currentTotal = 0;
  let latencyHistory = [];
  const maxHistory = 60;

  let frameCount = 0;
  let lastFpsTime = performance.now();
  let measuredFps = 0;

  const video = document.getElementById('remoteVideo');
  const connBadge = document.getElementById('connBadge');
  const recBadge = document.getElementById('recBadge');
  const valRtt = document.getElementById('valRtt');
  const valNetOneWay = document.getElementById('valNetOneWay');
  const valPipeline = document.getElementById('valPipeline');
  const valTotal = document.getElementById('valTotal');
  const valFps = document.getElementById('valFps');
  const valLoopHz = document.getElementById('valLoopHz');
  const hudResolution = document.getElementById('hudResolution');
  const hudFps = document.getElementById('hudFps');

  async function connect() {
    connBadge.className = 'badge badge-disconnected';
    connBadge.innerText = '● 連線中...';

    if (pc) {
      try { pc.close(); } catch(e) {}
    }

    pc = new RTCPeerConnection({
      iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
    });

    pc.onconnectionstatechange = () => {
      if (pc.connectionState === 'connected') {
        connBadge.className = 'badge badge-connected';
        connBadge.innerText = '● 連線正常';
      } else if (pc.connectionState === 'disconnected' || pc.connectionState === 'failed') {
        connBadge.className = 'badge badge-disconnected';
        connBadge.innerText = '● 斷線';
      }
    };

    pc.ontrack = (event) => {
      if (event.track.kind === 'video') {
        video.srcObject = event.streams[0];
        video.play().catch(e => console.log('Autoplay:', e));
      }
    };

    dc = pc.createDataChannel('telemetry');
    dc.onopen = () => {
      startPingPong();
    };
    dc.onclose = () => {
      stopPingPong();
    };
    dc.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === 'pong') {
          handlePong(msg);
        } else if (msg.type === 'telemetry') {
          handleTelemetry(msg);
        }
      } catch (err) {
        console.error('DataChannel error:', err);
      }
    };

    pc.addTransceiver('video', { direction: 'recvonly' });

    try {
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);

      await new Promise(resolve => {
        if (pc.iceGatheringState === 'complete') {
          resolve();
        } else {
          const checkState = () => {
            if (pc.iceGatheringState === 'complete') {
              pc.removeEventListener('icegatheringstatechange', checkState);
              resolve();
            }
          };
          pc.addEventListener('icegatheringstatechange', checkState);
          setTimeout(resolve, 800);
        }
      });

      const resp = await fetch('/offer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sdp: pc.localDescription.sdp,
          type: pc.localDescription.type
        })
      });

      if (!resp.ok) throw new Error('Offer error: ' + resp.status);
      const answer = await resp.json();
      await pc.setRemoteDescription(new RTCSessionDescription(answer));
    } catch (err) {
      console.error('WebRTC error:', err);
      connBadge.className = 'badge badge-disconnected';
      connBadge.innerText = '● 連線失敗，重試中';
      setTimeout(connect, 3000);
    }
  }

  function startPingPong() {
    stopPingPong();
    pingInterval = setInterval(() => {
      if (dc && dc.readyState === 'open') {
        dc.send(JSON.stringify({ type: 'ping', t: performance.now() }));
      }
    }, 500);
  }

  function stopPingPong() {
    if (pingInterval) {
      clearInterval(pingInterval);
      pingInterval = null;
    }
  }

  function handlePong(msg) {
    const now = performance.now();
    currentRtt = now - msg.t;
    const netOneWay = currentRtt / 2.0;

    valRtt.innerText = currentRtt.toFixed(1) + ' ms';
    valNetOneWay.innerText = '(單向 ' + netOneWay.toFixed(1) + ' ms)';

    if (currentRtt < 30) {
      valRtt.className = 'm-val val-good';
    } else if (currentRtt < 80) {
      valRtt.className = 'm-val val-warn';
    } else {
      valRtt.className = 'm-val val-bad';
    }

    updateTotalLatency();
  }

  function handleTelemetry(msg) {
    currentPipeline = msg.pipeline_latency_ms || 0;
    valPipeline.innerText = currentPipeline.toFixed(1) + ' ms';
    valLoopHz.innerText = '(' + (msg.fps ? msg.fps.toFixed(1) : '--') + ' Hz)';

    if (msg.status === 'recording') {
      recBadge.className = 'badge badge-recording';
      recBadge.innerText = '● 錄製中 (RECORDING)';
    } else if (msg.status === 'reset') {
      recBadge.className = 'badge badge-standby';
      recBadge.innerText = '● 重置中 (RESET)';
    } else {
      recBadge.className = 'badge badge-standby';
      recBadge.innerText = '● 待命 (STANDBY)';
    }

    updateTotalLatency();

    if (msg.arms) {
      renderArmTable('left', msg.arms.left);
      renderArmTable('right', msg.arms.right);
    }
  }

  function updateTotalLatency() {
    const netOneWay = currentRtt / 2.0;
    currentTotal = netOneWay + currentPipeline;
    valTotal.innerText = currentTotal.toFixed(1) + ' ms';

    if (currentTotal < 45) {
      valTotal.className = 'm-val val-good';
    } else if (currentTotal < 90) {
      valTotal.className = 'm-val val-warn';
    } else {
      valTotal.className = 'm-val val-bad';
    }

    latencyHistory.push(currentTotal);
    if (latencyHistory.length > maxHistory) latencyHistory.shift();
    drawChart();
  }

  function renderArmTable(side, armData) {
    if (!armData) return;
    const body = document.getElementById(side + 'JointBody');
    const joints = armData.joints || {};
    const keys = Object.keys(joints).sort();

    if (keys.length > 0) {
      let html = '';
      for (const k of keys) {
        const j = joints[k];
        const pos = j.pos !== undefined ? j.pos.toFixed(1) + '°' : '--';
        const tgt = j.target !== undefined ? j.target.toFixed(1) + '°' : '--';
        const diff = (j.pos !== undefined && j.target !== undefined) ? (j.target - j.pos).toFixed(1) + '°' : '--';
        html += `<tr><td>${k}</td><td>${pos}</td><td class="val-tgt">${tgt}</td><td>${diff}</td></tr>`;
      }
      body.innerHTML = html;
    }

    if (armData.gripper !== undefined) {
      const gVal = armData.gripper;
      document.getElementById(side + 'GripperVal').innerText = gVal.toFixed(1) + '°';
      const pct = Math.min(Math.max((gVal / 90.0) * 100, 0), 100);
      document.getElementById(side + 'GripperBar').style.width = pct.toFixed(0) + '%';
    }
  }

  if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
    function onVideoFrame() {
      frameCount++;
      const now = performance.now();
      if (now - lastFpsTime >= 1000) {
        measuredFps = (frameCount * 1000) / (now - lastFpsTime);
        valFps.innerText = measuredFps.toFixed(1) + ' FPS';
        hudFps.innerText = measuredFps.toFixed(0) + ' FPS';
        frameCount = 0;
        lastFpsTime = now;
      }
      if (video.videoWidth > 0) {
        hudResolution.innerText = video.videoWidth + 'x' + video.videoHeight;
      }
      video.requestVideoFrameCallback(onVideoFrame);
    }
    video.requestVideoFrameCallback(onVideoFrame);
  }

  const canvas = document.getElementById('latencyChart');
  const ctx = canvas.getContext('2d');
  function drawChart() {
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    ctx.clearRect(0, 0, w, h);
    if (latencyHistory.length < 2) return;

    const maxVal = Math.max(...latencyHistory, 50);
    const minVal = 0;
    const range = maxVal - minVal || 1;

    const avg = (latencyHistory.reduce((a, b) => a + b, 0) / latencyHistory.length).toFixed(1);
    const max = Math.max(...latencyHistory).toFixed(1);
    document.getElementById('chartStats').innerText = `平均: ${avg} ms | 最大: ${max} ms`;

    ctx.beginPath();
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 1.5;
    for (let i = 0; i < latencyHistory.length; i++) {
      const x = (i / (maxHistory - 1)) * w;
      const y = h - ((latencyHistory[i] - minVal) / range) * (h - 6) - 3;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    ctx.lineTo((latencyHistory.length - 1) / (maxHistory - 1) * w, h);
    ctx.lineTo(0, h);
    ctx.fillStyle = 'rgba(56, 189, 248, 0.1)';
    ctx.fill();
  }

  function reconnect() {
    connect();
  }

  window.addEventListener('load', connect);
</script>
</body>
</html>
"""


class RobotVideoStreamTrack(VideoStreamTrack):
    """自訂 WebRTC 影像串流軌道，持續讀取並編碼機器人當前攝影機畫面。"""

    kind = "video"

    def __init__(self, streamer: WebRTCStreamer):
        super().__init__()
        self.streamer = streamer

    async def recv(self) -> av.VideoFrame:
        pts, time_base = await self.next_timestamp()
        img = self.streamer.get_current_image()
        frame = av.VideoFrame.from_ndarray(img, format="bgr24")
        frame.pts = pts
        frame.time_base = time_base
        return frame


class WebRTCStreamer:
    """管理 WebRTC 串流伺服器與資料分發。"""

    def __init__(self, port: int = 8080):
        self.port = port
        self.local_ip = get_local_ip()
        self._is_recording = False
        self._is_running = False

        self._pcs: set[RTCPeerConnection] = set()
        self._data_channels: set[Any] = set()

        self._frame_lock = threading.Lock()
        self._current_image: np.ndarray | None = None
        self._placeholder_frame = self._create_placeholder()

        # 延遲與頻率計算
        self._last_t_cap = 0.0
        self._pipeline_latency_ms = 0.0
        self._obs_count = 0
        self._fps_t0 = time.perf_counter()
        self._current_fps = 0.0

        # 手臂關節狀態暫存
        self._last_arm_states: dict[str, Any] = {"left": {}, "right": {}}

        # 非同步背景執行緒與迴圈
        self.loop: asyncio.AbstractEventLoop | None = None
        self.thread: threading.Thread | None = None
        self.runner: web.AppRunner | None = None
        self._ready_event = threading.Event()

    def _create_placeholder(self) -> np.ndarray:
        """建立無相機時的預設狀態儀表板影像 (640x480)。"""
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        canvas[:] = (20, 24, 33)
        cv2.rectangle(canvas, (15, 15), (625, 465), (40, 48, 65), 2)
        cv2.putText(canvas, "OpenArm WebRTC Streamer", (40, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 210, 170), 2)
        cv2.putText(canvas, "Status: Active | Waiting for Camera Feed...", (40, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 170, 185), 1)
        cv2.line(canvas, (40, 125), (600, 125), (40, 48, 65), 1)
        return canvas

    def start(self):
        """啟動非同步伺服器執行緒。"""
        if self._is_running:
            return
        self._is_running = True
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.thread.start()
        self._ready_event.wait(timeout=5.0)

        banner = (
            f"\n{'='*75}\n"
            f"  [WebRTC] 實時監控串流伺服器已啟動！\n"
            f"  本機瀏覽器觀看:   http://localhost:{self.port}\n"
            f"  區域網路其他電腦: http://{self.local_ip}:{self.port}\n"
            f"  (支援即時影像拼貼、雙臂姿態儀表、網路 RTT 與管線延遲測量)\n"
            f"{'='*75}\n"
        )
        print(banner)
        logger.info("[WebRTC] 伺服器啟動於 port %d, LAN IP: %s", self.port, self.local_ip)

    def stop(self):
        """停止伺服器並釋放資源。"""
        if not self._is_running:
            return
        self._is_running = False
        if self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(self._cleanup_server(), self.loop)
        logger.info("[WebRTC] 伺服器已安全關閉")

    def _run_server(self):
        """背景執行緒入口：建立 asyncio 事件迴圈並執行 aiohttp。"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        app = web.Application()
        app.router.add_get("/", self._http_index)
        app.router.add_post("/offer", self._http_offer)
        app.router.add_get("/api/status", self._http_status)

        self.runner = web.AppRunner(app)
        self.loop.run_until_complete(self.runner.setup())
        site = web.TCPSite(self.runner, "0.0.0.0", self.port)
        self.loop.run_until_complete(site.start())

        self._ready_event.set()
        self.loop.run_forever()

    async def _cleanup_server(self):
        """在 asyncio 迴圈內清理 PeerConnections 與 runner。"""
        for pc in list(self._pcs):
            await pc.close()
        self._pcs.clear()
        self._data_channels.clear()
        if self.runner:
            await self.runner.cleanup()
        if self.loop:
            self.loop.stop()

    async def _http_index(self, _request: web.Request) -> web.Response:
        return web.Response(text=DASHBOARD_HTML, content_type="text/html")

    async def _http_status(self, _request: web.Request) -> web.Response:
        return web.json_response({
            "status": "recording" if self._is_recording else "standby",
            "clients": len(self._pcs),
            "fps": round(self._current_fps, 1),
            "pipeline_latency_ms": round(self._pipeline_latency_ms, 2),
            "local_ip": self.local_ip,
            "port": self.port,
        })

    async def _http_offer(self, request: web.Request) -> web.Response:
        """處理前端發送的 WebRTC SDP offer 並回傳 answer。"""
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        pc = RTCPeerConnection()
        self._pcs.add(pc)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            if pc.connectionState in ("failed", "closed"):
                await pc.close()
                self._pcs.discard(pc)

        @pc.on("datachannel")
        def on_datachannel(channel):
            self._data_channels.add(channel)

            @channel.on("close")
            def on_close():
                self._data_channels.discard(channel)

            @channel.on("message")
            def on_message(message):
                try:
                    data = json.loads(message)
                    if data.get("type") == "ping":
                        pong_msg = json.dumps({
                            "type": "pong",
                            "t": data.get("t"),
                            "server_time": time.time(),
                        })
                        channel.send(pong_msg)
                except Exception:
                    pass

        # 加入即時影像串流軌道
        video_track = RobotVideoStreamTrack(self)
        pc.addTrack(video_track)

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return web.json_response({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
        })

    def set_recording(self, is_recording: bool):
        self._is_recording = is_recording

    def get_current_image(self) -> np.ndarray:
        """獲取當前最新的攝影機畫面，供 WebRTC 軌道編碼傳輸。"""
        with self._frame_lock:
            if self._current_image is not None:
                return self._current_image
            return self._placeholder_frame

    def handle_observation(self, obs: dict[str, Any], t_cap: float):
        """處理 robot.get_observation() 的傳回值：提取影像與當前關節角度。"""
        self._last_t_cap = t_cap

        # 計算主迴圈頻率
        self._obs_count += 1
        now = time.perf_counter()
        if now - self._fps_t0 >= 1.0:
            self._current_fps = self._obs_count / (now - self._fps_t0)
            self._obs_count = 0
            self._fps_t0 = now

        # 提取相機畫面並合成為畫面
        images = self._extract_camera_images(obs)
        if images:
            mosaic = self._create_mosaic(images)
            with self._frame_lock:
                self._current_image = mosaic

        # 提取關節當前角度
        self._last_arm_states = self._parse_arm_states(obs)

    def handle_action(self, action: dict[str, Any] | None, t_send: float):
        """處理 teleop 或 policy 的 action：計算管線延遲並廣播至前端。"""
        if self._last_t_cap > 0:
            self._pipeline_latency_ms = (t_send - self._last_t_cap) * 1000.0

        if action:
            self._merge_action_targets(self._last_arm_states, action)

        # 組織 telemetry 封包並非同步廣播
        telemetry = {
            "type": "telemetry",
            "status": "recording" if self._is_recording else "standby",
            "pipeline_latency_ms": round(self._pipeline_latency_ms, 2),
            "fps": round(self._current_fps, 1),
            "server_time": time.time(),
            "arms": self._last_arm_states,
        }
        self.broadcast_telemetry(telemetry)

    def broadcast_telemetry(self, data: dict):
        """線程安全地透過 DataChannel 廣播 JSON 資料。"""
        if not self.loop or not self._is_running:
            return
        msg = json.dumps(data)
        self.loop.call_soon_threadsafe(self._do_broadcast, msg)

    def _do_broadcast(self, msg: str):
        for dc in list(self._data_channels):
            if dc.readyState == "open":
                try:
                    dc.send(msg)
                except Exception:
                    pass

    def attach(self, robot: Any, teleop: Any = None):
        """非侵入式攔截 robot.get_observation 與 robot.send_action。"""
        if robot is None or getattr(robot, "_webrtc_hooked", False):
            return
        robot._webrtc_hooked = True

        orig_get_obs = robot.get_observation
        orig_send_action = robot.send_action

        def wrapped_get_obs(*args, **kwargs):
            t_cap = time.perf_counter()
            obs = orig_get_obs(*args, **kwargs)
            try:
                self.handle_observation(obs, t_cap)
            except Exception as e:
                logger.debug("[WebRTC] 處理 observation 錯誤: %s", e)
            return obs

        def wrapped_send_action(action, *args, **kwargs):
            t_send = time.perf_counter()
            try:
                self.handle_action(action, t_send)
            except Exception as e:
                logger.debug("[WebRTC] 處理 action 錯誤: %s", e)
            return orig_send_action(action, *args, **kwargs)

        robot.get_observation = wrapped_get_obs
        robot.send_action = wrapped_send_action
        logger.info("[WebRTC] 已掛載至機器人控制迴圈，開始擷取實時資料")

    def _extract_camera_images(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        """從 observation 字典中提取所有攝影機畫面 (自動相容 PyTorch Tensor 與 NumPy 陣列)。"""
        images = {}
        for k, v in obs.items():
            if k.endswith("_depth"):
                continue

            arr = None
            if isinstance(v, np.ndarray):
                arr = v
            elif hasattr(v, "cpu") and hasattr(v, "numpy"):
                try:
                    arr = v.detach().cpu().numpy()
                except Exception:
                    continue

            if arr is not None:
                # 轉換為 (H, W, C)
                if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[2] not in (1, 3, 4):
                    arr = np.transpose(arr, (1, 2, 0))
                if arr.ndim == 3 and arr.shape[2] in (1, 3, 4):
                    if arr.dtype != np.uint8 and arr.max() <= 1.0:
                        arr = (arr * 255).astype(np.uint8)
                    images[k] = arr.astype(np.uint8)
                elif arr.ndim == 2:
                    images[k] = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        return images

    def _create_mosaic(self, images: dict[str, np.ndarray], target_h: int | None = None) -> np.ndarray:
        """將多台相機影像合成為單一拼貼畫布（不限制輸出解析度，保留相機原生畫質）。

        排版邏輯：
        - 1 台相機：原生全畫面顯示
        - 2 台相機：左腕與右腕水平並排
        - 3 台相機：
                     ego (置中)
            wright_left wright_right
        - 4 台以上：多相機網格排列
        """
        if not images:
            return self._placeholder_frame

        # 尋找關鍵相機：左腕、右腕、主視角 (ego)
        left_item = None
        right_item = None
        ego_item = None
        others = []

        for name, img in images.items():
            low = name.lower()
            if "left" in low or low.endswith("_l"):
                left_item = (name, img)
            elif "right" in low or low.endswith("_r"):
                right_item = (name, img)
            elif any(k in low for k in ("ego", "center", "main", "front", "top", "overhead")):
                ego_item = (name, img)
            else:
                others.append((name, img))

        def annotate(name: str, img: np.ndarray) -> np.ndarray:
            out = img.copy()
            cv2.putText(out, f"[{name}]", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
            cv2.putText(out, f"[{name}]", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 255), 2)
            return out

        # 情況 1: 3 台相機 -> 上排 ego 置中，下排 wright_left 與 wright_right 並排
        if len(images) == 3 and left_item and right_item and (ego_item or others):
            if not ego_item:
                ego_item = others[0]

            ego_img = annotate(ego_item[0], ego_item[1])
            wl_img = annotate(left_item[0], left_item[1])
            wr_img = annotate(right_item[0], right_item[1])

            # 下排：左右手腕相機並排（以較高者為基準對齊高度）
            h_bot = max(wl_img.shape[0], wr_img.shape[0])
            if wl_img.shape[0] != h_bot:
                wl_img = cv2.resize(wl_img, (int(wl_img.shape[1] * (h_bot / wl_img.shape[0])), h_bot), interpolation=cv2.INTER_LINEAR)
            if wr_img.shape[0] != h_bot:
                wr_img = cv2.resize(wr_img, (int(wr_img.shape[1] * (h_bot / wr_img.shape[0])), h_bot), interpolation=cv2.INTER_LINEAR)
            bot_row = np.hstack([wl_img, wr_img])

            # 總寬度由下排與 ego 中較寬者決定
            w_total = max(ego_img.shape[1], bot_row.shape[1])
            h_ego = ego_img.shape[0]

            # 上排：ego 置中於頂部
            top_row = np.zeros((h_ego, w_total, 3), dtype=np.uint8)
            top_row[:] = (15, 23, 42)
            x_ego = (w_total - ego_img.shape[1]) // 2
            top_row[:, x_ego : x_ego + ego_img.shape[1]] = ego_img

            # 下排：若寬度小於 w_total，則下排也置中留白
            if bot_row.shape[1] < w_total:
                bot_canvas = np.zeros((h_bot, w_total, 3), dtype=np.uint8)
                bot_canvas[:] = (15, 23, 42)
                x_bot = (w_total - bot_row.shape[1]) // 2
                bot_canvas[:, x_bot : x_bot + bot_row.shape[1]] = bot_row
                bot_row = bot_canvas

            mosaic = np.vstack([top_row, bot_row])
            return mosaic

        # 情況 2: 2 台相機 -> 水平並排
        if len(images) == 2 and left_item and right_item:
            wl = annotate(left_item[0], left_item[1])
            wr = annotate(right_item[0], right_item[1])
            h = max(wl.shape[0], wr.shape[0])
            if wl.shape[0] != h:
                wl = cv2.resize(wl, (int(wl.shape[1] * (h / wl.shape[0])), h), interpolation=cv2.INTER_LINEAR)
            if wr.shape[0] != h:
                wr = cv2.resize(wr, (int(wr.shape[1] * (h / wr.shape[0])), h), interpolation=cv2.INTER_LINEAR)
            return np.hstack([wl, wr])

        # 情況 3: 其他相機組合（單相機或 4 台以上）
        items = [(name, annotate(name, img)) for name, img in images.items()]
        if len(items) == 1:
            return items[0][1]
        elif len(items) == 2:
            h = max(items[0][1].shape[0], items[1][1].shape[0])
            i0 = cv2.resize(items[0][1], (int(items[0][1].shape[1] * (h / items[0][1].shape[0])), h))
            i1 = cv2.resize(items[1][1], (int(items[1][1].shape[1] * (h / items[1][1].shape[0])), h))
            return np.hstack([i0, i1])
        else:
            mid = (len(items) + 1) // 2
            h1 = max(it[1].shape[0] for it in items[:mid])
            r1_imgs = [cv2.resize(it[1], (int(it[1].shape[1] * (h1 / it[1].shape[0])), h1)) for it in items[:mid]]
            r1 = np.hstack(r1_imgs)

            h2 = max(it[1].shape[0] for it in items[mid:])
            r2_imgs = [cv2.resize(it[1], (int(it[1].shape[1] * (h2 / it[1].shape[0])), h2)) for it in items[mid:]]
            r2 = np.hstack(r2_imgs)

            w = max(r1.shape[1], r2.shape[1])
            if r1.shape[1] < w:
                pad = np.zeros((r1.shape[0], w - r1.shape[1], 3), dtype=np.uint8)
                pad[:] = (15, 23, 42)
                r1 = np.hstack([r1, pad])
            if r2.shape[1] < w:
                pad = np.zeros((r2.shape[0], w - r2.shape[1], 3), dtype=np.uint8)
                pad[:] = (15, 23, 42)
                r2 = np.hstack([r2, pad])
            return np.vstack([r1, r2])

    def _parse_arm_states(self, obs: dict[str, Any]) -> dict[str, Any]:
        """從 obs 解析關節與夾爪角度。"""
        arms = {"left": {"joints": {}, "gripper": 0.0}, "right": {"joints": {}, "gripper": 0.0}}
        for k, v in obs.items():
            if not isinstance(v, (int, float, np.number)):
                continue
            val = float(v)
            if k.startswith("left_"):
                sub = k[5:]
                if sub.startswith("joint_") and sub.endswith(".pos"):
                    jname = sub.split(".")[0]
                    arms["left"]["joints"].setdefault(jname, {})["pos"] = round(val, 1)
                elif "gripper" in sub and sub.endswith(".pos"):
                    arms["left"]["gripper"] = round(val, 1)
            elif k.startswith("right_"):
                sub = k[6:]
                if sub.startswith("joint_") and sub.endswith(".pos"):
                    jname = sub.split(".")[0]
                    arms["right"]["joints"].setdefault(jname, {})["pos"] = round(val, 1)
                elif "gripper" in sub and sub.endswith(".pos"):
                    arms["right"]["gripper"] = round(val, 1)
            elif k.startswith("joint_") and k.endswith(".pos"):
                jname = k.split(".")[0]
                arms["left"]["joints"].setdefault(jname, {})["pos"] = round(val, 1)
            elif "gripper" in k and k.endswith(".pos"):
                arms["left"]["gripper"] = round(val, 1)
        return arms

    def _merge_action_targets(self, arms: dict[str, Any], action: dict[str, Any]):
        """將 action 目標角度合併至 arms 狀態中。"""
        for k, v in action.items():
            if not isinstance(v, (int, float, np.number)):
                continue
            val = float(v)
            if k.startswith("left_"):
                sub = k[5:]
                if sub.startswith("joint_"):
                    jname = sub.split(".")[0]
                    arms["left"]["joints"].setdefault(jname, {})["target"] = round(val, 1)
                elif "gripper" in sub:
                    arms["left"]["gripper_target"] = round(val, 1)
            elif k.startswith("right_"):
                sub = k[6:]
                if sub.startswith("joint_"):
                    jname = sub.split(".")[0]
                    arms["right"]["joints"].setdefault(jname, {})["target"] = round(val, 1)
                elif "gripper" in sub:
                    arms["right"]["gripper_target"] = round(val, 1)
            elif k.startswith("joint_"):
                jname = k.split(".")[0]
                arms["left"]["joints"].setdefault(jname, {})["target"] = round(val, 1)
            elif "gripper" in k:
                arms["left"]["gripper_target"] = round(val, 1)


# 單例管理
_STREAMER_INSTANCE: WebRTCStreamer | None = None
_STREAMER_LOCK = threading.Lock()


def get_webrtc_streamer(port: int = 8080) -> WebRTCStreamer:
    """取得或建立全域唯一的 WebRTCStreamer 實例。"""
    global _STREAMER_INSTANCE
    with _STREAMER_LOCK:
        if _STREAMER_INSTANCE is None:
            _STREAMER_INSTANCE = WebRTCStreamer(port=port)
        return _STREAMER_INSTANCE


def stop_webrtc_streamer():
    """安全停止全域 WebRTCStreamer。"""
    global _STREAMER_INSTANCE
    with _STREAMER_LOCK:
        if _STREAMER_INSTANCE is not None:
            _STREAMER_INSTANCE.stop()
            _STREAMER_INSTANCE = None

