"""
GPT Chess — single-file full app
Features:
- password-based accounts (bcrypt via passlib)
- session tokens for auth
- lobby + create/join public & private rooms
- realtime play via WebSockets (FastAPI)
- clocks with increments and time-forfeit handling
- play vs Stockfish via Lichess Cloud Eval API (fallback to random legal move)
- Elo rating updates, JSON persistence (chess_data/)
- single-file: save as app.py and run

Requirements:
pip install fastapi uvicorn python-chess passlib[bcrypt] requests

Run:
python app.py
Open: http://localhost:8000
"""

import os
import json
import time
import threading
import traceback
import random
import secrets
from math import pow
from typing import Dict, Any

import chess
from passlib.hash import bcrypt
import requests
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
import uvicorn
import asyncio

# ---------------- Configuration & Storage ----------------
DATA_DIR = "chess_data"
USERS_FILE = os.path.join(DATA_DIR, "users.json")
GAMES_FILE = os.path.join(DATA_DIR, "games.json")
os.makedirs(DATA_DIR, exist_ok=True)

file_lock = threading.Lock()

def load_json(path, default):
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump(default, f)
        return default
    with open(path, "r") as f:
        try:
            return json.load(f)
        except:
            return default

def save_json_atomic(path, data):
    with file_lock:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)

users = load_json(USERS_FILE, {})  # username -> {id,password_hash,rating,games_played}
games_store = load_json(GAMES_FILE, {})  # id -> game metadata

_next_user_id = max([u.get("id", 0) for u in users.values()], default=0) + 1
_next_game_id = max([int(k) for k in games_store.keys()], default=0) + 1

# session tokens (in-memory). For demo simplicity; restart clears tokens.
session_tokens: Dict[str, str] = {}  # token -> username

# ---------------- User management ----------------
def register_user(username: str, password: str):
    global _next_user_id
    username = username.strip()
    if not username or not password:
        return False, "username and password required"
    if username in users:
        return False, "username exists"
    pw_hash = bcrypt.hash(password)
    users[username] = {"id": _next_user_id, "password_hash": pw_hash, "rating": 1200, "games_played": 0}
    _next_user_id += 1
    save_json_atomic(USERS_FILE, users)
    return True, users[username]

def verify_login(username: str, password: str):
    u = users.get(username)
    if not u:
        return False, "no such user"
    try:
        ok = bcrypt.verify(password, u["password_hash"])
    except Exception:
        ok = False
    if not ok:
        return False, "invalid password"
    token = secrets.token_urlsafe(24)
    session_tokens[token] = username
    return True, token

def get_user_by_token(token: str):
    return session_tokens.get(token)

def save_users():
    save_json_atomic(USERS_FILE, users)

# ---------------- Elo helpers ----------------
def expected_score(r1, r2):
    return 1.0 / (1 + pow(10, (r2 - r1) / 400.0))

def new_elo(r, opp, score, k=32):
    e = expected_score(r, opp)
    return int(round(r + k * (score - e)))

# ---------------- Game Room / Manager ----------------
class GameRoom:
    def __init__(self, room_id: str, time_control_seconds: int = 300, increment_seconds: int = 0, is_private=False):
        self.room_id = room_id
        self.game = chess.Board()
        self.players: Dict[str, WebSocket] = {"white": None, "black": None}
        self.players_usernames: Dict[str, Any] = {"white": None, "black": None}
        self.spectators = set()
        self.lock = threading.Lock()
        self.started = False
        self.time_control = time_control_seconds
        self.increment = increment_seconds
        self.remaining = {"white": float(self.time_control), "black": float(self.time_control)}
        self.last_move_ts = None
        self.turn = "white"
        self.is_private = is_private
        self.move_list = []
        self.result = None  # "1-0","0-1","1/2-1/2" or None

    def to_public(self):
        return {
            "room_id": self.room_id,
            "started": self.started,
            "players": {c: (self.players_usernames[c] or None) for c in ["white", "black"]},
            "is_private": self.is_private,
            "time_control": self.time_control,
            "increment": self.increment,
            "fen": self.game.fen(),
        }

    def add_player(self, ws: WebSocket, username: str):
        with self.lock:
            if not self.players["white"]:
                self.players["white"] = ws
                self.players_usernames["white"] = username
                return "white"
            elif not self.players["black"]:
                self.players["black"] = ws
                self.players_usernames["black"] = username
                return "black"
            else:
                self.spectators.add(ws)
                return "spectator"

    def remove_ws(self, ws: WebSocket):
        with self.lock:
            for c in ["white", "black"]:
                if self.players.get(c) == ws:
                    self.players[c] = None
                    self.players_usernames[c] = None
                    self.started = False
            if ws in self.spectators:
                self.spectators.remove(ws)

    def start_if_ready(self):
        with self.lock:
            if self.players.get("white") and self.players.get("black"):
                self.started = True
                self.last_move_ts = time.time()
                self.turn = "white"
                return True
            return False

    def apply_move(self, uci_move: str):
        with self.lock:
            now = time.time()
            if self.last_move_ts and self.turn:
                elapsed = now - self.last_move_ts
                self.remaining[self.turn] -= elapsed
                if self.remaining[self.turn] < 0:
                    winner = "black" if self.turn == "white" else "white"
                    self.result = "1-0" if winner == "white" else "0-1"
                    return False, f"time forfeit; winner {winner}"
            try:
                mv = chess.Move.from_uci(uci_move)
            except Exception:
                return False, "invalid move format"
            if mv not in self.game.legal_moves:
                return False, "illegal move"
            self.game.push(mv)
            mover = self.turn
            self.move_list.append(uci_move)
            self.remaining[mover] += self.increment
            self.turn = "black" if self.turn == "white" else "white"
            self.last_move_ts = time.time()
            if self.game.is_game_over():
                self.result = self.game.result() or "1/2-1/2"
            return True, self.game.fen()

    def to_result_payload(self):
        return {
            "room_id": self.room_id,
            "result": self.result,
            "moves": self.move_list,
            "fen": self.game.fen(),
            "players": self.players_usernames,
            "time_control": self.time_control,
            "increment": self.increment
        }

rooms: Dict[str, GameRoom] = {}
rooms_lock = threading.Lock()

def create_room(time_control=300, increment=0, is_private=False):
    room_id = str(int(time.time() * 1000))
    with rooms_lock:
        room = GameRoom(room_id, time_control_seconds=time_control, increment_seconds=increment, is_private=is_private)
        rooms[room_id] = room
    save_games_store()
    return room

def list_public_rooms():
    with rooms_lock:
        return [r.to_public() for r in rooms.values() if not r.is_private]

# ---------------- Lichess cloud-eval integration ----------------
LICHESS_CLOUD_URL = "https://lichess.org/api/cloud-eval"

def call_lichess_cloud_eval_sync(fen: str, multiPv: int = 1):
    try:
        params = {"fen": fen, "multiPv": multiPv}
        headers = {"User-Agent": "GPT-Chess-Demo/1.0 (+https://example)"}
        resp = requests.get(LICHESS_CLOUD_URL, params=params, headers=headers, timeout=5.0)
        if resp.status_code != 200:
            return None
        data = resp.json()
        if "pvs" in data and isinstance(data["pvs"], list) and len(data["pvs"]) > 0:
            first = data["pvs"][0]
            moves_str = first.get("moves", "")
            if moves_str:
                parts = moves_str.split()
                if len(parts) > 0:
                    return parts[0]
        if "bestMove" in data:
            return data["bestMove"]
        return None
    except Exception:
        return None

async def lichess_best_move(fen: str):
    mv = await asyncio.to_thread(call_lichess_cloud_eval_sync, fen, 1)
    return mv

# ---------------- fallback AI ----------------
def random_ai_move(fen):
    board = chess.Board(fen)
    moves = list(board.legal_moves)
    if not moves:
        return None
    return random.choice(moves).uci()

async def choose_ai_move(fen):
    try:
        candidate = await lichess_best_move(fen)
        if candidate:
            board = chess.Board(fen)
            # try UCI
            if len(candidate) in (4,5) and all(ch.isalnum() for ch in candidate):
                try:
                    mv = chess.Move.from_uci(candidate)
                    if mv in board.legal_moves:
                        return mv.uci()
                except:
                    pass
            # try SAN -> UCI
            try:
                mv = board.parse_san(candidate)
                if mv and mv in board.legal_moves:
                    return mv.uci()
            except Exception:
                pass
    except Exception:
        pass
    return random_ai_move(fen)

# ---------------- FastAPI app + UI (single-file front-end) ----------------
app = FastAPI()

INDEX_HTML = r"""
<!doctype html>
<html><head><meta charset="utf-8"><title>GPT Chess — single file</title>
<style>
body{font-family:system-ui,Segoe UI,Roboto,Arial;margin:12px}
#board{width:480px;height:480px;border:2px solid #333;display:grid;grid-template-columns:repeat(8,1fr)}
.square{display:flex;align-items:center;justify-content:center;font-size:36px;user-select:none}
.light{background:#f0d9b5}.dark{background:#b58863}
.room{padding:6px;border:1px solid #ddd;margin-bottom:8px}
</style>
</head><body>
<h2>GPT Chess — Single file (Password + Lichess AI)</h2>
<div>
  <label>Username: <input id="username"/></label>
  <label>Password: <input id="password" type="password"/></label>
  <button id="btnRegister">Register</button>
  <button id="btnLogin">Login</button>
  <span id="userInfo"></span>
</div>

<div id="main" style="display:none;margin-top:12px">
  <div style="display:flex;gap:18px">
    <div style="min-width:520px">
      <div style="display:flex;gap:8px;align-items:center;">
        <span id="userBadge"></span><span id="ratingBadge"></span>
        <label>Time(s): <input id="timeControl" value="300" style="width:80px"/></label>
        <label>Inc(s): <input id="incControl" value="2" style="width:60px"/></label>
        <label>Private: <input id="privateRoom" type="checkbox"/></label>
        <button id="btnCreate">Create Room</button>
        <button id="btnRefresh">Refresh Rooms</button>
      </div>
      <h3>Lobby</h3>
      <div id="lobby"></div>
    </div>

    <div>
      <h3>Room</h3>
      <div id="currentRoomInfo">No room</div>
      <div id="board"></div>
      <div>Turn: <span id="turnLabel"></span></div>
      <div>Clocks — White: <span id="clockWhite"></span> — Black: <span id="clockBlack"></span></div>
      <div>Play vs AI: <button id="btnVsAI">Start AI Game</button></div>
      <div id="chat" style="height:120px;overflow:auto;border:1px solid #ccc;padding:6px;margin-top:6px"></div>
      <input id="chatInput" placeholder="chat" style="width:320px"/><button id="sendChat">Send</button>
    </div>
  </div>
</div>

<script>
const pieceToChar = {"p":"♟","r":"♜","n":"♞","b":"♝","q":"♛","k":"♚","P":"♙","R":"♖","N":"♘","B":"♗","Q":"♕","K":"♔"};
let currentUser=null, authToken=null, ws=null, currentRoom=null, fen=null, selected=null;
let boardEl = document.getElementById("board");
function buildBoard(){ boardEl.innerHTML=""; for(let r=0;r<8;r++) for(let c=0;c<8;c++){ let s=document.createElement("div"); s.className="square"; s.dataset.r=r; s.dataset.c=c; s.onclick=()=>onSquareClick(r,c); boardEl.appendChild(s);} resizeSquares();}
function resizeSquares(){ let w=boardEl.clientWidth, size=Math.floor(w/8); for(let el of boardEl.children){ el.style.width=size+"px"; el.style.height=size+"px"; el.style.fontSize=Math.floor(size*0.6)+"px"; }}
window.addEventListener("resize", resizeSquares); buildBoard();

function drawFromFen(fenStr){
 fen = fenStr;
 let parts = fenStr.split(" ")[0].split("/");
 for(let r=0;r<8;r++){
   let row = parts[r], col=0;
   for(let ch of row){
     if(!isNaN(ch)){ col+=parseInt(ch); continue; }
     let idx = r*8 + col; let el = boardEl.children[idx]; el.textContent = pieceToChar[ch]||""; col++;
   }
   while(col<8){ let el = boardEl.children[r*8 + col]; el.textContent=""; col++; }
 }
 for(let r=0;r<8;r++) for(let c=0;c<8;c++){ let el=boardEl.children[r*8+c]; if((r+c)%2==0){ el.classList.add("light"); el.classList.remove("dark"); } else { el.classList.add("dark"); el.classList.remove("light"); } }
}

function onSquareClick(r,c){
 if(!currentRoom) return;
 if(!selected){ selected={r,c}; highlightSelected(); } else {
   const files="abcdefgh";
   let uci = files[selected.c] + (8-selected.r) + files[c] + (8-r);
   sendMove(uci); selected=null; clearHighlights();
 }
}
function highlightSelected(){ for(let el of boardEl.children) el.style.outline=""; if(selected) boardEl.children[selected.r*8+selected.c].style.outline="3px solid yellow"; }
function clearHighlights(){ for(let el of boardEl.children) el.style.outline=""; }

function setUser(u, token){ currentUser=u; authToken=token; document.getElementById("userBadge").textContent="User: "+u; fetch("/api/user/"+encodeURIComponent(u)).then(r=>r.json()).then(j=>{ document.getElementById("ratingBadge").textContent="Rating: "+j.rating; }); document.getElementById("main").style.display=""; document.getElementById("username").value=""; document.getElementById("password").value=""; }

document.getElementById("btnRegister").onclick = ()=>{
 let u=document.getElementById("username").value.trim(); let p=document.getElementById("password").value;
 if(!u||!p) return alert("enter username & password");
 fetch("/api/register", {method:"POST", body: new URLSearchParams({username:u,password:p})}).then(r=>r.json()).then(j=>{ if(j.ok) alert("Registered! Now login."); else alert("Register failed: "+j.reason); });
}
document.getElementById("btnLogin").onclick = ()=>{
 let u=document.getElementById("username").value.trim(); let p=document.getElementById("password").value;
 if(!u||!p) return alert("enter username & password");
 fetch("/api/login", {method:"POST", body: new URLSearchParams({username:u,password:p})}).then(r=>r.json()).then(j=>{ if(j.ok){ setUser(u,j.token);} else alert("Login failed: "+j.reason); });
}

document.getElementById("btnCreate").onclick = ()=>{
 let t=parseInt(document.getElementById("timeControl").value||300), inc=parseInt(document.getElementById("incControl").value||0);
 let priv=document.getElementById("privateRoom").checked?1:0;
 if(!authToken) return alert("login first");
 fetch("/api/create_room", {method:"POST", body:new URLSearchParams({time:t,inc:inc,private:priv,token:authToken})}).then(r=>r.json()).then(j=>{ if(j.ok) joinRoom(j.room_id); else alert("create failed"); });
}
document.getElementById("btnRefresh").onclick = loadLobby;
function loadLobby(){ fetch("/api/rooms").then(r=>r.json()).then(list=>{ let el=document.getElementById("lobby"); el.innerHTML=""; if(list.length==0) el.textContent="No public rooms"; for(let r of list){ let d=document.createElement("div"); d.className="room"; d.innerHTML=`<b>Room ${r.room_id}</b><div>Players: ${r.players.white||"-"} vs ${r.players.black||"-"}</div><div>Time ${r.time_control}s + ${r.increment}s</div><button onclick="joinRoom('${r.room_id}')">Join</button>`; el.appendChild(d);} }); }
loadLobby();

function joinRoom(room_id){ if(!authToken) return alert("login first"); fetch("/api/join_room", {method:"POST", body:new URLSearchParams({room_id,token:authToken})}).then(r=>r.json()).then(j=>{ if(!j.ok) return alert("join failed: "+j.reason); currentRoom = room_id; document.getElementById("currentRoomInfo").innerText = "Room: "+room_id; openWS(room_id); }); }

function openWS(room_id){
 if(ws){ try{ ws.close(); }catch{} ws=null; }
 ws = new WebSocket("ws://"+location.host+"/ws/"+room_id+"/"+encodeURIComponent(authToken));
 ws.onopen = ()=> appendChat("Connected to room "+room_id);
 ws.onmessage = (e)=> {
   let msg = JSON.parse(e.data);
   if(msg.type==="init"||msg.type==="move"){ drawFromFen(msg.fen); updateClocks(msg); }
   else if(msg.type==="illegal"){ alert("Illegal move!"); }
   else if(msg.type==="chat"){ appendChat(msg.user + ": " + msg.text); }
   else if(msg.type==="gameover"){ appendChat("GAME OVER: " + msg.result); fetch("/api/user/"+encodeURIComponent(currentUser)).then(r=>r.json()).then(j=>{ document.getElementById("ratingBadge").textContent="Rating: "+j.rating; }); }
 };
 ws.onclose = ()=> appendChat("Socket closed");
}

function sendMove(uci){ if(!ws) return alert("not connected"); ws.send(JSON.stringify({type:"move", uci:uci})); }

function updateClocks(msg){ if(msg.turn) document.getElementById("turnLabel").innerText = msg.turn; if(msg.remaining){ document.getElementById("clockWhite").innerText = Math.max(0,msg.remaining.white.toFixed(1)) + "s"; document.getElementById("clockBlack").innerText = Math.max(0,msg.remaining.black.toFixed(1)) + "s"; } }

function appendChat(t){ let ch=document.getElementById("chat"); ch.textContent += "\n"+t; ch.scrollTop = ch.scrollHeight; }
document.getElementById("sendChat").onclick = ()=>{ let txt=document.getElementById("chatInput").value; if(!ws) return alert("connect to a room"); ws.send(JSON.stringify({type:"chat", user:currentUser, text:txt})); document.getElementById("chatInput").value=""; }

document.getElementById("btnVsAI").onclick = ()=>{ if(!authToken) return alert("login first"); let t=parseInt(document.getElementById("timeControl").value||300), inc=parseInt(document.getElementById("incControl").value||0); fetch("/api/create_ai_room", {method:"POST", body:new URLSearchParams({time:t,inc:inc,token:authToken})}).then(r=>r.json()).then(j=>{ if(j.ok) joinRoom(j.room_id); }); }
</script>
</body></html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(INDEX_HTML)

# ---------------- API endpoints ----------------
@app.post("/api/register")
async def api_register(request: Request):
    form = await request.form()
    username = (form.get("username") or "").strip()
    password = form.get("password") or ""
    ok, info = register_user(username, password)
    if ok:
        return {"ok": True, "user": info}
    return {"ok": False, "reason": info}

@app.post("/api/login")
async def api_login(request: Request):
    form = await request.form()
    username = (form.get("username") or "").strip()
    password = form.get("password") or ""
    ok, info = verify_login(username, password)
    if ok:
        return {"ok": True, "token": info}
    else:
        return {"ok": False, "reason": info}

@app.get("/api/user/{username}")
async def api_get_user(username: str):
    u = users.get(username)
    if not u:
        return HTMLResponse("not found", status_code=404)
    return {"username": username, "rating": u["rating"], "games_played": u.get("games_played", 0)}

@app.get("/api/rooms")
async def api_rooms():
    return list_public_rooms()

@app.post("/api/create_room")
async def api_create_room(request: Request):
    form = await request.form()
    t = int(form.get("time") or 300)
    inc = int(form.get("inc") or 0)
    priv = bool(int(form.get("private") or 0))
    token = form.get("token")
    username = get_user_by_token(token) if token else None
    room = create_room(time_control=t, increment=inc, is_private=priv)
    if username:
        rooms[room.room_id].players_usernames["white"] = username
    save_games_store()
    return {"ok": True, "room_id": room.room_id}

@app.post("/api/create_ai_room")
async def api_create_ai_room(request: Request):
    form = await request.form()
    t = int(form.get("time") or 300)
    inc = int(form.get("inc") or 0)
    token = form.get("token")
    username = get_user_by_token(token) if token else None
    room = create_room(time_control=t, increment=inc, is_private=True)
    if username:
        rooms[room.room_id].players_usernames["white"] = username
    rooms[room.room_id].players_usernames["black"] = "AI"
    save_games_store()
    return {"ok": True, "room_id": room.room_id}

@app.post("/api/join_room")
async def api_join_room(request: Request):
    form = await request.form()
    room_id = form.get("room_id")
    token = form.get("token")
    username = get_user_by_token(token) if token else None
    if room_id not in rooms:
        return {"ok": False, "reason": "no such room"}
    room = rooms[room_id]
    if room.players_usernames.get("white") and room.players_usernames.get("black") and room.players_usernames.get("white")!="AI" and room.players_usernames.get("black")!="AI":
        return {"ok": True, "role": "spectator"}
    return {"ok": True, "room_id": room_id}

# persist game summaries
def save_games_store():
    with file_lock:
        persist = {}
        for k, v in rooms.items():
            persist[k] = {
                "players": v.players_usernames,
                "started": v.started,
                "time_control": v.time_control,
                "increment": v.increment,
                "fen": v.game.fen(),
                "moves": v.move_list,
                "result": v.result
            }
        save_json_atomic(GAMES_FILE, persist)

# ---------------- WebSocket endpoint ----------------
@app.websocket("/ws/{room_id}/{token}")
async def ws_room(ws: WebSocket, room_id: str, token: str):
    await ws.accept()
    username = get_user_by_token(token)
    if not username:
        await ws.send_json({"type":"error","message":"invalid token"})
        await ws.close()
        return
    if room_id not in rooms:
        rooms[room_id] = GameRoom(room_id)
    room = rooms[room_id]
    role = room.add_player(ws, username)
    room.start_if_ready()
    await ws.send_json({"type":"init", "fen": room.game.fen(), "remaining": room.remaining, "turn": room.turn})
    try:
        if (room.players_usernames.get("white")=="AI" and room.turn=="white") or (room.players_usernames.get("black")=="AI" and room.turn=="black"):
            await maybe_handle_ai_turn(room)
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except:
                continue
            if msg.get("type") == "chat":
                payload = {"type":"chat","user":msg.get("user"), "text": msg.get("text")}
                await broadcast_room(room, payload)
                continue
            if msg.get("type") == "move":
                uci = msg.get("uci")
                ok, res = room.apply_move(uci)
                if not ok:
                    if "time forfeit" in res:
                        await broadcast_room(room, {"type":"gameover","result":room.result})
                        finalize_game(room)
                    else:
                        await ws.send_json({"type":"illegal", "reason": res})
                    continue
                await broadcast_room(room, {"type":"move","fen":res, "remaining": room.remaining, "turn": room.turn})
                if room.result:
                    await broadcast_room(room, {"type":"gameover","result":room.result})
                    finalize_game(room)
                    continue
                await maybe_handle_ai_turn(room)
    except WebSocketDisconnect:
        room.remove_ws(ws)
        save_games_store()
    except Exception:
        try:
            room.remove_ws(ws)
        except:
            pass
        traceback.print_exc()

async def broadcast_room(room: GameRoom, payload: Dict[str, Any]):
    to_remove = []
    for c in ["white","black"]:
        sock = room.players.get(c)
        if sock:
            try:
                await sock.send_json(payload)
            except:
                to_remove.append(sock)
    for s in list(room.spectators):
        try:
            await s.send_json(payload)
        except:
            to_remove.append(s)
    for r in to_remove:
        room.remove_ws(r)

async def maybe_handle_ai_turn(room: GameRoom):
    with room.lock:
        if (room.players_usernames.get("white")=="AI" and room.turn=="white") or (room.players_usernames.get("black")=="AI" and room.turn=="black"):
            fen = room.game.fen()
        else:
            return
    mv = await choose_ai_move(fen)
    if not mv:
        with room.lock:
            room.result = room.game.result() or "1/2-1/2"
        await broadcast_room(room, {"type":"gameover","result":room.result})
        finalize_game(room)
        return
    await asyncio.sleep(0.2 + random.random()*0.6)
    ok, res = room.apply_move(mv)
    if ok:
        await broadcast_room(room, {"type":"move","fen":res, "remaining": room.remaining, "turn": room.turn})
        if room.result:
            await broadcast_room(room, {"type":"gameover","result":room.result})
            finalize_game(room)

# ---------------- finalize game ----------------
def finalize_game(room: GameRoom):
    global _next_game_id
    payload = room.to_result_payload()
    p_white = room.players_usernames.get("white")
    p_black = room.players_usernames.get("black")
    res = room.result
    s_white = s_black = 0.5
    if res == "1-0": s_white, s_black = 1.0, 0.0
    elif res == "0-1": s_white, s_black = 0.0, 1.0
    elif res == "1/2-1/2": s_white, s_black = 0.5, 0.5
    if p_white and p_black and p_white in users and p_black in users and p_white!="AI" and p_black!="AI":
        r1 = users[p_white]["rating"]; r2 = users[p_black]["rating"]
        nr1 = new_elo(r1, r2, s_white); nr2 = new_elo(r2, r1, s_black)
        users[p_white]["rating"] = nr1; users[p_white]["games_played"] = users[p_white].get("games_played",0)+1
        users[p_black]["rating"] = nr2; users[p_black]["games_played"] = users[p_black].get("games_played",0)+1
        save_users()
    with file_lock:
        gid = str(_next_game_id); _next_game_id += 1
        games_store[gid] = payload
        save_json_atomic(GAMES_FILE, games_store)
        save_games_store()

# ---------------- run server ----------------
if __name__ == "__main__":
    print("GPT Chess (single-file) starting on http://localhost:8000")
    print("This uses Lichess cloud-eval for AI moves (no API key). For heavy use run local Stockfish instead.")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
