"use strict";

const $ = (id) => document.getElementById(id);
const state = { prevPoints: null, prevSets: null, demo: false };

if (new URLSearchParams(location.search).get("compact") === "1") {
  document.body.classList.add("compact");
}

function fmtTime(ts) {
  if (!ts) return "";
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString("zh-CN", { hour12: false });
}

function sideLabel(side) {
  if (side === "A") return "A";
  if (side === "B") return "B";
  return "";
}

function actionText(item, players) {
  const who = sideLabel(item.side);
  const name = who ? (players[who] || `选手 ${who}`) : "";
  const source = { vision: "视觉", voice: "语音", button: "按键", human: "人工", ui: "演示" }[item.source] || item.source || "";
  switch (item.action) {
    case "award": return `${source}判定 ${name} 得分`;
    case "undo": return "撤销上一分";
    case "redo": return "重做";
    case "reset": return "重置比分";
    case "let": return "擦网重发";
    case "propose_pending": return "提出待确认回合";
    case "reject_pending": return "取消待确认";
    default: return `${item.action || "动作"}`;
  }
}

function chip(text, cls) {
  const el = document.createElement("span");
  el.className = "chip " + (cls || "");
  el.textContent = text;
  return el;
}

function render(data) {
  $("offline").hidden = true;
  $("title").textContent = data.title || "乒乓球智能裁判";
  $("set-label").textContent = `第 ${data.set_index + 1} 局`;
  $("format").textContent = data.best_of === 3 ? "三局两胜" : "五局三胜";

  const status = $("status");
  status.className = "status";
  if (data.match_over) {
    status.classList.add("over");
    status.textContent = "比赛结束";
  } else if (data.pending) {
    status.classList.add("pending");
    status.textContent = "待人工确认";
  } else {
    status.textContent = "比赛进行中";
  }

  $("name-a").textContent = data.players.A;
  $("name-b").textContent = data.players.B;
  $("points-a").textContent = data.points.A;
  $("points-b").textContent = data.points.B;
  $("sets-a").textContent = data.sets.A;
  $("sets-b").textContent = data.sets.B;

  const aLeft = data.side_of_a === "left";
  $("side-a").textContent = aLeft ? "左半台" : "右半台";
  $("side-b").textContent = aLeft ? "右半台" : "左半台";

  $("serve-a").classList.toggle("active", data.server === "A");
  $("serve-b").classList.toggle("active", data.server === "B");
  $("player-a").classList.toggle("serving", data.server === "A");
  $("player-b").classList.toggle("serving", data.server === "B");
  $("player-a").classList.toggle("leading", data.points.A > data.points.B);
  $("player-b").classList.toggle("leading", data.points.B > data.points.A);

  const meta = $("meta-strip");
  if (meta) {
    const serverSide = data.server === "A"
      ? (data.side_of_a === "left" ? "左半台" : "右半台")
      : (data.side_of_a === "left" ? "右半台" : "左半台");
    meta.textContent = `第 ${data.set_index + 1} 局 · 局分 ${data.sets.A} : ${data.sets.B}` +
      ` · 发球 ${data.players[data.server]}（${serverSide}）`;
  }

  const flags = $("flags");
  flags.innerHTML = "";
  if (data.match_over) {
    const winner = data.sets.A > data.sets.B ? "A" : "B";
    flags.appendChild(chip(`冠军 ${data.players[winner]}`, "match"));
  } else if (data.match_point) {
    flags.appendChild(chip(`赛点 ${data.players[data.match_point]}`, "match"));
  } else if (data.set_point) {
    flags.appendChild(chip(`局点 ${data.players[data.set_point]}`, "point"));
  }
  if (data.deuce) flags.appendChild(chip("平分", "deuce"));

  const pending = $("pending");
  if (data.pending) {
    const side = data.pending.side;
    const name = side === "A" || side === "B" ? data.players[side] : "某一方";
    pending.textContent = `待人工确认：${name} 疑似得分（语音说“确认这一分”或按键确认）`;
    pending.hidden = false;
  } else {
    pending.hidden = true;
  }

  const recent = $("recent");
  recent.innerHTML = "";
  (data.recent || []).slice(-4).reverse().forEach((item) => {
    const row = document.createElement("div");
    row.className = "item";
    const time = document.createElement("span");
    time.className = "time";
    time.textContent = fmtTime(item.ts);
    const text = document.createElement("span");
    text.textContent = actionText(item, data.players);
    row.append(time, text);
    recent.appendChild(row);
  });

  const demo = $("demo");
  demo.hidden = !data.demo;
  state.demo = data.demo;

  if (state.prevPoints) {
    bumpIfChanged("player-a", state.prevPoints.A, data.points.A);
    bumpIfChanged("player-b", state.prevPoints.B, data.points.B);
  }
  state.prevPoints = { ...data.points };
  state.prevSets = { ...data.sets };
}

function bumpIfChanged(id, before, after) {
  if (before === after) return;
  const el = $(id);
  el.classList.remove("bump");
  void el.offsetWidth;
  el.classList.add("bump");
}

async function refresh() {
  try {
    const response = await fetch("/api/state", { cache: "no-store" });
    render(await response.json());
  } catch (error) {
    $("offline").hidden = false;
  }
}

async function demoAction(action) {
  try {
    await fetch(`/api/demo/${action}`, { method: "POST" });
  } finally {
    refresh();
  }
}

document.querySelectorAll("[data-demo]").forEach((button) => {
  button.addEventListener("click", () => demoAction(button.dataset.demo));
});

document.addEventListener("keydown", (event) => {
  if (!state.demo) return;
  const key = event.key.toLowerCase();
  const map = {
    a: "award?side=A", b: "award?side=B", z: "undo", y: "redo",
    p: "pending", c: "confirm", x: "reject", r: "reset",
  };
  if (map[key]) demoAction(map[key]);
});

refresh();
setInterval(refresh, 500);
