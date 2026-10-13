/* Tiện ích dùng chung: gọi API, toast, hộp thoại xác nhận, socket, huy hiệu điều hướng */
(function () {
  "use strict";
  const csrf = () => (document.querySelector('meta[name="csrf-token"]') || {}).content || "";

  async function api(method, url, body, opts = {}) {
    const init = { method, headers: { "X-CSRFToken": csrf(), Accept: "application/json" }, credentials: "same-origin" };
    if (body instanceof FormData) init.body = body;
    else if (body !== undefined && body !== null) { init.headers["Content-Type"] = "application/json"; init.body = JSON.stringify(body); }
    let res;
    try { res = await fetch(url, init); } catch (e) { const err = new Error("Không kết nối được máy chủ."); err.network = true; throw err; }
    let data = null;
    try { data = await res.json(); } catch (e) { /* không phải JSON */ }
    if (!res.ok) {
      if (!opts.silent) {
        if (res.status === 401) { location.href = "/login"; }
        else if (res.status === 503 && data && data.code === "no_model") { location.href = "/"; }
      }
      const err = new Error((data && data.error) || `Lỗi ${res.status}`);
      err.status = res.status; err.code = data && data.code;
      throw err;
    }
    return data;
  }

  function toast(msg, type = "") {
    const box = document.getElementById("toasts");
    if (!box) return;
    const el = document.createElement("div");
    el.className = "toast " + type; el.textContent = msg;
    box.appendChild(el);
    setTimeout(() => { el.style.opacity = "0"; el.style.transition = "opacity .3s"; setTimeout(() => el.remove(), 300); }, 4200);
  }

  function dialog({ title, text, ok = "Đồng ý", cancel = "Hủy", danger = false, icon = "", single = false, key = "" }) {
    return new Promise((resolve) => {
      if (key && document.querySelector(`.overlay[data-key="${key}"]`)) { resolve(false); return; }   // không mở trùng
      const ov = document.createElement("div"); if (key) ov.dataset.key = key;
      ov.className = "overlay"; ov.setAttribute("role", "dialog"); ov.setAttribute("aria-modal", "true");
      const d = document.createElement("div"); d.className = "dialog" + (danger ? " danger" : "");
      if (icon) { const i = document.createElement("div"); i.className = "big-icon"; i.textContent = icon; d.appendChild(i); }
      const h = document.createElement("h3"); h.textContent = title; d.appendChild(h);
      const p = document.createElement("p"); p.textContent = text; d.appendChild(p);
      const a = document.createElement("div"); a.className = "actions";
      const done = (v) => { ov.remove(); document.removeEventListener("keydown", onKey); resolve(v); };
      if (!single) { const c = document.createElement("button"); c.className = "btn"; c.textContent = cancel; c.onclick = () => done(false); a.appendChild(c); }
      const o = document.createElement("button"); o.className = "btn " + (danger ? "btn-danger" : "btn-primary"); o.textContent = ok; o.onclick = () => done(true); a.appendChild(o);
      d.appendChild(a); ov.appendChild(d); document.body.appendChild(ov); o.focus();
      const onKey = (e) => { if (e.key === "Escape") done(false); };
      document.addEventListener("keydown", onKey);
      ov.addEventListener("mousedown", (e) => { if (e.target === ov && !single) done(false); });
    });
  }

  function avatar(user, cls = "") {
    const el = document.createElement("span");
    el.className = "avatar " + cls; el.style.background = user.color || "#1b74e4";
    el.textContent = (user.username || "?").slice(0, 1);
    if (user.online) { const dot = document.createElement("i"); dot.className = "dot"; el.appendChild(dot); }
    return el;
  }

  function fmtTime(iso) {
    if (!iso) return "";
    const d = new Date(iso), now = new Date();
    const hm = d.toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit" });
    if (d.toDateString() === now.toDateString()) return hm;
    const diff = (now - d) / 86400000;
    if (diff < 6) return d.toLocaleDateString("vi-VN", { weekday: "short" });
    return d.toLocaleDateString("vi-VN", { day: "2-digit", month: "2-digit" });
  }
  function fmtFull(iso) { return iso ? new Date(iso).toLocaleString("vi-VN") : ""; }
  function fmtSize(n) { return n < 1024 ? n + " B" : n < 1048576 ? (n / 1024).toFixed(1) + " KB" : (n / 1048576).toFixed(1) + " MB"; }
  function lastSeen(iso) {
    if (!iso) return "Ngoại tuyến";
    const m = Math.max(0, Math.round((Date.now() - new Date(iso)) / 60000));
    if (m < 1) return "Vừa mới hoạt động"; if (m < 60) return `Hoạt động ${m} phút trước`;
    if (m < 1440) return `Hoạt động ${Math.round(m / 60)} giờ trước`; return `Hoạt động ${Math.round(m / 1440)} ngày trước`;
  }

  const SC = window.SC = { api, toast, dialog, avatar, fmtTime, fmtFull, fmtSize, lastSeen, socket: null, me: (window.SAFECHAT || {}).me || null };

  // ---- menu người dùng ----
  const mbtn = document.getElementById("user-menu-btn"), menu = document.getElementById("user-menu");
  if (mbtn && menu) {
    mbtn.addEventListener("click", (e) => { e.stopPropagation(); const open = menu.classList.toggle("hidden") === false; mbtn.setAttribute("aria-expanded", open); });
    document.addEventListener("click", () => { menu.classList.add("hidden"); mbtn.setAttribute("aria-expanded", "false"); });
  }

  // ---- huy hiệu + socket (chỉ khi đã đăng nhập) ----
  if (SC.me && window.io) {
    let timer = null;
    const refresh = () => { clearTimeout(timer); timer = setTimeout(async () => {
      try {
        const s = await api("GET", "/api/summary", null, { silent: true });
        const n = document.getElementById("nav-unread");
        if (n) { n.textContent = s.unread_total > 99 ? "99+" : s.unread_total; n.classList.toggle("hidden", !s.unread_total); }
        const f = document.getElementById("feed-requests");
        if (f) { f.textContent = s.requests; f.classList.toggle("hidden", !s.requests); }
        const m = document.getElementById("model-name"); if (m && s.model) m.textContent = s.model;
        SC.requests = s.requests; document.dispatchEvent(new CustomEvent("sc:summary", { detail: s }));
      } catch (e) { /* bỏ qua */ }
    }, 120); };
    SC.refreshSummary = refresh;
    SC.socket = io({ transports: ["websocket", "polling"], reconnection: true });
    ["message:new", "conversation:read", "friend:update", "block:update"].forEach((ev) => SC.socket.on(ev, refresh));
    SC.socket.on("connect", refresh);
    SC.socket.on("system:model_changed", () => { location.href = "/"; });   // quản trị viên đổi mô hình -> đăng xuất
    refresh();
  }
})();
