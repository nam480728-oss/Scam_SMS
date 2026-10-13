/* Trang Tin nhắn: danh sách, bạn bè, chat thời gian thực, cảnh báo AI 3 mức */
(function () {
  "use strict";
  const SC = window.SC, me = SC.me, api = SC.api, socket = SC.socket;
  const $ = (id) => document.getElementById(id);
  const layout = $("chat-layout");
  if (!layout || !me) return;

  const EMOJIS = "😀 😃 😄 😁 😆 😅 😂 🤣 🙂 😉 😊 😇 🥰 😍 😘 😋 😛 😜 🤪 🤗 🤔 🤐 😐 😶 🙄 😏 😴 😌 😔 😢 😭 😤 😠 😡 🥺 😱 😨 😰 🤯 😳 🥳 😎 🤓 🤝 👍 👎 👏 🙌 🙏 💪 👌 ✌️ 🤞 👋 ❤️ 🧡 💛 💚 💙 💜 🖤 💔 🔥 ✨ 🎉 🎁 🍀 ☕ 🍜 🍕 🎂 🐶 🐱 ⚠️ 🚫 ✅ ❌ 💯".split(" ");

  const state = {
    convs: new Map(), msgs: new Map(), hasMore: new Map(), activeId: null, tab: "chats",
    friends: [], requests: { incoming: [], outgoing: [] }, blocked: [], search: "", results: [],
    pendingFile: null, sending: false, typingUntil: 0, typingSent: 0, firstConnect: true, loadingOlder: false,
  };
  const active = () => state.convs.get(state.activeId) || null;

  /* ---------------------------------------------------------- tiện ích DOM */
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
  const btn = (label, cls, fn) => { const b = el("button", "btn btn-sm " + (cls || ""), label); b.type = "button"; b.addEventListener("click", (e) => { e.stopPropagation(); fn(b); }); return b; };
  const guard = (fn) => async (...a) => { try { return await fn(...a); } catch (e) { if (e.status !== 401) SC.toast(e.message, "error"); } };

  function linkify(text, parent, clickable) {
    const re = /(https?:\/\/[^\s<]+|www\.[^\s<]+)/gi; let last = 0, m;
    while ((m = re.exec(text))) {
      if (m.index > last) parent.appendChild(document.createTextNode(text.slice(last, m.index)));
      const url = m[0];
      if (clickable) { const a = el("a", null, url); a.href = url.startsWith("www.") ? "https://" + url : url; a.target = "_blank"; a.rel = "noopener noreferrer nofollow"; parent.appendChild(a); }
      else { const s = el("span", null, url); s.style.textDecoration = "underline dotted"; s.title = "Liên kết bị vô hiệu hóa vì tin nhắn nghi lừa đảo"; parent.appendChild(s); }
      last = m.index + url.length;
    }
    if (last < text.length) parent.appendChild(document.createTextNode(text.slice(last)));
  }

  /* ------------------------------------------------------------ dữ liệu */
  function setConv(c) {
    const old = state.convs.get(c.id);
    state.convs.set(c.id, c);
    return old;
  }
  function sortedConvs() {
    return [...state.convs.values()].filter((c) => c.last_message_at || c.id === state.activeId)
      .sort((a, b) => (b.last_message_at || "").localeCompare(a.last_message_at || "") || b.id - a.id);
  }
  function upsertMessage(m) {
    const arr = state.msgs.get(m.conversation_id) || [];
    const i = arr.findIndex((x) => x.id === m.id);
    if (i >= 0) arr[i] = { ...arr[i], ...m }; else { arr.push(m); arr.sort((a, b) => a.id - b.id); }
    state.msgs.set(m.conversation_id, arr);
  }
  const loadConversations = guard(async () => {
    const d = await api("GET", "/api/conversations");
    const fresh = new Map(d.conversations.map((c) => [c.id, c]));
    const a = active();
    if (a && !fresh.has(a.id)) fresh.set(a.id, a);              // giữ cuộc chat trống đang mở
    state.convs = fresh;
    renderList();
    if (state.activeId && state.convs.has(state.activeId)) renderHeader();
  });
  const loadFriends = guard(async () => { state.friends = (await api("GET", "/api/friends")).friends; if (state.tab === "friends" && !state.search) renderList(); });
  const loadRequests = guard(async () => {
    state.requests = await api("GET", "/api/friend-requests");
    const n = state.requests.incoming.length, b = $("tab-req-badge");
    b.textContent = n; b.classList.toggle("hidden", !n);
    if (state.tab === "requests" && !state.search) renderList();
  });
  const loadBlocked = guard(async () => { state.blocked = (await api("GET", "/api/blocks")).blocked; if (state.tab === "blocked" && !state.search) renderList(); });

  /* ------------------------------------------------------- cột trái: list */
  function convItem(c) {
    const it = el("div", "item" + (c.id === state.activeId ? " active" : "") + (c.unread ? " unread" : ""));
    it.tabIndex = 0; it.setAttribute("role", "button");
    it.appendChild(SC.avatar(c.other, ""));
    const meta = el("div", "meta"), name = el("div", "name"), nm = el("span", null, c.other.username);
    if (c.warning > 0 && !c.i_blocked) nm.appendChild(el("span", "warn-mark", c.warning >= 2 ? "🚨" : "⚠️"));
    name.append(nm, el("time", null, SC.fmtTime(c.last_message_at)));
    const lm = c.last_message;
    meta.append(name, el("div", "snippet", lm ? (lm.mine ? "Bạn: " : "") + lm.preview : "Chưa có tin nhắn"));
    it.appendChild(meta);
    if (c.unread) it.appendChild(el("span", "badge", c.unread > 99 ? "99+" : c.unread));
    const open = () => openConversation(c.id);
    it.addEventListener("click", open); it.addEventListener("keydown", (e) => { if (e.key === "Enter") open(); });
    return it;
  }
  function personItem(u, actions) {
    const it = el("div", "item");
    it.appendChild(SC.avatar(u, ""));
    const meta = el("div", "meta"); meta.appendChild(el("div", "name", u.username));
    meta.appendChild(el("div", "snippet", u.online ? "Đang hoạt động" : ""));
    it.appendChild(meta);
    const a = el("div", "actions"); actions.forEach((x) => a.appendChild(x)); it.appendChild(a);
    return it;
  }
  const empty = (icon, text) => { const d = el("div", "empty"); d.append(el("span", "big", icon), document.createTextNode(text)); return d; };

  function renderList() {
    const list = $("list"); list.replaceChildren();
    if (state.search) return renderSearch(list);
    if (state.tab === "chats") {
      const cs = sortedConvs();
      if (!cs.length) list.appendChild(empty("💬", "Chưa có đoạn chat nào. Vào tab Bạn bè hoặc tìm kiếm để bắt đầu."));
      cs.forEach((c) => list.appendChild(convItem(c)));
    } else if (state.tab === "friends") {
      if (!state.friends.length) list.appendChild(empty("👥", "Chưa có bạn bè. Gõ tên tài khoản vào ô tìm kiếm để kết bạn."));
      state.friends.forEach((u) => list.appendChild(personItem(u, [
        u.i_blocked ? btn("Bỏ chặn", "", () => unblock(u)) : btn("Nhắn tin", "btn-primary", () => openWithUser(u.id))])));
    } else if (state.tab === "requests") {
      const inc = state.requests.incoming, out = state.requests.outgoing;
      if (!inc.length && !out.length) list.appendChild(empty("📭", "Không có lời mời nào."));
      if (inc.length) list.appendChild(el("div", "section-title", "Lời mời đến"));
      inc.forEach((r) => list.appendChild(personItem(r.user, [
        btn("Chấp nhận", "btn-primary", () => respond(r.id, true)), btn("Từ chối", "", () => respond(r.id, false))])));
      if (out.length) list.appendChild(el("div", "section-title", "Đã gửi"));
      out.forEach((r) => list.appendChild(personItem(r.user, [btn("Hủy lời mời", "", () => cancelRequest(r.id))])));
    } else if (state.tab === "blocked") {
      if (!state.blocked.length) list.appendChild(empty("🛡️", "Bạn chưa chặn ai."));
      state.blocked.forEach((u) => list.appendChild(personItem(u, [btn("Bỏ chặn", "", () => unblock(u))])));
    }
  }

  function renderSearch(list) {
    list.appendChild(el("div", "section-title", `Kết quả cho “${state.search}”`));
    if (!state.results.length) { list.appendChild(empty("🔎", "Không tìm thấy người dùng nào.")); return; }
    state.results.forEach((u) => {
      let acts = [];
      if (u.i_blocked) acts = [btn("Bỏ chặn", "", () => unblock(u))];
      else if (u.relation === "friend") acts = [btn("Nhắn tin", "btn-primary", () => openWithUser(u.id))];
      else if (u.relation === "sent") acts = [btn("Hủy lời mời", "", () => cancelRequest(u.request_id))];
      else if (u.relation === "received") acts = [btn("Chấp nhận", "btn-primary", () => respond(u.request_id, true)), btn("Từ chối", "", () => respond(u.request_id, false))];
      else acts = [btn("Kết bạn", "btn-primary", () => addFriend(u))];
      list.appendChild(personItem(u, acts));
    });
  }
  let searchTimer = null, searchSeq = 0;
  $("search").addEventListener("input", (e) => {
    const q = e.target.value.trim(); state.search = q; clearTimeout(searchTimer);
    if (!q) { renderList(); return; }
    searchTimer = setTimeout(guard(async () => {
      const seq = ++searchSeq;
      const d = await api("GET", "/api/users/search?q=" + encodeURIComponent(q));
      if (seq === searchSeq && state.search === q) { state.results = d.users; renderList(); }
    }), 250);
  });
  const refreshSearch = async () => {
    if (!state.search) return;
    try { state.results = (await api("GET", "/api/users/search?q=" + encodeURIComponent(state.search), null, { silent: true })).users; renderList(); } catch (e) { /* bỏ qua */ }
  };
  $("tabs").addEventListener("click", (e) => {
    const t = e.target.closest(".tab"); if (!t) return;
    state.tab = t.dataset.tab; state.search = ""; $("search").value = "";
    document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === t));
    if (state.tab === "friends") loadFriends(); if (state.tab === "requests") loadRequests(); if (state.tab === "blocked") loadBlocked();
    renderList();
  });

  /* ------------------------------------------------- hành động bạn bè/chặn */
  const afterSocial = async () => { await Promise.all([loadFriends(), loadRequests(), loadBlocked(), loadConversations(), refreshSearch()]); };
  const addFriend = guard(async (u) => { await api("POST", "/api/friend-requests", { user_id: u.id }); SC.toast(`Đã gửi lời mời kết bạn tới ${u.username}`, "ok"); await afterSocial(); });
  const respond = guard(async (rid, ok) => { await api("POST", `/api/friend-requests/${rid}/${ok ? "accept" : "decline"}`); SC.toast(ok ? "Đã trở thành bạn bè 🎉" : "Đã từ chối lời mời", ok ? "ok" : ""); await afterSocial(); });
  const cancelRequest = guard(async (rid) => { await api("DELETE", `/api/friend-requests/${rid}`); await afterSocial(); });
  const unblock = guard(async (u) => { await api("DELETE", `/api/blocks/${u.id}`); SC.toast(`Đã bỏ chặn ${u.username}`, "ok"); await afterSocial(); });
  const unfriend = guard(async (u) => {
    if (!(await SC.dialog({ title: `Hủy kết bạn với ${u.username}?`, text: "Hai bạn sẽ không thể nhắn tin cho nhau cho đến khi kết bạn lại.", ok: "Hủy kết bạn", danger: true }))) return;
    await api("DELETE", `/api/friends/${u.id}`); await afterSocial();
  });
  const blockUser = guard(async (u) => {
    const okk = await SC.dialog({ title: `Chặn ${u.username}?`, icon: "🛡️", danger: true, ok: "Chặn",
      text: `${u.username} sẽ không thể nhắn tin cho bạn và bạn cũng không nhắn được cho họ. Lịch sử trò chuyện vẫn được giữ. Bạn có thể bỏ chặn bất cứ lúc nào.` });
    if (!okk) return;
    await api("POST", "/api/blocks", { user_id: u.id }); SC.toast(`Đã chặn ${u.username}`, "ok"); await afterSocial();
  });

  /* ------------------------------------------------------- mở cuộc trò chuyện */
  const openWithUser = guard(async (uid) => {
    const d = await api("POST", "/api/conversations", { user_id: uid });
    setConv(d.conversation); await openConversation(d.conversation.id);
    state.tab = "chats"; state.search = ""; $("search").value = "";
    document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x.dataset.tab === "chats"));
    renderList();
  });

  const openConversation = guard(async (cid) => {
    state.activeId = cid;
    $("chat-empty").classList.add("hidden"); $("chat-box").classList.remove("hidden"); layout.classList.add("show-chat");
    clearAttachment(); $("input").value = ""; autosize(); $("typing").textContent = "";
    let detail;
    try { detail = (await api("GET", `/api/conversations/${cid}`)).conversation; }
    catch (e) { state.activeId = null; $("chat-box").classList.add("hidden"); $("chat-empty").classList.remove("hidden"); layout.classList.remove("show-chat"); history.replaceState(null, "", "/messages"); throw e; }
    setConv(detail);
    const h = await api("GET", `/api/conversations/${cid}/messages`);
    state.msgs.set(cid, h.messages); state.hasMore.set(cid, h.has_more);
    history.replaceState(null, "", `/messages/${cid}`);
    renderList(); renderHeader(); renderBanners(); renderMessages(true); renderComposerState();
    markRead(); if (!matchMedia("(max-width:860px)").matches) $("input").focus();
  });

  const markRead = async () => {
    const c = active(); if (!c || document.hidden) return;
    if (!c.unread && !(state.msgs.get(c.id) || []).some((m) => !m.mine && !m.read_at)) return;
    try {
      await api("POST", `/api/conversations/${c.id}/read`, null, { silent: true });
      c.unread = 0; (state.msgs.get(c.id) || []).forEach((m) => { if (!m.mine && !m.read_at) m.read_at = new Date().toISOString(); });
      renderList(); SC.refreshSummary && SC.refreshSummary();
    } catch (e) { /* bỏ qua */ }
  };

  /* --------------------------------------------------------- tiêu đề khung chat */
  function statusText(c) {
    if (c.i_blocked || c.blocked_me) return "";
    if (c.other.online) return "Đang hoạt động";
    return SC.lastSeen(c.other.last_seen);
  }
  function renderHeader() {
    const c = active(); if (!c) return;
    $("head-avatar").replaceChildren(SC.avatar(c.other, ""));
    $("head-name").textContent = c.other.username;
    $("head-status").textContent = statusText(c);
    const menu = $("head-menu"); menu.replaceChildren();
    const mk = (label, fn, danger) => { const b = el("button", null, label); b.type = "button"; if (danger) b.style.color = "var(--red)"; b.addEventListener("click", () => { menu.classList.add("hidden"); fn(); }); return b; };
    if (c.i_blocked) menu.appendChild(mk("✅ Bỏ chặn " + c.other.username, () => unblock(c.other)));
    else menu.appendChild(mk("🚫 Chặn " + c.other.username, () => blockUser(c.other), true));
    menu.appendChild(mk("💔 Hủy kết bạn", () => unfriend(c.other), true));
  }
  $("head-menu-btn").addEventListener("click", (e) => { e.stopPropagation(); $("head-menu").classList.toggle("hidden"); });
  document.addEventListener("click", () => { $("head-menu").classList.add("hidden"); $("emoji-pop").classList.add("hidden"); });
  $("back-btn").addEventListener("click", () => { layout.classList.remove("show-chat"); history.replaceState(null, "", "/messages"); });

  /* ---------------------------------------------------- banner cảnh báo 3 mức */
  const BANNERS = {
    1: { icon: "⚠️", title: "Cảnh báo mức 1", text: (n) => `Tin nhắn của ${n} có dấu hiệu lừa đảo. Hãy cẩn thận: đừng chuyển tiền, đừng cung cấp mã OTP hay mật khẩu.` },
    2: { icon: "🚨", title: "Cảnh báo mức 2 — Nghi ngờ cao", text: (n) => `${n} tiếp tục gửi nội dung nghi lừa đảo. Không bấm vào liên kết, không chuyển tiền, không cung cấp thông tin cá nhân.` },
    3: { icon: "🚨", title: "NGUY HIỂM — Cảnh báo mức 3", text: (n) => `${n} đã nhiều lần gửi nội dung nghi lừa đảo. Chúng tôi gợi ý bạn chặn người này để bảo vệ bản thân (hệ thống không tự động chặn).` },
  };
  function renderBanners(pulse) {
    const box = $("banner-box"), c = active(); box.replaceChildren(); if (!c) return;
    if (c.warning > 0 && !c.i_blocked) {
      const cfg = BANNERS[c.warning];
      const b = el("div", `banner l${c.warning}` + (pulse && c.warning >= 3 ? " pulse" : "")); b.setAttribute("role", "alert");
      const t = el("div", "btxt"); t.append(el("b", null, cfg.title), document.createTextNode(cfg.text(c.other.username)));
      b.append(el("div", "bicon", cfg.icon), t);
      if (c.warning >= 3) b.appendChild(btn("🚫 Chặn người này", "btn-danger", () => blockUser(c.other)));
      box.appendChild(b);
    }
    if (c.self_warning) {
      const b = el("div", "banner l3 self" + (pulse ? " pulse" : "")); b.setAttribute("role", "alert");
      const t = el("div", "btxt"); t.append(el("b", null, "CẢNH BÁO TỪ HỆ THỐNG"), document.createTextNode(SELF_TEXT));
      b.append(el("div", "bicon", "🚨"), t); box.appendChild(b);
    }
  }
  const SELF_TEXT = "Hệ thống phát hiện bạn đang là đối tượng đi lừa đảo người khác, hãy dừng lại không thì bạn sẽ đi TÒ";

  /* --------------------------------------------------------- hiển thị tin nhắn */
  const dayKey = (iso) => new Date(iso).toDateString();
  const dayLabel = (iso) => { const d = new Date(iso), n = new Date(); if (d.toDateString() === n.toDateString()) return "Hôm nay"; const y = new Date(n - 864e5); if (d.toDateString() === y.toDateString()) return "Hôm qua"; return d.toLocaleDateString("vi-VN"); };

  function bubbleFor(m) {
    const clickable = !(m.warning_level > 0);
    if (m.kind === "image" && m.attachment) {
      const b = el("div", "bubble img"); const img = el("img"); img.src = m.attachment.url; img.alt = m.attachment.name || "Ảnh"; img.loading = "lazy";
      img.addEventListener("click", () => lightbox(m.attachment.url));
      img.addEventListener("load", () => { if (nearBottom._stick) scrollBottom(); });
      b.appendChild(img);
      if (m.body) { const cap = el("span", "cap"); linkify(m.body, cap, clickable); b.appendChild(cap); b.style.background = m.mine ? "var(--bubble-me)" : "var(--bubble-other)"; b.style.color = m.mine ? "#fff" : ""; b.classList.remove("img"); b.style.padding = "3px"; }
      return b;
    }
    if (m.kind === "file" && m.attachment) {
      const wrap = el("div");
      const a = el("a", "fileCard"); a.href = m.attachment.url; a.setAttribute("download", m.attachment.name || "tep");
      const info = el("div"); info.append(el("b", null, m.attachment.name), el("small", null, SC.fmtSize(m.attachment.size || 0)));
      a.append(el("span", "ficon", "📄"), info); wrap.appendChild(a);
      if (m.body) { const b = el("div", "bubble"); linkify(m.body, b, clickable); b.style.marginTop = "3px"; wrap.appendChild(b); }
      return wrap;
    }
    const b = el("div", "bubble" + (m.warning_level ? " flag-" + m.warning_level : "")); linkify(m.body || "", b, clickable); return b;
  }
  function flagLabel(m, c) {
    const L = m.warning_level; if (!L || m.mine) return null;
    const txt = { 1: "⚠️ Lần 1 · Tin nhắn có dấu hiệu lừa đảo", 2: "🚨 Lần 2 · Nghi ngờ lừa đảo cao", 3: "Lần 3+ · Nguy cơ lừa đảo rất cao" }[L];
    const d = el("div", "flaglabel l" + Math.min(L, 3));
    if (L >= 3) d.appendChild(el("span", "shake", "🚨"));
    d.appendChild(el("span", null, txt));
    return d;
  }
  function statusLine(m) {
    const s = el("div", "status");
    s.textContent = m.read_at ? "Đã xem" : m.delivered_at ? "Đã nhận" : "Đã gửi"; s.title = SC.fmtFull(m.read_at || m.delivered_at || m.created_at);
    return s;
  }
  function nearBottom() { const m = $("msgs"); const v = m.scrollHeight - m.scrollTop - m.clientHeight < 140; nearBottom._stick = v; return v; }
  const scrollBottom = () => { const m = $("msgs"); m.scrollTop = m.scrollHeight; };

  function renderMessages(forceBottom) {
    const c = active(), box = $("msgs"); if (!c) return;
    const stick = forceBottom || nearBottom();
    const arr = state.msgs.get(c.id) || [];
    const frag = document.createDocumentFragment();
    if (state.hasMore.get(c.id)) frag.appendChild(el("div", "day-sep", "Cuộn lên để xem tin cũ hơn…"));
    if (!arr.length) { const d = el("div", "empty"); d.append(el("span", "big", "👋"), document.createTextNode(`Hãy gửi lời chào tới ${c.other.username}!`)); frag.appendChild(d); }
    const lastMine = [...arr].reverse().find((x) => x.mine);
    arr.forEach((m, i) => {
      const prev = arr[i - 1], next = arr[i + 1];
      if (!prev || dayKey(prev.created_at) !== dayKey(m.created_at)) frag.appendChild(el("div", "day-sep", dayLabel(m.created_at)));
      const sameP = prev && prev.sender_id === m.sender_id && dayKey(prev.created_at) === dayKey(m.created_at) && (new Date(m.created_at) - new Date(prev.created_at)) < 300000;
      const sameN = next && next.sender_id === m.sender_id && dayKey(next.created_at) === dayKey(m.created_at) && (new Date(next.created_at) - new Date(m.created_at)) < 300000;
      const row = el("div", "mrow" + (m.mine ? " mine" : "") + (sameP ? "" : " gap")); row.dataset.id = m.id;
      if (!m.mine) { if (!sameN) row.appendChild(SC.avatar(c.other, "sm")); else { const sp = el("span"); sp.style.width = "32px"; sp.style.flex = "none"; row.appendChild(sp); } }
      const stack = el("div", "stack"); stack.title = SC.fmtFull(m.created_at);
      stack.appendChild(bubbleFor(m));
      const fl = flagLabel(m, c); if (fl) stack.appendChild(fl);
      if (!sameN) stack.appendChild(el("div", "tmeta", new Date(m.created_at).toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit" })));
      if (m.mine && lastMine && m.id === lastMine.id) stack.appendChild(statusLine(m));
      row.appendChild(stack); frag.appendChild(row);
    });
    const prevH = box.scrollHeight, prevT = box.scrollTop;
    box.replaceChildren(frag);
    if (stick) scrollBottom(); else if (renderMessages._keep) box.scrollTop = prevT + (box.scrollHeight - prevH);
  }

  function renderComposerState() {
    const c = active(); if (!c) return;
    const blocked = !c.can_send;
    $("composer-inner").classList.toggle("hidden", blocked);
    const bar = $("blocked-bar"); bar.classList.toggle("hidden", !blocked); bar.replaceChildren();
    if (blocked) {
      bar.appendChild(document.createTextNode(c.send_block_reason || "Không thể nhắn tin."));
      if (c.i_blocked) { bar.appendChild(document.createTextNode(" ")); bar.appendChild(btn("Bỏ chặn", "btn-primary", () => unblock(c.other))); }
    }
    updateSend();
  }

  /* ------------------------------------------------------------- soạn tin */
  const input = $("input");
  function autosize() { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 130) + "px"; }
  function updateSend() { $("send-btn").disabled = state.sending || !(input.value.trim() || state.pendingFile) || !(active() && active().can_send); }
  function clearAttachment() { state.pendingFile = null; $("attach-slot").replaceChildren(); $("file-input").value = ""; }
  function setAttachment(f) {
    if (!f) return;
    if (f.size > 10 * 1024 * 1024) { SC.toast("Tệp quá lớn (tối đa 10 MB).", "error"); return; }
    state.pendingFile = f;
    const chip = el("div", "attach-chip");
    if (f.type.startsWith("image/")) { const i = el("img"); i.src = URL.createObjectURL(f); chip.appendChild(i); } else chip.appendChild(el("span", null, "📄"));
    chip.append(el("span", null, `${f.name} (${SC.fmtSize(f.size)})`), btn("✕", "", () => { clearAttachment(); updateSend(); }));
    $("attach-slot").replaceChildren(chip); updateSend();
  }
  input.addEventListener("input", () => {
    autosize(); updateSend();
    const c = active(); if (!c || !socket || !c.can_send) return;
    const now = Date.now();
    if (input.value && now - state.typingSent > 2500) { socket.emit("typing", { conversation_id: c.id, is_typing: true }); state.typingSent = now; }
    clearTimeout(input._t); input._t = setTimeout(() => { socket.emit("typing", { conversation_id: c.id, is_typing: false }); state.typingSent = 0; }, 3000);
  });
  input.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(); } });
  $("send-btn").addEventListener("click", send);
  $("attach-btn").addEventListener("click", () => $("file-input").click());
  $("file-input").addEventListener("change", (e) => setAttachment(e.target.files[0]));
  input.addEventListener("paste", (e) => { const f = [...(e.clipboardData || {}).files || []].find((x) => x.type.startsWith("image/")); if (f) { e.preventDefault(); setAttachment(f); } });
  const panel = $("chat-panel");
  ["dragenter", "dragover"].forEach((ev) => panel.addEventListener(ev, (e) => { if (active() && active().can_send) { e.preventDefault(); panel.classList.add("drag-over"); } }));
  ["dragleave", "drop"].forEach((ev) => panel.addEventListener(ev, (e) => { if (ev === "dragleave" && panel.contains(e.relatedTarget)) return; e.preventDefault(); panel.classList.remove("drag-over"); if (ev === "drop" && active() && active().can_send) setAttachment(e.dataTransfer.files[0]); }));

  // emoji
  const pop = $("emoji-pop");
  EMOJIS.forEach((em) => { const b = el("button", null, em); b.type = "button"; b.addEventListener("click", (e) => { e.stopPropagation(); const s = input.selectionStart ?? input.value.length; input.value = input.value.slice(0, s) + em + input.value.slice(input.selectionEnd ?? s); input.focus(); input.selectionStart = input.selectionEnd = s + em.length; autosize(); updateSend(); }); pop.appendChild(b); });
  $("emoji-btn").addEventListener("click", (e) => { e.stopPropagation(); pop.classList.toggle("hidden"); });
  pop.addEventListener("click", (e) => e.stopPropagation());

  async function send() {
    const c = active(); if (!c || state.sending || !c.can_send) return;
    const body = input.value.trim(), file = state.pendingFile;
    if (!body && !file) return;
    if (body.length > 2000) { SC.toast("Tin nhắn quá dài (tối đa 2000 ký tự).", "error"); return; }
    state.sending = true; updateSend();
    try {
      let d;
      if (file) { const fd = new FormData(); if (body) fd.append("body", body); fd.append("file", file); d = await api("POST", `/api/conversations/${c.id}/messages`, fd); }
      else d = await api("POST", `/api/conversations/${c.id}/messages`, { body });
      input.value = ""; clearAttachment(); autosize();
      socket && socket.emit("typing", { conversation_id: c.id, is_typing: false }); state.typingSent = 0;
      if (state.activeId === c.id) { upsertMessage(d.message); renderMessages(true); }
    } catch (e) {
      if (e.status !== 401) SC.toast(e.message, "error");
      if (e.status === 403) { await refreshActive(); }
    } finally { state.sending = false; updateSend(); if (!matchMedia("(max-width:860px)").matches) input.focus(); }
  }

  async function refreshActive() {
    const c = active(); if (!c) return;
    try { setConv((await api("GET", `/api/conversations/${c.id}`, null, { silent: true })).conversation); renderHeader(); renderBanners(); renderComposerState(); renderList(); } catch (e) { /* bỏ qua */ }
  }

  /* ------------------------------------------------------ cuộn tải tin cũ */
  $("msgs").addEventListener("scroll", async (e) => {
    const box = e.target, c = active();
    nearBottom();
    if (!c || box.scrollTop > 60 || !state.hasMore.get(c.id) || state.loadingOlder) return;
    state.loadingOlder = true;
    try {
      const arr = state.msgs.get(c.id) || [];
      const d = await api("GET", `/api/conversations/${c.id}/messages?before=${arr[0].id}`, null, { silent: true });
      d.messages.forEach(upsertMessage); state.hasMore.set(c.id, d.has_more);
      renderMessages._keep = true; renderMessages(false); renderMessages._keep = false;
    } catch (err) { /* bỏ qua */ } finally { state.loadingOlder = false; }
  });

  function lightbox(url) {
    const lb = el("div", "lightbox"); const i = el("img"); i.src = url; lb.appendChild(i);
    const close = () => { lb.remove(); document.removeEventListener("keydown", onKey); };
    const onKey = (e) => { if (e.key === "Escape") close(); };
    lb.addEventListener("click", close); document.addEventListener("keydown", onKey); document.body.appendChild(lb);
  }

  /* ------------------------------------------------------------ socket */
  function setTitle() { const n = [...state.convs.values()].reduce((s, c) => s + (c.unread || 0), 0); document.title = (n ? `(${n}) ` : "") + "Tin nhắn · SafeChat"; }
  let typingTimer = null;
  if (socket) {
    socket.on("message:new", (p) => {
      const old = setConv(p.conversation); const m = p.message;
      const isActive = state.activeId === m.conversation_id;
      if (isActive) {
        upsertMessage(m);
        if (!m.mine && !document.hidden) { p.conversation.unread = 0; markRead(); }
        const levelUp = old && p.conversation.warning > old.warning;
        renderHeader(); renderBanners(levelUp); renderMessages(m.mine); renderComposerState();
        if (!m.mine) { $("typing").textContent = ""; }
      } else if (!m.mine) {
        SC.toast(`💬 ${p.conversation.other.username}: ${p.conversation.last_message ? p.conversation.last_message.preview.slice(0, 80) : ""}`);
      }
      if (!m.mine && p.conversation.warning > 0 && (!old || p.conversation.warning > old.warning) && !isActive)
        SC.toast(`${p.conversation.warning >= 3 ? "🚨" : "⚠️"} ${p.conversation.other.username} gửi tin có dấu hiệu lừa đảo`, "error");
      if (p.self_warning && isActive) {
        if (p.self_warning_count === 3) SC.dialog({ key: "self-warning", title: "Cảnh báo từ hệ thống", text: p.self_warning, icon: "🚨", danger: true, ok: "Tôi đã hiểu", single: true });
        else { renderBanners(true); SC.toast("🚨 " + p.self_warning, "error"); }          // các lần sau: banner rung + thông báo nhỏ
      }
      renderList(); setTitle();
    });
    socket.on("message:status", (p) => {
      const arr = state.msgs.get(p.conversation_id); if (!arr) return;
      arr.forEach((m) => { if (p.message_ids.includes(m.id)) { if (p.status === "seen") { m.read_at = p.at; m.delivered_at = m.delivered_at || p.at; } else m.delivered_at = m.delivered_at || p.at; } });
      if (state.activeId === p.conversation_id) renderMessages(false);
    });
    socket.on("conversation:read", (p) => { const c = state.convs.get(p.conversation_id); if (c) { c.unread = 0; renderList(); setTitle(); } });
    socket.on("typing", (p) => {
      if (p.conversation_id !== state.activeId) return;
      const t = $("typing"); clearTimeout(typingTimer);
      if (!p.is_typing) { t.textContent = ""; return; }
      const c = active(); t.replaceChildren(document.createTextNode(`${c.other.username} đang nhập`), (() => { const d = el("span", "dots"); d.innerHTML = "<span>.</span><span>.</span><span>.</span>"; return d; })());
      typingTimer = setTimeout(() => (t.textContent = ""), 4500);
      if (nearBottom()) scrollBottom();
    });
    socket.on("presence", (p) => {
      state.convs.forEach((c) => { if (c.other.id === p.user_id) { c.other.online = p.online; c.other.last_seen = p.last_seen || c.other.last_seen; } });
      state.friends.forEach((u) => { if (u.id === p.user_id) u.online = p.online; });
      if (active() && active().other.id === p.user_id) renderHeader();
      renderList();
    });
    socket.on("friend:update", () => afterSocial());
    socket.on("block:update", async () => { await afterSocial(); await refreshActive(); });
    socket.on("connect", async () => { if (state.firstConnect) { state.firstConnect = false; return; } await loadConversations(); const c = active(); if (c) { const h = await api("GET", `/api/conversations/${c.id}/messages`, null, { silent: true }).catch(() => null); if (h) { state.msgs.set(c.id, h.messages); state.hasMore.set(c.id, h.has_more); renderMessages(true); renderBanners(); renderComposerState(); markRead(); } } });
  }
  document.addEventListener("visibilitychange", () => { if (!document.hidden) markRead(); });
  window.addEventListener("focus", markRead);
  setInterval(() => { const c = active(); if (c && !c.other.online && c.other.last_seen) $("head-status").textContent = statusText(c); }, 60000);

  /* ---------------------------------------------------------- khởi động */
  (async function init() {
    await Promise.all([loadConversations(), loadRequests(), loadFriends(), loadBlocked()]);
    setTitle();
    const initial = parseInt(layout.dataset.initialCid, 10);
    if (initial) await openConversation(initial);
  })();
})();
