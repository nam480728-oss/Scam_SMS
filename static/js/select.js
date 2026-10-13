/* Màn hình chọn mô hình AI */
(function () {
  "use strict";
  const page = document.getElementById("select-page");
  if (!page) return;
  const next = page.dataset.next;
  const loading = document.getElementById("loading"), errBox = document.getElementById("select-error");
  const title = document.getElementById("loading-title"), sub = document.getElementById("loading-sub");
  let busy = false;

  // CSRF: trang này có thể chưa có meta (khách chưa đăng nhập) -> lấy từ meta của base.html
  async function post(url) {
    const token = (document.querySelector('meta[name="csrf-token"]') || {}).content || "";
    const res = await fetch(url, { method: "POST", headers: { "X-CSRFToken": token }, credentials: "same-origin" });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || "Lỗi " + res.status);
    return data;
  }

  async function poll(name) {
    for (;;) {
      await new Promise((r) => setTimeout(r, 600));
      let s;
      try { s = await (await fetch("/api/models/status", { credentials: "same-origin" })).json(); }
      catch (e) { continue; }
      if (s.error) throw new Error(s.error);
      if (!s.loading) return s;
      sub.textContent = `Đã chờ ${Math.round(s.elapsed || 0)} giây…`;
    }
  }

  async function choose(card) {
    if (busy || card.classList.contains("disabled")) return;
    busy = true; errBox.classList.add("hidden");
    const key = card.dataset.key, name = card.querySelector("h2").textContent;
    title.textContent = `Đang nạp ${name}…`; sub.textContent = "Lần đầu có thể mất vài giây đến vài chục giây.";
    loading.classList.remove("hidden");
    try {
      let s = await post(`/api/models/select/${key}`);
      if (s.loading || s.active !== key) s = await poll(name);
      if (s.active !== key) throw new Error("Không nạp được mô hình.");
      title.textContent = "Sẵn sàng!"; sub.textContent = "Đang chuyển trang…";
      location.href = next;
    } catch (e) {
      loading.classList.add("hidden"); busy = false;
      errBox.textContent = "❌ " + e.message; errBox.classList.remove("hidden");
    }
  }

  document.querySelectorAll(".model-card").forEach((card) => {
    card.addEventListener("click", () => choose(card));
    card.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); choose(card); } });
  });
})();
