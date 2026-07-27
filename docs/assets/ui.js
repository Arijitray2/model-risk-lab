/* Shared UI layer: command palette (Ctrl+K or /), cursor spotlight,
   typed terminal headers, reveal-on-scroll. Vanilla JS, no deps. */
"use strict";

/* ---------- cursor spotlight ---------- */
(function () {
  if (matchMedia("(pointer: coarse)").matches) return;
  const s = document.createElement("div");
  s.id = "spotlite";
  document.body.appendChild(s);
  addEventListener("mousemove", (e) => {
    s.style.setProperty("--mx", e.clientX + "px");
    s.style.setProperty("--my", e.clientY + "px");
  }, { passive: true });
})();

/* ---------- magnetic primary buttons ---------- */
(function () {
  if (matchMedia("(pointer: coarse)").matches) return;
  document.querySelectorAll(".btn.primary").forEach((b) => {
    b.addEventListener("mousemove", (e) => {
      const r = b.getBoundingClientRect();
      const dx = e.clientX - (r.left + r.width / 2);
      const dy = e.clientY - (r.top + r.height / 2);
      b.style.transform = `translate(${dx * 0.08}px, ${dy * 0.18}px)`;
    });
    b.addEventListener("mouseleave", () => { b.style.transform = ""; });
  });
})();

/* ---------- typed terminal headers ---------- */
(function () {
  const heads = document.querySelectorAll("h2.term");
  if (!heads.length) return;
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  heads.forEach((h) => {
    const spans = [...h.querySelectorAll("span")].filter(s => !s.classList.contains("prompt"));
    const targets = h.childNodes; // keep prompt; type the rest
    const original = [];
    // collect text nodes + non-prompt spans after the prompt
    h.querySelectorAll(":scope > *:not(.prompt)").forEach(()=>{});
    const parts = [];
    targets.forEach((n) => {
      if (n.nodeType === 3 && n.textContent.trim()) parts.push({ node: n, text: n.textContent });
      if (n.nodeType === 1 && !n.classList.contains("prompt")) parts.push({ node: n.firstChild || n, text: n.textContent, el: n });
    });
    if (reduce || !parts.length) return;
    const io = new IntersectionObserver((es) => {
      es.forEach((e) => {
        if (!e.isIntersecting || h.dataset.typed) return;
        h.dataset.typed = 1;
        const caret = document.createElement("span");
        caret.className = "caret";
        h.appendChild(caret);
        parts.forEach((p) => { if (p.el) p.el.textContent = ""; else p.node.textContent = ""; });
        let pi = 0, ci = 0;
        (function type() {
          if (pi >= parts.length) { setTimeout(() => caret.remove(), 1200); return; }
          const p = parts[pi];
          ci++;
          const slice = p.text.slice(0, ci);
          if (p.el) p.el.textContent = slice; else p.node.textContent = slice;
          if (ci >= p.text.length) { pi++; ci = 0; }
          setTimeout(type, 26);
        })();
        io.unobserve(h);
      });
    }, { threshold: 0.6 });
    io.observe(h);
  });
})();

/* ---------- reveal on scroll ---------- */
(function () {
  const els = document.querySelectorAll(".reveal");
  if (!els.length) return;
  const io = new IntersectionObserver((es) => {
    es.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } });
  }, { threshold: 0.15 });
  els.forEach((el) => io.observe(el));
})();

/* ---------- command palette ---------- */
(function () {
  const ITEMS = [
    { t: "overview — the story & live desk", href: "index.html", k: "go" },
    { t: "methodology — every equation, from scratch", href: "methodology.html", k: "go" },
    { t: "live demo — run the desk yourself", href: "demo.html", k: "go" },
    { t: "results — figures & tables", href: "results.html", k: "go" },
    { t: "report — 24-page PDF", href: "report/model_risk_lab_report.pdf", k: "open" },
    { t: "source — github repository", href: "https://github.com/Arijitray2/model-risk-lab", k: "open" },
    { t: "inject market crash (overview page)", action: "crash", k: "run" },
    { t: "start the 60-second tour", action: "tour", k: "run" },
  ];
  const root = document.createElement("div");
  root.id = "palette";
  root.innerHTML = `<div class="pal-box">
    <input type="text" placeholder="type a command… (crash, demo, results…)" spellcheck="false">
    <div class="pal-list"></div>
    <div class="pal-hint">↑↓ navigate · ↵ run · esc close — open anytime with Ctrl+K or /</div>
  </div>`;
  document.body.appendChild(root);
  const input = root.querySelector("input"), list = root.querySelector(".pal-list");
  let sel = 0, view = ITEMS;

  function render() {
    list.innerHTML = view.map((it, i) =>
      `<div class="pal-item${i === sel ? " sel" : ""}" data-i="${i}">
         <span>${it.t}</span><span class="k">${it.k}</span></div>`).join("") ||
      `<div class="pal-item">no match</div>`;
    list.querySelectorAll(".pal-item[data-i]").forEach((el) => {
      el.addEventListener("mouseenter", () => { sel = +el.dataset.i; render(); });
      el.addEventListener("click", run);
    });
  }
  function open() { root.classList.add("open"); input.value = ""; view = ITEMS; sel = 0; render(); input.focus(); }
  function close() { root.classList.remove("open"); }
  function run() {
    const it = view[sel]; if (!it) return;
    close();
    if (it.action === "crash") {
      if (window.__injectCrash) window.__injectCrash();
      else location.href = "index.html#crash";
    } else if (it.action === "tour") {
      if (window.__startTour) window.__startTour();
      else location.href = "index.html#tour";
    } else location.href = it.href;
  }
  addEventListener("keydown", (e) => {
    const typing = /input|textarea|select/i.test(document.activeElement.tagName);
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") { e.preventDefault(); open(); return; }
    if (e.key === "/" && !typing && !root.classList.contains("open")) { e.preventDefault(); open(); return; }
    if (!root.classList.contains("open")) return;
    if (e.key === "Escape") close();
    else if (e.key === "ArrowDown") { e.preventDefault(); sel = Math.min(sel + 1, view.length - 1); render(); }
    else if (e.key === "ArrowUp") { e.preventDefault(); sel = Math.max(sel - 1, 0); render(); }
    else if (e.key === "Enter") run();
  });
  root.addEventListener("click", (e) => { if (e.target === root) close(); });
  input.addEventListener("input", () => {
    const q = input.value.toLowerCase();
    view = ITEMS.filter((it) => it.t.toLowerCase().includes(q));
    sel = 0; render();
  });
})();

/* ---------- "crash" easter egg (any page; acts on overview) ---------- */
(function () {
  let buf = "";
  addEventListener("keydown", (e) => {
    if (/input|textarea|select/i.test(document.activeElement.tagName)) return;
    if (e.key.length !== 1) return;
    buf = (buf + e.key.toLowerCase()).slice(-5);
    if (buf === "crash") {
      if (window.__injectCrash) window.__injectCrash();
      else location.href = "index.html#crash";
      buf = "";
    }
  });
})();

/* ---------- reading progress bar ---------- */
(function () {
  const bar = document.createElement("div");
  bar.id = "readbar";
  document.body.appendChild(bar);
  addEventListener("scroll", () => {
    const h = document.documentElement;
    const p = h.scrollTop / Math.max(h.scrollHeight - h.clientHeight, 1);
    bar.style.width = (p * 100).toFixed(2) + "%";
  }, { passive: true });
})();

/* ---------- data tape (real calibrated numbers) ---------- */
(function () {
  const nav = document.querySelector("nav");
  if (!nav) return;
  const ITEMS = [
    'SPY 2008–2025 <b>n=4,504d</b>',
    'GBM rejected <b class="b">LR=1627</b> (crit 7.8)',
    'NIFTY 50 <b class="b">LR=1313</b>',
    'physical jumps <b>λ̂=87.9/yr</b> μ̂J=−0.26%',
    'COVID chain 2020-03-16 <b class="r">λ=3.68/yr · μJ=−31%</b>',
    'matched desk edge <b class="g">+0.153</b> [0.146, 0.159]',
    'mispriced desk edge <b class="r">−10.20</b> while books say +0.15',
    'break-even spread <b class="g">8.8¢</b> vs <b class="r">&gt;$15</b>',
    'VaR exceptions <b class="r">811/1500</b> (expected 75)',
    'CUSUM alarm <b class="g">trade 83 of 1,467</b>',
    'smile fit RMSE <b>0.67–1.27 vol pts</b>',
    '15/15 unit tests <b class="g">passing</b>',
  ];
  const half = ITEMS.map(t => `<span>${t}</span>`).join("");
  const tape = document.createElement("div");
  tape.className = "tape";
  tape.innerHTML = `<div class="tape-inner">${half}${half}</div>`;
  nav.insertAdjacentElement("afterend", tape);
})();

/* ---------- methodology TOC + scroll-spy ---------- */
(function () {
  const toc = document.getElementById("toc");
  if (!toc) return;
  const heads = [...document.querySelectorAll(".mcontent h2, .mcontent h3")];
  let html = '<div class="tl">on this page</div>';
  heads.forEach((h, i) => {
    if (!h.id) h.id = "s" + i;
    const txt = h.textContent.replace(/\s+/g, " ").trim();
    html += `<a href="#${h.id}" class="${h.tagName === "H3" ? "h3" : ""}" data-t="${h.id}">${txt}</a>`;
  });
  toc.innerHTML = html;
  const links = new Map([...toc.querySelectorAll("a")].map(a => [a.dataset.t, a]));
  const io = new IntersectionObserver((es) => {
    es.forEach(e => {
      if (e.isIntersecting) {
        links.forEach(a => a.classList.remove("on"));
        const a = links.get(e.target.id);
        if (a) { a.classList.add("on");
          a.scrollIntoView({ block: "nearest" }); }
      }
    });
  }, { rootMargin: "-72px 0px -74% 0px" });
  heads.forEach(h => io.observe(h));
})();
