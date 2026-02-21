import { app } from "../../scripts/app.js";

const CSS = `
.qvl-overlay {
  position: fixed; inset: 0; background: rgba(0,0,0,0.75);
  z-index: 99999; display: flex; align-items: center; justify-content: center;
}
.qvl-win {
  background: #1e1e2e; border: 1px solid #555; border-radius: 8px;
  width: 90vw; max-width: 1400px; height: 85vh;
  display: flex; flex-direction: column; color: #ddd;
  font-family: 'Segoe UI', system-ui, sans-serif; font-size: 13px;
  box-shadow: 0 8px 40px rgba(0,0,0,0.6); overflow: hidden;
}

/* Title bar */
.qvl-titlebar {
  display: flex; align-items: center; padding: 6px 12px;
  background: #181828; border-bottom: 1px solid #333; gap: 8px; flex-shrink: 0;
}
.qvl-titlebar-icon { font-size: 16px; }
.qvl-titlebar-text { flex: 1; font-size: 12px; font-weight: 600; color: #ccc; }
.qvl-titlebar-close {
  width: 32px; height: 24px; display: flex; align-items: center; justify-content: center;
  border: none; background: transparent; color: #aaa; font-size: 16px; cursor: pointer;
  border-radius: 4px;
}
.qvl-titlebar-close:hover { background: #c42b1c; color: #fff; }

/* Navigation bar */
.qvl-navbar {
  display: flex; align-items: center; gap: 4px;
  padding: 6px 10px; background: #1a1a2a; border-bottom: 1px solid #2a2a3a;
  flex-shrink: 0;
}
.qvl-nav-btn {
  width: 28px; height: 28px; display: flex; align-items: center; justify-content: center;
  border: 1px solid transparent; border-radius: 4px;
  background: transparent; color: #aaa; cursor: pointer; font-size: 14px;
}
.qvl-nav-btn:hover:not(:disabled) { background: #2a2a4a; border-color: #444; color: #fff; }
.qvl-nav-btn:disabled { opacity: 0.3; cursor: default; }
.qvl-nav-sep { width: 1px; height: 20px; background: #333; margin: 0 2px; }
.qvl-breadcrumb {
  flex: 1; display: flex; align-items: center; gap: 0;
  background: #111; border: 1px solid #333; border-radius: 4px;
  padding: 0 8px; height: 28px; overflow: hidden;
}
.qvl-crumb {
  padding: 2px 6px; cursor: pointer; color: #8bf; white-space: nowrap;
  border-radius: 3px;
}
.qvl-crumb:hover { background: #2a2a4a; }
.qvl-crumb-sep { color: #555; padding: 0 1px; font-size: 11px; }
.qvl-search {
  width: 180px; height: 28px; background: #111; border: 1px solid #333;
  border-radius: 4px; padding: 0 8px; color: #ccc; font-size: 12px;
}
.qvl-search:focus { border-color: #68f; outline: none; }

/* Toolbar */
.qvl-toolbar {
  display: flex; align-items: center; gap: 8px;
  padding: 4px 10px; background: #1c1c2c; border-bottom: 1px solid #2a2a3a;
  flex-shrink: 0;
}
.qvl-tool-label { font-size: 11px; color: #888; }
.qvl-tool-select {
  background: #111; border: 1px solid #333; border-radius: 4px;
  padding: 3px 6px; color: #ccc; font-size: 11px;
}
.qvl-tool-select:focus { border-color: #68f; outline: none; }
.qvl-tool-sep { width: 1px; height: 18px; background: #333; }
.qvl-view-btns { display: flex; gap: 2px; }
.qvl-view-btn {
  width: 26px; height: 24px; display: flex; align-items: center; justify-content: center;
  border: 1px solid transparent; border-radius: 3px;
  background: transparent; color: #888; cursor: pointer; font-size: 12px;
}
.qvl-view-btn:hover { background: #2a2a4a; color: #ccc; }
.qvl-view-btn.active { background: #2a2a5a; border-color: #446; color: #aaf; }
.qvl-page-info { font-size: 11px; color: #777; margin-left: auto; }

/* Main content */
.qvl-main {
  display: flex; flex: 1; overflow: hidden;
}

/* Sidebar */
.qvl-sidebar {
  width: 180px; background: #161626; border-right: 1px solid #2a2a3a;
  overflow-y: auto; flex-shrink: 0; padding: 6px 0;
}
.qvl-side-section { padding: 4px 0; }
.qvl-side-header {
  padding: 4px 12px; font-size: 10px; color: #666;
  text-transform: uppercase; letter-spacing: 0.5px;
}
.qvl-side-item {
  display: flex; align-items: center; gap: 6px;
  padding: 5px 12px; cursor: pointer; color: #bbb; font-size: 12px;
}
.qvl-side-item:hover { background: #222240; color: #fff; }
.qvl-side-item.active { background: #2a2a5a; color: #adf; }
.qvl-side-icon { font-size: 14px; width: 18px; text-align: center; }

/* File grid */
.qvl-files {
  flex: 1; overflow-y: auto; padding: 8px;
}
.qvl-grid-large {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 6px;
}
.qvl-grid-medium {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(100px, 1fr)); gap: 4px;
}
.qvl-grid-small {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(70px, 1fr)); gap: 3px;
}
.qvl-grid-list {
  display: flex; flex-direction: column; gap: 1px;
}
.qvl-item {
  display: flex; flex-direction: column; align-items: center;
  padding: 6px; border-radius: 4px; cursor: pointer;
  border: 2px solid transparent; transition: border-color 0.1s;
}
.qvl-item:hover { background: #222240; border-color: #335; }
.qvl-item.selected { background: #1a2a4a; border-color: #48f; }
.qvl-item-thumb {
  width: 100%; aspect-ratio: 1; object-fit: cover;
  border-radius: 3px; background: #0a0a1a;
}
.qvl-item-name {
  margin-top: 4px; font-size: 11px; color: #bbb; text-align: center;
  width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
/* Folder items */
.qvl-item-folder .qvl-item-thumb {
  object-fit: contain; font-size: 48px; display: flex;
  align-items: center; justify-content: center; background: transparent;
}
/* List view */
.qvl-grid-list .qvl-item {
  flex-direction: row; padding: 3px 8px; gap: 8px;
}
.qvl-grid-list .qvl-item-thumb { width: 24px; height: 24px; aspect-ratio: auto; border-radius: 2px; }
.qvl-grid-list .qvl-item-name { text-align: left; margin: 0; font-size: 12px; }
.qvl-grid-list .qvl-item-size { font-size: 11px; color: #666; margin-left: auto; white-space: nowrap; }

/* Preview panel */
.qvl-preview {
  width: 300px; background: #111; border-left: 1px solid #2a2a3a;
  display: flex; flex-direction: column; align-items: center;
  justify-content: center; padding: 16px; flex-shrink: 0;
}
.qvl-preview img {
  max-width: 100%; max-height: 70%; object-fit: contain; border-radius: 4px;
}
.qvl-preview-info {
  margin-top: 12px; text-align: center; font-size: 11px; color: #888;
  word-break: break-all;
}
.qvl-preview-info strong { color: #ccc; display: block; margin-bottom: 4px; font-size: 12px; }

/* Footer */
.qvl-footer {
  display: flex; align-items: center; gap: 8px;
  padding: 8px 12px; background: #181828; border-top: 1px solid #333;
  flex-shrink: 0;
}
.qvl-filename {
  flex: 1; height: 28px; background: #111; border: 1px solid #333;
  border-radius: 4px; padding: 0 10px; color: #ccc; font-size: 12px;
}
.qvl-filename:focus { border-color: #68f; outline: none; }
.qvl-footer-btn {
  padding: 6px 20px; border-radius: 4px; font-size: 12px;
  cursor: pointer; border: 1px solid #444;
}
.qvl-btn-open { background: #2a5a2a; color: #cfc; border-color: #4a4; }
.qvl-btn-open:hover { background: #3a7a3a; }
.qvl-btn-cancel { background: #333; color: #ccc; }
.qvl-btn-cancel:hover { background: #444; }
.qvl-empty { padding: 40px; text-align: center; color: #555; }
`;

class ServerBrowser {
  constructor(callback) {
    this.callback = callback;
    this.currentPath = "/workspace";
    this.selected = null;
    this.selectedName = "";
    this.files = [];
    this.dirs = [];
    this.allFiles = [];
    this.page = 0;
    this.perPage = 100;
    this.sortBy = "name";
    this.sortAsc = true;
    this.viewMode = "large"; // large, medium, small, list
    this.searchQuery = "";
    this.history = [];
    this.historyIdx = -1;
    this.quickLinks = [
      { name: "workspace", path: "/workspace", icon: "🏠" },
      { name: "scored_keep", path: "/workspace/scored_keep", icon: "⭐" },
      { name: "dataset", path: "/workspace/dataset", icon: "📦" },
      { name: "all_frames", path: "/workspace/all_frames", icon: "🎞" },
      { name: "videos", path: "/workspace/videos", icon: "🎬" },
      { name: "ComfyUI input", path: "/workspace/ComfyUI/input", icon: "📁" },
      { name: "ComfyUI output", path: "/workspace/ComfyUI/output", icon: "📁" },
    ];
    this._onMouse = this._mouseNav.bind(this);
    this._onMouseDown = (e) => { if (e.button === 2 || e.button === 3 || e.button === 4) { e.preventDefault(); e.stopPropagation(); } };
    this._onContextMenu = this._handleContextMenu.bind(this);
    this._injectCSS();
    this._build();
    this.navigate(this.currentPath);
  }

  _injectCSS() {
    if (!document.getElementById("qvl-css")) {
      const s = document.createElement("style");
      s.id = "qvl-css";
      s.textContent = CSS;
      document.head.appendChild(s);
    }
  }

  _mouseNav(e) {
    if (e.button === 2) {
      e.preventDefault();
      e.stopPropagation();
      // Right-click on empty space = go back (Windows Explorer behavior)
      const target = e.target;
      const isFileArea = target.classList.contains('qvl-files') || target.classList.contains('qvl-grid-large') ||
                         target.classList.contains('qvl-grid-medium') || target.classList.contains('qvl-grid-small') ||
                         target.classList.contains('qvl-grid-list');
      if (isFileArea) {
        this.goBack();
      }
    }
    else if (e.button === 3) { e.preventDefault(); e.stopPropagation(); this.goBack(); }
    else if (e.button === 4) { e.preventDefault(); e.stopPropagation(); this.goForward(); }
  }

  _handleContextMenu(e) {
    // Prevent default context menu within the modal
    const isInModal = e.target.closest('.qvl-win');
    if (isInModal) {
      e.preventDefault();
      e.stopPropagation();
      return false;
    }
  }

  _pushHistory(p) {
    this.history = this.history.slice(0, this.historyIdx + 1);
    this.history.push(p);
    this.historyIdx = this.history.length - 1;
  }

  goBack() { if (this.historyIdx > 0) { this.historyIdx--; this._load(this.history[this.historyIdx]); } }
  goForward() { if (this.historyIdx < this.history.length - 1) { this.historyIdx++; this._load(this.history[this.historyIdx]); } }
  goUp() { const p = this.currentPath.replace(/\/[^/]+$/, "") || "/"; this.navigate(p); }

  navigate(path) {
    this._pushHistory(path);
    this._load(path);
  }

  async _load(path) {
    this.currentPath = path;
    this.selected = null;
    this.selectedName = "";
    this.page = 0;
    this.filesEl.innerHTML = '<div class="qvl-empty">Loading...</div>';
    this._updatePreview();
    this._updateNav();
    this._updateSidebar();

    try {
      const r = await fetch(`/qvl/browse?path=${encodeURIComponent(path)}`);
      const d = await r.json();
      if (d.error) { this.filesEl.innerHTML = `<div class="qvl-empty" style="color:#f88">${d.error}</div>`; return; }
      this.dirs = d.dirs || [];
      this.allFiles = d.files || [];
      this._applySort();
      this._applySearch();
      this._renderBreadcrumb();
      this._updateNav();
      this._renderFiles();
    } catch (err) {
      this.filesEl.innerHTML = `<div class="qvl-empty" style="color:#f88">${err.message}</div>`;
    }
  }

  _applySort() {
    const s = this.sortBy;
    const m = this.sortAsc ? 1 : -1;
    this.allFiles.sort((a, b) => {
      if (s === "name") return a.name.localeCompare(b.name) * m;
      if (s === "size") return (a.size - b.size) * m;
      return 0;
    });
    this.dirs.sort((a, b) => a.name.localeCompare(b.name));
  }

  _applySearch() {
    if (!this.searchQuery) { this.files = this.allFiles; return; }
    const q = this.searchQuery.toLowerCase();
    this.files = this.allFiles.filter(f => f.name.toLowerCase().includes(q));
  }

  _renderBreadcrumb() {
    const parts = this.currentPath.split("/").filter(Boolean);
    this.breadcrumbEl.innerHTML = "";
    let built = "";
    // Root
    const root = document.createElement("span");
    root.className = "qvl-crumb";
    root.textContent = "/";
    root.onclick = () => this.navigate("/");
    this.breadcrumbEl.appendChild(root);

    for (let i = 0; i < parts.length; i++) {
      built += "/" + parts[i];
      const sep = document.createElement("span");
      sep.className = "qvl-crumb-sep";
      sep.textContent = "›";
      this.breadcrumbEl.appendChild(sep);

      const crumb = document.createElement("span");
      crumb.className = "qvl-crumb";
      crumb.textContent = parts[i];
      const target = built;
      crumb.onclick = () => this.navigate(target);
      this.breadcrumbEl.appendChild(crumb);
    }
  }

  _updateNav() {
    this.backBtn.disabled = this.historyIdx <= 0;
    this.fwdBtn.disabled = this.historyIdx >= this.history.length - 1;
    this.upBtn.disabled = this.currentPath === "/";
  }

  _updateSidebar() {
    this.sidebarEl.querySelectorAll(".qvl-side-item").forEach(el => {
      el.classList.toggle("active", el.dataset.path === this.currentPath);
    });
  }

  _updatePreview() {
    if (!this.selected) {
      this.previewEl.innerHTML = '<div style="color:#444; font-size:13px">No image selected</div>';
      this.filenameInput.value = "";
      return;
    }
    const name = this.selected.split("/").pop();
    this.filenameInput.value = name;
    const imgUrl = `/qvl/image?path=${encodeURIComponent(this.selected)}`;
    this.previewEl.innerHTML = `
      <img src="${imgUrl}" onerror="this.style.display='none'">
      <div class="qvl-preview-info">
        <strong>${name}</strong>
        ${this.currentPath}
      </div>`;
  }

  _renderFiles() {
    const viewClass = `qvl-grid-${this.viewMode}`;
    const start = this.page * this.perPage;
    const pageFiles = this.files.slice(start, start + this.perPage);
    const totalPages = Math.ceil(this.files.length / this.perPage);

    // Update page info
    this.pageInfo.textContent = this.files.length > this.perPage
      ? `Page ${this.page + 1}/${totalPages} · ${this.files.length} images`
      : `${this.files.length} images`;

    let html = `<div class="${viewClass}">`;

    // Folders first
    for (const d of this.dirs) {
      const fp = this.currentPath + "/" + d.name;
      html += `<div class="qvl-item qvl-item-folder" data-dir="${fp}">
        <div class="qvl-item-thumb">📁</div>
        <div class="qvl-item-name">${d.name}</div>
      </div>`;
    }

    // Images
    for (const f of pageFiles) {
      const fp = this.currentPath + "/" + f.name;
      const sizeKB = Math.round(f.size / 1024);
      html += `<div class="qvl-item" data-file="${fp}">
        <img class="qvl-item-thumb" loading="lazy" src="/qvl/thumbnail?path=${encodeURIComponent(fp)}" alt="${f.name}">
        <div class="qvl-item-name" title="${f.name}">${f.name}</div>
        ${this.viewMode === "list" ? `<div class="qvl-item-size">${sizeKB} KB</div>` : ""}
      </div>`;
    }

    if (this.dirs.length === 0 && pageFiles.length === 0) {
      html += '<div class="qvl-empty">No images found</div>';
    }

    html += "</div>";
    this.filesEl.innerHTML = html;

    // Events
    this.filesEl.querySelectorAll("[data-dir]").forEach(el => {
      el.addEventListener("dblclick", () => this.navigate(el.dataset.dir));
      el.addEventListener("click", () => {
        this.filesEl.querySelectorAll(".selected").forEach(e => e.classList.remove("selected"));
        el.classList.add("selected");
      });
      // Right-click on folder = select it
      el.addEventListener("mouseup", (e) => {
        if (e.button === 2) {
          e.preventDefault();
          e.stopPropagation();
          this.filesEl.querySelectorAll(".selected").forEach(el => el.classList.remove("selected"));
          el.classList.add("selected");
        }
      });
    });
    this.filesEl.querySelectorAll("[data-file]").forEach(el => {
      el.addEventListener("click", () => {
        this.filesEl.querySelectorAll(".selected").forEach(e => e.classList.remove("selected"));
        el.classList.add("selected");
        this.selected = el.dataset.file;
        this.selectedName = el.dataset.file.split("/").pop();
        this._updatePreview();
      });
      el.addEventListener("dblclick", () => {
        this.selected = el.dataset.file;
        this._confirm();
      });
      // Right-click on file = select it and update preview
      el.addEventListener("mouseup", (e) => {
        if (e.button === 2) {
          e.preventDefault();
          e.stopPropagation();
          this.filesEl.querySelectorAll(".selected").forEach(el => el.classList.remove("selected"));
          el.classList.add("selected");
          this.selected = el.dataset.file;
          this.selectedName = el.dataset.file.split("/").pop();
          this._updatePreview();
        }
      });
    });
  }

  _confirm() {
    if (this.selected) {
      this.callback(this.selected);
      this.close();
    }
  }

  _build() {
    // Overlay
    this.overlay = document.createElement("div");
    this.overlay.className = "qvl-overlay";
    this.overlay.addEventListener("click", e => { if (e.target === this.overlay) this.close(); });

    const win = document.createElement("div");
    win.className = "qvl-win";

    // Title bar
    const titlebar = document.createElement("div");
    titlebar.className = "qvl-titlebar";
    titlebar.innerHTML = `<span class="qvl-titlebar-icon">📂</span><span class="qvl-titlebar-text">Open — Server File Browser</span>`;
    const closeBtn = document.createElement("button");
    closeBtn.className = "qvl-titlebar-close";
    closeBtn.textContent = "✕";
    closeBtn.onclick = () => this.close();
    titlebar.appendChild(closeBtn);

    // Navbar
    const navbar = document.createElement("div");
    navbar.className = "qvl-navbar";

    this.backBtn = this._navBtn("←", "Back", () => this.goBack());
    this.fwdBtn = this._navBtn("→", "Forward", () => this.goForward());
    this.upBtn = this._navBtn("↑", "Up one level", () => this.goUp());
    navbar.append(this.backBtn, this.fwdBtn, this._sep(), this.upBtn, this._sep());

    this.breadcrumbEl = document.createElement("div");
    this.breadcrumbEl.className = "qvl-breadcrumb";
    navbar.appendChild(this.breadcrumbEl);

    const search = document.createElement("input");
    search.className = "qvl-search";
    search.placeholder = "🔍 Search...";
    search.addEventListener("input", () => {
      this.searchQuery = search.value;
      this.page = 0;
      this._applySearch();
      this._renderFiles();
    });
    navbar.appendChild(search);

    // Toolbar
    const toolbar = document.createElement("div");
    toolbar.className = "qvl-toolbar";

    toolbar.appendChild(this._label("Sort:"));
    const sortSel = document.createElement("select");
    sortSel.className = "qvl-tool-select";
    sortSel.innerHTML = `<option value="name">Name</option><option value="size">Size</option>`;
    sortSel.onchange = () => { this.sortBy = sortSel.value; this._applySort(); this._applySearch(); this._renderFiles(); };
    toolbar.appendChild(sortSel);

    const ascBtn = document.createElement("button");
    ascBtn.className = "qvl-view-btn";
    ascBtn.textContent = "↕";
    ascBtn.title = "Toggle sort direction";
    ascBtn.onclick = () => { this.sortAsc = !this.sortAsc; this._applySort(); this._applySearch(); this._renderFiles(); };
    toolbar.appendChild(ascBtn);

    toolbar.appendChild(this._toolSep());
    toolbar.appendChild(this._label("View:"));

    const views = [
      { mode: "large", icon: "▦", title: "Large icons" },
      { mode: "medium", icon: "▤", title: "Medium icons" },
      { mode: "small", icon: "▪", title: "Small icons" },
      { mode: "list", icon: "☰", title: "List" },
    ];
    const viewBtns = document.createElement("div");
    viewBtns.className = "qvl-view-btns";
    for (const v of views) {
      const b = document.createElement("button");
      b.className = `qvl-view-btn ${v.mode === this.viewMode ? "active" : ""}`;
      b.textContent = v.icon;
      b.title = v.title;
      b.onclick = () => {
        this.viewMode = v.mode;
        viewBtns.querySelectorAll(".qvl-view-btn").forEach(x => x.classList.remove("active"));
        b.classList.add("active");
        this._renderFiles();
      };
      viewBtns.appendChild(b);
    }
    toolbar.appendChild(viewBtns);

    toolbar.appendChild(this._toolSep());

    // Pagination
    const prevBtn = document.createElement("button");
    prevBtn.className = "qvl-view-btn";
    prevBtn.textContent = "◄";
    prevBtn.title = "Previous page";
    prevBtn.onclick = () => { if (this.page > 0) { this.page--; this._renderFiles(); } };
    toolbar.appendChild(prevBtn);

    const nextBtn = document.createElement("button");
    nextBtn.className = "qvl-view-btn";
    nextBtn.textContent = "►";
    nextBtn.title = "Next page";
    nextBtn.onclick = () => {
      const maxPage = Math.ceil(this.files.length / this.perPage) - 1;
      if (this.page < maxPage) { this.page++; this._renderFiles(); }
    };
    toolbar.appendChild(nextBtn);

    this.pageInfo = document.createElement("span");
    this.pageInfo.className = "qvl-page-info";
    toolbar.appendChild(this.pageInfo);

    // Main content
    const main = document.createElement("div");
    main.className = "qvl-main";

    // Sidebar
    this.sidebarEl = document.createElement("div");
    this.sidebarEl.className = "qvl-sidebar";
    this._renderSidebar();

    // Files area
    this.filesEl = document.createElement("div");
    this.filesEl.className = "qvl-files";

    // Preview
    this.previewEl = document.createElement("div");
    this.previewEl.className = "qvl-preview";
    this.previewEl.innerHTML = '<div style="color:#444">No image selected</div>';

    main.append(this.sidebarEl, this.filesEl, this.previewEl);

    // Footer
    const footer = document.createElement("div");
    footer.className = "qvl-footer";

    const fnLabel = this._label("File name:");
    fnLabel.style.color = "#999";
    footer.appendChild(fnLabel);

    this.filenameInput = document.createElement("input");
    this.filenameInput.className = "qvl-filename";
    this.filenameInput.readOnly = true;
    footer.appendChild(this.filenameInput);

    const openBtn = document.createElement("button");
    openBtn.className = "qvl-footer-btn qvl-btn-open";
    openBtn.textContent = "Open";
    openBtn.onclick = () => this._confirm();
    footer.appendChild(openBtn);

    const cancelBtn = document.createElement("button");
    cancelBtn.className = "qvl-footer-btn qvl-btn-cancel";
    cancelBtn.textContent = "Cancel";
    cancelBtn.onclick = () => this.close();
    footer.appendChild(cancelBtn);

    // Assemble
    win.append(titlebar, navbar, toolbar, main, footer);
    this.overlay.appendChild(win);
    document.body.appendChild(this.overlay);

    // Mouse nav
    document.addEventListener("mouseup", this._onMouse, true);
    document.addEventListener("mousedown", this._onMouseDown, true);
    document.addEventListener("contextmenu", this._onContextMenu, true);

    // Keyboard
    this._onKey = (e) => {
      if (e.key === "Escape") this.close();
      if (e.key === "Enter" && this.selected) this._confirm();
    };
    document.addEventListener("keydown", this._onKey);
  }

  _renderSidebar() {
    let html = '<div class="qvl-side-section"><div class="qvl-side-header">Quick Access</div>';
    for (const q of this.quickLinks) {
      html += `<div class="qvl-side-item" data-path="${q.path}">
        <span class="qvl-side-icon">${q.icon}</span>${q.name}
      </div>`;
    }
    html += "</div>";
    this.sidebarEl.innerHTML = html;
    this.sidebarEl.querySelectorAll(".qvl-side-item").forEach(el => {
      el.addEventListener("click", () => this.navigate(el.dataset.path));
    });
  }

  _navBtn(text, title, onclick) {
    const b = document.createElement("button");
    b.className = "qvl-nav-btn";
    b.textContent = text;
    b.title = title;
    b.onclick = onclick;
    return b;
  }

  _sep() { const d = document.createElement("div"); d.className = "qvl-nav-sep"; return d; }
  _toolSep() { const d = document.createElement("div"); d.className = "qvl-tool-sep"; return d; }
  _label(t) { const s = document.createElement("span"); s.className = "qvl-tool-label"; s.textContent = t; return s; }

  close() {
    document.removeEventListener("mouseup", this._onMouse, true);
    document.removeEventListener("mousedown", this._onMouseDown, true);
    document.removeEventListener("contextmenu", this._onContextMenu, true);
    document.removeEventListener("keydown", this._onKey);
    if (this.overlay) { this.overlay.remove(); this.overlay = null; }
  }
}

// Show preview image on a node (like LoadImage does)
function showPreviewOnNode(node, imagePath) {
  if (!imagePath || imagePath.endsWith("/")) return;
  const img = new Image();
  img.src = `/qvl/image?path=${encodeURIComponent(imagePath)}`;
  img.onload = () => {
    node.imgs = [img];
    node.imageIndex = 0;
    // Resize node to fit preview
    const aspect = img.naturalHeight / img.naturalWidth;
    const previewW = Math.max(node.size[0], 360);
    const previewH = previewW * aspect;
    const widgetH = (node.widgets?.length || 0) * 28 + 10;
    node.size[0] = previewW;
    node.size[1] = widgetH + previewH;
    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
  };
}

// Register extension
app.registerExtension({
  name: "QVL.ServerBrowser",
  async nodeCreated(node) {
    if (node.comfyClass !== "QVL_ServerImage") return;
    const pw = node.widgets?.find(w => w.name === "image_path");
    if (!pw) return;

    // Browse button
    node.addWidget("button", "browse_server", "Browse Server", () => {
      new ServerBrowser((path) => {
        pw.value = path;
        if (pw.callback) pw.callback(path);
        showPreviewOnNode(node, path);
      });
    });

    // Show preview for current value on load
    setTimeout(() => showPreviewOnNode(node, pw.value), 500);

    // Watch for manual path changes
    const origCallback = pw.callback;
    pw.callback = (value) => {
      if (origCallback) origCallback(value);
      showPreviewOnNode(node, value);
    };

    node.setSize([360, 300]);
  },
});
