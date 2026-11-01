/**
 * ARGUS Phase 2 — Watchlist Management UI Controller
 * Handles: login/auth, watchlist CRUD, face analysis trigger,
 *          candidate match review queue, reference/crop display.
 *
 * Design constraints:
 *  - No fake/mock data. All values come from authenticated API.
 *  - Reference images only loaded for admin/reviewer after auth.
 *  - Mandatory advisory banner on every match display.
 *  - Embeddings are never displayed in the UI.
 */

(function () {
  "use strict";

  // ── State ────────────────────────────────────────────────────────────────
  const WL = {
    token: null,
    role: null,
    userId: null,
    username: null,
  };

  function persistAuth(data) {
    WL.token = data.access_token;
    WL.role = data.role;
    WL.userId = data.user_id;
    WL.username = data.username;
    try {
      sessionStorage.setItem("argus_wl_token", data.access_token);
      sessionStorage.setItem("argus_wl_role", data.role);
      sessionStorage.setItem("argus_wl_user", data.username);
    } catch (_) {}
    updateAuthUI();
  }

  function restoreAuth() {
    try {
      const tok = sessionStorage.getItem("argus_wl_token");
      if (tok) {
        WL.token = tok;
        WL.role = sessionStorage.getItem("argus_wl_role") || "";
        WL.username = sessionStorage.getItem("argus_wl_user") || "";
        updateAuthUI();
      }
    } catch (_) {}
  }

  function clearAuth() {
    WL.token = null;
    WL.role = null;
    WL.userId = null;
    WL.username = null;
    try {
      sessionStorage.removeItem("argus_wl_token");
      sessionStorage.removeItem("argus_wl_role");
      sessionStorage.removeItem("argus_wl_user");
    } catch (_) {}
    updateAuthUI();
  }

  function authHeaders() {
    return WL.token
      ? { Authorization: `Bearer ${WL.token}`, "Content-Type": "application/json" }
      : { "Content-Type": "application/json" };
  }

  function canDo(permission) {
    const perms = {
      admin: {
        can_enroll: true, can_revoke: true, can_delete: true,
        can_review: true, can_manage_users: true, can_view_audit: true,
        can_trigger_analysis: true, can_view_references: true,
      },
      reviewer: {
        can_enroll: false, can_revoke: false, can_delete: false,
        can_review: true, can_manage_users: false, can_view_audit: false,
        can_trigger_analysis: false, can_view_references: true,
      },
      operator: {
        can_enroll: false, can_revoke: false, can_delete: false,
        can_review: false, can_manage_users: false, can_view_audit: false,
        can_trigger_analysis: true, can_view_references: false,
      },
    };
    return !!(perms[WL.role] || {})[permission];
  }

  // ── API helpers ──────────────────────────────────────────────────────────
  async function apiGet(path) {
    const resp = await fetch(path, { headers: authHeaders() });
    if (resp.status === 401) { clearAuth(); throw new Error("Session expired. Please log in."); }
    if (!resp.ok) {
      const j = await resp.json().catch(() => ({}));
      throw new Error(j.detail || `HTTP ${resp.status}`);
    }
    return resp.json();
  }

  async function apiPost(path, body, isFormData = false) {
    const opts = { method: "POST" };
    if (isFormData) {
      opts.body = body;
      opts.headers = WL.token ? { Authorization: `Bearer ${WL.token}` } : {};
    } else {
      opts.body = JSON.stringify(body);
      opts.headers = authHeaders();
    }
    const resp = await fetch(path, opts);
    if (resp.status === 401) { clearAuth(); throw new Error("Session expired. Please log in."); }
    if (!resp.ok) {
      const j = await resp.json().catch(() => ({}));
      throw new Error(j.detail || `HTTP ${resp.status}`);
    }
    return resp.json();
  }

  async function apiDelete(path) {
    const resp = await fetch(path, { method: "DELETE", headers: authHeaders() });
    if (!resp.ok) {
      const j = await resp.json().catch(() => ({}));
      throw new Error(j.detail || `HTTP ${resp.status}`);
    }
    return resp.json();
  }

  async function apiPut(path, body) {
    const resp = await fetch(path, {
      method: "PUT",
      headers: authHeaders(),
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      const j = await resp.json().catch(() => ({}));
      throw new Error(j.detail || `HTTP ${resp.status}`);
    }
    return resp.json();
  }

  // ── Panel bootstrap ──────────────────────────────────────────────────────
  function updateAuthUI() {
    const loginSection = document.getElementById("wlLoginSection");
    const mainSection = document.getElementById("wlMainSection");
    const wlUserInfo = document.getElementById("wlUserInfo");
    const wlRoleBadge = document.getElementById("wlRoleBadge");

    if (!loginSection || !mainSection) return;

    if (WL.token) {
      loginSection.style.display = "none";
      mainSection.style.display = "block";
      if (wlUserInfo) wlUserInfo.textContent = WL.username || "";
      if (wlRoleBadge) {
        wlRoleBadge.textContent = (WL.role || "").toUpperCase();
        wlRoleBadge.className = `wl-role-badge role-${WL.role}`;
      }
      // Show/hide role-dependent controls
      document.querySelectorAll("[data-require-perm]").forEach((el) => {
        const perm = el.dataset.requirePerm;
        el.style.display = canDo(perm) ? "" : "none";
      });
      document.querySelectorAll("[data-require-role]").forEach((el) => {
        const roles = el.dataset.requireRole.split(",").map((r) => r.trim());
        el.style.display = roles.includes(WL.role) ? "" : "none";
      });
    } else {
      loginSection.style.display = "block";
      mainSection.style.display = "none";
    }
  }

  // ── Login ────────────────────────────────────────────────────────────────
  function initLogin() {
    const form = document.getElementById("wlLoginForm");
    const errEl = document.getElementById("wlLoginError");
    if (!form) return;
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      errEl.textContent = "";
      const username = form.querySelector("#wlLoginUsername").value.trim();
      const password = form.querySelector("#wlLoginPassword").value;
      try {
        const data = await apiPost("/api/auth/login", { username, password });
        persistAuth(data);
        loadWatchlistTab("wlTabEnroll");
      } catch (err) {
        errEl.textContent = err.message;
      }
    });

    const logoutBtn = document.getElementById("wlLogoutBtn");
    if (logoutBtn) {
      logoutBtn.addEventListener("click", () => {
        clearAuth();
        loadWatchlistTab("wlTabEnroll");
      });
    }
  }

  // ── Tab navigation ───────────────────────────────────────────────────────
  function loadWatchlistTab(tabId) {
    document.querySelectorAll(".wl-tab-content").forEach((el) => {
      el.style.display = "none";
    });
    document.querySelectorAll(".wl-tab-btn").forEach((btn) => {
      btn.classList.remove("active");
    });
    const panel = document.getElementById(tabId + "Content");
    const btn = document.getElementById(tabId + "Btn");
    if (panel) panel.style.display = "block";
    if (btn) btn.classList.add("active");

    // Lazy load data for tabs
    if (tabId === "wlTabList" && WL.token) loadWatchlistEntries();
    if (tabId === "wlTabMatches" && WL.token) loadMatchObservations();
    if (tabId === "wlTabAudit" && WL.token) loadAuditLog();
  }

  function initTabs() {
    document.querySelectorAll(".wl-tab-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        if (!WL.token) { showWlError("Please log in first."); return; }
        loadWatchlistTab(btn.dataset.tab);
      });
    });
  }

  // ── Enrollment form ──────────────────────────────────────────────────────
  function initEnrollmentForm() {
    const form = document.getElementById("wlEnrollForm");
    const feedbackEl = document.getElementById("wlEnrollFeedback");
    if (!form) return;

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      if (!canDo("can_enroll")) {
        feedbackEl.innerHTML = `<span class="wl-error">Your role does not permit enrollment.</span>`;
        return;
      }
      feedbackEl.innerHTML = `<span class="wl-info">Validating reference image...</span>`;
      const fd = new FormData(form);
      try {
        const result = await apiPost("/api/watchlist/enroll", fd, true);
        feedbackEl.innerHTML = `
          <span class="wl-success">
            ✓ Enrolled: <strong>${escHtml(result.display_name)}</strong>
            (ID: ${escHtml(result.entry_id)}).
            Status: <strong>PENDING</strong> — an admin must activate this entry
            before it is used in face matching.
            Face quality: ${escHtml(result.face_quality)} (score: ${result.face_quality_score}).
          </span>`;
        form.reset();
      } catch (err) {
        feedbackEl.innerHTML = `<span class="wl-error">Enrollment failed: ${escHtml(err.message)}</span>`;
      }
    });
  }

  // ── Watchlist entries list ───────────────────────────────────────────────
  async function loadWatchlistEntries() {
    const container = document.getElementById("wlEntriesContainer");
    if (!container) return;
    container.innerHTML = `<p class="wl-loading">Loading watchlist...</p>`;
    try {
      const data = await apiGet("/api/watchlist");
      renderWatchlistEntries(data.entries || [], container);
    } catch (err) {
      container.innerHTML = `<p class="wl-error">${escHtml(err.message)}</p>`;
    }
  }

  function renderWatchlistEntries(entries, container) {
    if (!entries.length) {
      container.innerHTML = `<p class="wl-empty">No watchlist entries found.</p>`;
      return;
    }
    const rows = entries.map((e) => {
      const statusClass = `wl-status-${(e.record_status || "unknown").toLowerCase()}`;
      const expiry = e.expiry_date
        ? new Date(e.expiry_date * 1000).toLocaleDateString()
        : "None";
      return `
        <tr data-entry-id="${escHtml(e.entry_id)}">
          <td class="wl-mono">${escHtml(e.entry_id.slice(0, 12))}…</td>
          <td>${escHtml(e.display_name)}</td>
          <td><span class="wl-status-badge ${statusClass}">${escHtml(e.record_status)}</span></td>
          <td>${escHtml(e.authorization_status || "—")}</td>
          <td>${new Date(e.enrollment_date * 1000).toLocaleDateString()}</td>
          <td>${expiry}</td>
          <td class="wl-actions-cell">
            ${canDo("can_enroll") && e.record_status === "pending"
              ? `<button class="wl-btn-sm wl-btn-activate" onclick="ARGUS_WL.activateEntry('${escHtml(e.entry_id)}')">Activate</button>`
              : ""}
            ${canDo("can_revoke") && !["revoked","deleted"].includes(e.record_status)
              ? `<button class="wl-btn-sm wl-btn-revoke" onclick="ARGUS_WL.revokeEntry('${escHtml(e.entry_id)}')">Revoke</button>`
              : ""}
            ${canDo("can_delete") && e.record_status !== "deleted"
              ? `<button class="wl-btn-sm wl-btn-delete" onclick="ARGUS_WL.deleteEntry('${escHtml(e.entry_id)}')">Delete Data</button>`
              : ""}
          </td>
        </tr>`;
    });
    container.innerHTML = `
      <table class="wl-table">
        <thead>
          <tr>
            <th>ID</th><th>Name</th><th>Status</th><th>Auth</th>
            <th>Enrolled</th><th>Expires</th><th>Actions</th>
          </tr>
        </thead>
        <tbody>${rows.join("")}</tbody>
      </table>`;
  }

  // ── Face analysis trigger ────────────────────────────────────────────────
  function initAnalysisTrigger() {
    const form = document.getElementById("wlAnalyzeForm");
    const feedbackEl = document.getElementById("wlAnalyzeFeedback");
    if (!form) return;

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      if (!canDo("can_trigger_analysis")) {
        feedbackEl.innerHTML = `<span class="wl-error">Not authorized to trigger analysis.</span>`;
        return;
      }
      const videoSource = form.querySelector("#wlAnalyzeSource").value.trim();
      if (!videoSource) {
        feedbackEl.innerHTML = `<span class="wl-error">Please enter a video filename.</span>`;
        return;
      }
      feedbackEl.innerHTML = `<span class="wl-info">Triggering face-watchlist analysis...</span>`;
      try {
        const result = await apiPost("/api/watchlist/analyze", { video_source: videoSource });
        feedbackEl.innerHTML = `
          <div class="wl-advisory-banner">
            ⚠ ${escHtml(result.advisory)}
          </div>
          <span class="wl-success">
            Analysis job started: <strong>${escHtml(result.face_job_id)}</strong>
            for video <strong>${escHtml(result.video_source)}</strong>.
            Check the Matches tab for results.
          </span>`;
        pollFaceJob(result.face_job_id, feedbackEl);
      } catch (err) {
        feedbackEl.innerHTML = `<span class="wl-error">Failed: ${escHtml(err.message)}</span>`;
      }
    });
  }

  async function pollFaceJob(jobId, feedbackEl) {
    let attempts = 0;
    const maxAttempts = 120;
    const interval = setInterval(async () => {
      attempts++;
      if (attempts > maxAttempts) {
        clearInterval(interval);
        return;
      }
      try {
        const job = await apiGet(`/api/watchlist/jobs/${jobId}`);
        if (job.status === "done" && job.result) {
          clearInterval(interval);
          const r = job.result;
          feedbackEl.innerHTML += `<br><span class="wl-success">
            ✓ Complete — Frames sampled: ${r.frames_sampled},
            Faces detected: ${r.faces_detected},
            Observations saved: ${r.observations_saved},
            Possible matches: <strong>${r.possible_matches}</strong>.
            Processing time: ${r.processing_time_s}s.
          </span>
          <div class="wl-advisory-banner">⚠ ${escHtml(r.advisory)}</div>`;
        } else if (job.status === "error") {
          clearInterval(interval);
          feedbackEl.innerHTML += `<br><span class="wl-error">Analysis failed: ${escHtml(job.error)}</span>`;
        } else {
          feedbackEl.querySelector(".wl-info") &&
            (feedbackEl.querySelector(".wl-info").textContent =
              `Running… ${job.stage || ""} ${job.progress ? job.progress + "%" : ""}`);
        }
      } catch (_) {}
    }, 3000);
  }

  // ── Match observations list ──────────────────────────────────────────────
  async function loadMatchObservations() {
    const container = document.getElementById("wlMatchesContainer");
    if (!container) return;
    container.innerHTML = `<p class="wl-loading">Loading observations...</p>`;
    try {
      const data = await apiGet("/api/watchlist/matches");
      renderObservations(data.observations || [], data.advisory || "", container);
    } catch (err) {
      container.innerHTML = `<p class="wl-error">${escHtml(err.message)}</p>`;
    }
  }

  function renderObservations(obs, advisory, container) {
    const banner = `<div class="wl-advisory-banner">⚠ ${escHtml(advisory)}</div>`;
    if (!obs.length) {
      container.innerHTML = banner + `<p class="wl-empty">No face-match observations found.</p>`;
      return;
    }

    const statusIcon = (r) => ({
      possible_match: "🔴",
      inconclusive: "🟡",
      no_match: "🟢",
      insufficient_evidence: "⚪",
      processing_failure: "❌",
    }[r] || "⚪");

    const rows = obs.map((o) => `
      <tr class="wl-obs-row ${o.match_result === "possible_match" ? "wl-possible-match-row" : ""}">
        <td class="wl-mono">${escHtml((o.obs_id || "").slice(0, 12))}…</td>
        <td>${escHtml(o.video_source || "—")}</td>
        <td>${o.source_timestamp != null ? o.source_timestamp.toFixed(2) + "s" : "—"}</td>
        <td>${statusIcon(o.match_result)} ${escHtml(o.match_result || "—")}</td>
        <td>${o.best_similarity != null ? (o.best_similarity * 100).toFixed(1) + "%" : "—"}</td>
        <td>${escHtml(o.image_quality || "—")}</td>
        <td>${o.reviewed ? "✓ Reviewed" : "Pending"}</td>
        <td>
          ${canDo("can_review") ? `<button class="wl-btn-sm" onclick="ARGUS_WL.openObservation('${escHtml(o.obs_id)}')">Review</button>` : ""}
        </td>
      </tr>`);

    container.innerHTML = banner + `
      <table class="wl-table">
        <thead>
          <tr>
            <th>ID</th><th>Video</th><th>Timestamp</th>
            <th>Result</th><th>Similarity</th><th>Quality</th>
            <th>Reviewed</th><th>Action</th>
          </tr>
        </thead>
        <tbody>${rows.join("")}</tbody>
      </table>`;
  }

  // ── Observation detail modal ─────────────────────────────────────────────
  async function openObservation(obsId) {
    const modal = document.getElementById("wlObsModal");
    if (!modal) return;
    document.getElementById("wlObsModalContent").innerHTML =
      `<p class="wl-loading">Loading observation detail...</p>`;
    modal.style.display = "flex";

    try {
      const data = await apiGet(`/api/watchlist/matches/${obsId}`);
      renderObservationModal(data);
    } catch (err) {
      document.getElementById("wlObsModalContent").innerHTML =
        `<p class="wl-error">${escHtml(err.message)}</p>`;
    }
  }

  function renderObservationModal(data) {
    const obs = data.observation || {};
    const entry = data.matched_entry || null;
    const reviews = data.reviews || [];
    const advisory = data.advisory || "";

    let refSection = "";
    if (entry && canDo("can_view_references")) {
      refSection = `
        <div class="wl-img-pair">
          <div class="wl-img-slot">
            <p class="wl-img-label">CCTV Face Crop</p>
            <img src="/api/watchlist/matches/${escHtml(obs.obs_id)}/crop"
                 class="wl-face-img" alt="CCTV face crop"
                 onerror="this.alt='Not available'">
          </div>
          <div class="wl-img-slot">
            <p class="wl-img-label">Reference Image (${escHtml(entry.display_name)})</p>
            <img src="/api/watchlist/matches/${escHtml(obs.obs_id)}/reference"
                 class="wl-face-img" alt="Reference image"
                 onerror="this.alt='Not available'">
          </div>
        </div>`;
    } else if (canDo("can_view_references")) {
      refSection = `<div class="wl-img-pair"><div class="wl-img-slot">
        <p class="wl-img-label">CCTV Face Crop</p>
        <img src="/api/watchlist/matches/${escHtml(obs.obs_id)}/crop"
             class="wl-face-img" alt="CCTV face crop"
             onerror="this.alt='Not available'">
      </div></div>`;
    }

    const reviewRows = reviews.map((r) => `
      <tr>
        <td>${escHtml(r.reviewer_id)}</td>
        <td>${escHtml(r.decision)}</td>
        <td>${r.notes ? escHtml(r.notes) : "—"}</td>
        <td>${new Date(r.review_timestamp * 1000).toLocaleString()}</td>
      </tr>`).join("") || `<tr><td colspan="4">No reviews yet.</td></tr>`;

    document.getElementById("wlObsModalContent").innerHTML = `
      <div class="wl-advisory-banner">⚠ ${escHtml(advisory)}</div>
      <div class="wl-obs-meta">
        <div><strong>Observation ID:</strong> <span class="wl-mono">${escHtml(obs.obs_id)}</span></div>
        <div><strong>Video:</strong> ${escHtml(obs.video_source)}</div>
        <div><strong>Source Timestamp:</strong> ${obs.source_timestamp != null ? obs.source_timestamp.toFixed(2) + "s" : "—"}</div>
        <div><strong>Match Result:</strong> <span class="wl-result-${obs.match_result}">${escHtml(obs.match_result)}</span></div>
        <div><strong>Similarity Score:</strong> ${obs.best_similarity != null ? (obs.best_similarity * 100).toFixed(1) + "%" : "—"}</div>
        <div><strong>Image Quality:</strong> ${escHtml(obs.image_quality)}</div>
        ${entry ? `<div><strong>Matched Entry:</strong> ${escHtml(entry.display_name)}</div>` : ""}
      </div>

      ${refSection}

      <p class="wl-disclaimer">
        A similarity score is <strong>not a probability of identity</strong>.
        This observation is AI-generated and requires independent human verification.
      </p>

      ${canDo("can_review") ? `
      <form id="wlReviewForm" class="wl-review-form">
        <h4>Submit Human Review</h4>
        <div class="wl-form-row">
          <label>Decision</label>
          <select id="wlReviewDecision" required>
            <option value="">— Select —</option>
            <option value="confirmed">Confirmed possible match (after assessment)</option>
            <option value="rejected">Rejected — does not match</option>
            <option value="inconclusive">Inconclusive</option>
            <option value="further_review">Requires further review</option>
          </select>
        </div>
        <div class="wl-form-row">
          <label>Notes</label>
          <textarea id="wlReviewNotes" rows="3" placeholder="Reviewer notes..."></textarea>
        </div>
        <button type="submit" class="wl-btn">Submit Review</button>
        <div id="wlReviewFeedback"></div>
        <p class="wl-disclaimer">
          Confirming does NOT constitute conclusive proof of identity and must not
          trigger automatic consequential actions without independent verification.
        </p>
      </form>` : ""}

      <h4>Review History</h4>
      <table class="wl-table wl-reviews-table">
        <thead><tr><th>Reviewer</th><th>Decision</th><th>Notes</th><th>Timestamp</th></tr></thead>
        <tbody>${reviewRows}</tbody>
      </table>`;

    // Wire review form
    const rf = document.getElementById("wlReviewForm");
    if (rf) {
      rf.addEventListener("submit", async (evt) => {
        evt.preventDefault();
        const decision = document.getElementById("wlReviewDecision").value;
        const notes = document.getElementById("wlReviewNotes").value.trim();
        const fb = document.getElementById("wlReviewFeedback");
        if (!decision) { fb.innerHTML = `<span class="wl-error">Select a decision.</span>`; return; }
        try {
          await apiPost(`/api/watchlist/matches/${obs.obs_id}/review`, { decision, notes });
          fb.innerHTML = `<span class="wl-success">✓ Review submitted.</span>`;
          rf.querySelector("button[type=submit]").disabled = true;
        } catch (err) {
          fb.innerHTML = `<span class="wl-error">${escHtml(err.message)}</span>`;
        }
      });
    }
  }

  // ── Audit log ────────────────────────────────────────────────────────────
  async function loadAuditLog() {
    const container = document.getElementById("wlAuditContainer");
    if (!container || !canDo("can_view_audit")) {
      if (container) container.innerHTML = `<p class="wl-error">Access restricted to administrators.</p>`;
      return;
    }
    container.innerHTML = `<p class="wl-loading">Loading audit log...</p>`;
    try {
      const data = await apiGet("/api/watchlist/audit?limit=200");
      const rows = (data.audit_log || []).map((e) => `
        <tr>
          <td>${new Date(e.timestamp * 1000).toLocaleString()}</td>
          <td class="wl-mono">${escHtml(e.actor_id || "system")}</td>
          <td>${escHtml(e.actor_role || "—")}</td>
          <td>${escHtml(e.action)}</td>
          <td>${escHtml(e.target_type || "—")}</td>
          <td class="wl-mono">${escHtml((e.target_id || "—").slice(0, 16))}</td>
        </tr>`).join("") || `<tr><td colspan="6">No audit entries.</td></tr>`;
      container.innerHTML = `
        <table class="wl-table">
          <thead><tr>
            <th>Time</th><th>Actor</th><th>Role</th>
            <th>Action</th><th>Type</th><th>Target</th>
          </tr></thead>
          <tbody>${rows}</tbody>
        </table>`;
    } catch (err) {
      container.innerHTML = `<p class="wl-error">${escHtml(err.message)}</p>`;
    }
  }

  // ── Entry actions ────────────────────────────────────────────────────────
  async function activateEntry(entryId) {
    if (!confirm("Activate this watchlist entry? It will be used in face matching.")) return;
    try {
      await apiPost(`/api/watchlist/${entryId}/activate`, {});
      showWlSuccess("Entry activated.");
      loadWatchlistEntries();
    } catch (err) { showWlError(err.message); }
  }

  async function revokeEntry(entryId) {
    const reason = prompt("Reason for revocation:");
    if (reason === null) return;
    try {
      await apiPost(`/api/watchlist/${entryId}/revoke?reason=${encodeURIComponent(reason)}`, {});
      showWlSuccess("Entry revoked.");
      loadWatchlistEntries();
    } catch (err) { showWlError(err.message); }
  }

  async function deleteEntry(entryId) {
    if (!confirm(
      "Permanently delete biometric data for this entry?\n\n" +
      "The entry record and audit trail will be retained.\n" +
      "This action cannot be undone."
    )) return;
    try {
      await apiDelete(`/api/watchlist/${entryId}`);
      showWlSuccess("Biometric data deleted. Entry record and audit trail preserved.");
      loadWatchlistEntries();
    } catch (err) { showWlError(err.message); }
  }

  // ── Utilities ────────────────────────────────────────────────────────────
  function escHtml(str) {
    if (str == null) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function showWlError(msg) {
    const el = document.getElementById("wlGlobalError");
    if (el) { el.textContent = msg; el.style.display = "block"; setTimeout(() => { el.style.display = "none"; }, 5000); }
  }

  function showWlSuccess(msg) {
    const el = document.getElementById("wlGlobalSuccess");
    if (el) { el.textContent = msg; el.style.display = "block"; setTimeout(() => { el.style.display = "none"; }, 4000); }
  }

  // ── Modal close ──────────────────────────────────────────────────────────
  function initModal() {
    const modal = document.getElementById("wlObsModal");
    const closeBtn = document.getElementById("wlObsModalClose");
    if (!modal) return;
    if (closeBtn) closeBtn.addEventListener("click", () => { modal.style.display = "none"; });
    modal.addEventListener("click", (e) => { if (e.target === modal) modal.style.display = "none"; });
  }

  // ── Public API ───────────────────────────────────────────────────────────
  window.ARGUS_WL = {
    activateEntry,
    revokeEntry,
    deleteEntry,
    openObservation,
  };

  // ── Init ─────────────────────────────────────────────────────────────────
  document.addEventListener("DOMContentLoaded", () => {
    restoreAuth();
    initLogin();
    initTabs();
    initEnrollmentForm();
    initAnalysisTrigger();
    initModal();

    // Nav button
    const navBtnWatchlist = document.getElementById("navBtnWatchlist");
    if (navBtnWatchlist) {
      navBtnWatchlist.addEventListener("click", () => {
        const watchlistScreen = document.getElementById("watchlistScreen");
        if (watchlistScreen) {
          document.querySelectorAll(".screen").forEach((s) => s.classList.remove("active"));
          watchlistScreen.classList.add("active");
          updateAuthUI();
        }
      });
    }
  });
})();

