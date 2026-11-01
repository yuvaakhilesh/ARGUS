// ==========================================================================
// ARGUS — Forensic Surveillance Intelligence & Crime Detection Controller
// Three-View Architecture: Overview, Flagged Clips, Detected Objects & Tracking
// ==========================================================================

document.addEventListener("DOMContentLoaded", () => {
  // ------------------------------------------------------------------------
  // DOM Elements
  // ------------------------------------------------------------------------

  // Screens
  const uploadScreen = document.getElementById("uploadScreen");
  const processingScreen = document.getElementById("processingScreen");
  const resultsScreen = document.getElementById("resultsScreen");

  // Screen 1: Upload
  const dropZone = document.getElementById("dropZone");
  const browseTriggerBtn = document.getElementById("browseTriggerBtn");
  const fileInput = document.getElementById("fileInput");
  const uploadErrorBanner = document.getElementById("uploadErrorBanner");
  const uploadErrorText = document.getElementById("uploadErrorText");
  const dismissErrorBtn = document.getElementById("dismissErrorBtn");
  const uploadLoadReportBtn = document.getElementById("uploadLoadReportBtn");
  const uploadReportInput = document.getElementById("uploadReportInput");

  // Screen 2: Processing
  const previewThumbnailCanvas = document.getElementById("previewThumbnailCanvas");
  const clientPreviewVideo = document.getElementById("clientPreviewVideo");
  const processingFileName = document.getElementById("processingFileName");
  const processingFileSize = document.getElementById("processingFileSize");
  const currentStageText = document.getElementById("currentStageText");
  const progressPercentageText = document.getElementById("progressPercentageText");
  const progressBarTrack = document.getElementById("progressBarTrack");
  const progressBarFill = document.getElementById("progressBarFill");
  const cancelProcessBtn = document.getElementById("cancelProcessBtn");

  const stageUploading = document.getElementById("stage-uploading");
  const stageExtracting = document.getElementById("stage-extracting");
  const stageRunning = document.getElementById("stage-running");
  const stageSummary = document.getElementById("stage-summary");

  // Screen 3: Results Header Actions
  const analyzeAnotherBtn = document.getElementById("analyzeAnotherBtn");
  const loadReportBtn = document.getElementById("loadReportBtn");
  const loadReportInput = document.getElementById("loadReportInput");
  const exportReportBtn = document.getElementById("exportReportBtn");

  // Three View Tabs
  const tabOverview = document.getElementById("tabOverview");
  const tabFlaggedClips = document.getElementById("tabFlaggedClips");
  const tabTracking = document.getElementById("tabTracking");
  const tabClipsBadge = document.getElementById("tabClipsBadge");

  // View Containers
  const viewOverview = document.getElementById("viewOverview");
  const viewFlaggedClips = document.getElementById("viewFlaggedClips");
  const viewTracking = document.getElementById("viewTracking");

  // View 1: Overview Elements
  const verdictBadge = document.getElementById("verdictBadge");
  const topClassTitle = document.getElementById("topClassTitle");
  const confidenceValue = document.getElementById("confidenceValue");
  const severityBadge = document.getElementById("severityBadge");
  const summaryDescription = document.getElementById("summaryDescription");

  const overviewVideoName = document.getElementById("overviewVideoName");
  const overviewVerdict = document.getElementById("overviewVerdict");
  const overviewIncidentsCount = document.getElementById("overviewIncidentsCount");
  const overviewFlaggedDuration = document.getElementById("overviewFlaggedDuration");
  const overviewFlaggedClipsCount = document.getElementById("overviewFlaggedClipsCount");
  const overviewSentToObjectAnalysis = document.getElementById("overviewSentToObjectAnalysis");
  const overviewProcessingTime = document.getElementById("overviewProcessingTime");
  const overviewModelDevice = document.getElementById("overviewModelDevice");

  const resultsVideoPlayer = document.getElementById("resultsVideoPlayer");
  const timelineBar = document.getElementById("timelineBar");
  const timelineSegments = document.getElementById("timelineSegments");
  const timelinePlayhead = document.getElementById("timelinePlayhead");
  const timelineStartLabel = document.getElementById("timelineStartLabel");
  const timelineCurrentLabel = document.getElementById("timelineCurrentLabel");
  const timelineEndLabel = document.getElementById("timelineEndLabel");
  const breakdownContent = document.getElementById("breakdownContent");

  // View 2: Flagged Clips Elements
  const clipsCountBadge = document.getElementById("clipsCountBadge");
  const clipsClassFilter = document.getElementById("clipsClassFilter");
  const clipsSortSelect = document.getElementById("clipsSortSelect");
  const clipsGrid = document.getElementById("clipsGrid");

  // View 3: Detected Objects & Tracking Elements
  const trackingClipSelect = document.getElementById("trackingClipSelect");
  const trackingVideoWrapper = document.getElementById("trackingVideoWrapper");
  const trackingVideoPlayer = document.getElementById("trackingVideoPlayer");
  const trackingCanvas = document.getElementById("trackingCanvas");
  const toggleOverlay = document.getElementById("toggleOverlay");
  const togglePersons = document.getElementById("togglePersons");
  const toggleObjects = document.getElementById("toggleObjects");
  const toggleWeapons = document.getElementById("toggleWeapons");
  const toggleVehicles = document.getElementById("toggleVehicles");
  const toggleOtherObjects = document.getElementById("toggleOtherObjects");
  const toggleTrajectories = document.getElementById("toggleTrajectories");
  const trackingTimeIndicator = document.getElementById("trackingTimeIndicator");

  const trackingSearchInput = document.getElementById("trackingSearchInput");
  const trackingSearchBtn = document.getElementById("trackingSearchBtn");
  const trackingResetFilterBtn = document.getElementById("trackingResetFilterBtn");
  const trackingSearchStatusBanner = document.getElementById("trackingSearchStatusBanner");
  const trackingSearchStatusText = document.getElementById("trackingSearchStatusText");
  const trackingSearchResultsBadge = document.getElementById("trackingSearchResultsBadge");
  const triggerModeBScanBtn = document.getElementById("triggerModeBScanBtn");
  const trackingCategoryFilter = document.getElementById("trackingCategoryFilter");
  const trackingSearchResultsCard = document.getElementById("trackingSearchResultsCard");
  const trackingResultsCount = document.getElementById("trackingResultsCount");
  const trackingSearchResultsList = document.getElementById("trackingSearchResultsList");
  const trackingAdvancedSettings = document.getElementById("trackingAdvancedSettings");

  const targetedScanClassSelect = document.getElementById("targetedScanClassSelect");
  const targetedScanStart = document.getElementById("targetedScanStart");
  const targetedScanEnd = document.getElementById("targetedScanEnd");
  const targetedScanFps = document.getElementById("targetedScanFps");
  const targetedScanRunBtn = document.getElementById("targetedScanRunBtn");
  const targetedScanFeedback = document.getElementById("targetedScanFeedback");

  const exportTrackingJsonBtn = document.getElementById("exportTrackingJsonBtn");
  const exportTrackingCsvBtn = document.getElementById("exportTrackingCsvBtn");

  const trackDetailModal = document.getElementById("trackDetailModal");
  const trackDetailTitle = document.getElementById("trackDetailTitle");
  const trackDetailGrid = document.getElementById("trackDetailGrid");
  const trackDetailIntervalsList = document.getElementById("trackDetailIntervalsList");
  const seekFirstSeenBtn = document.getElementById("seekFirstSeenBtn");
  const seekLastSeenBtn = document.getElementById("seekLastSeenBtn");
  const prevObservationBtn = document.getElementById("prevObservationBtn");
  const nextObservationBtn = document.getElementById("nextObservationBtn");
  const closeTrackDetailBtn = document.getElementById("closeTrackDetailBtn");

  const trackingEmptyBanner = document.getElementById("trackingEmptyBanner");
  const trackingEmptyBannerText = document.getElementById("trackingEmptyBannerText");
  const trackingRecommendationTitle = document.getElementById("trackingRecommendationTitle");
  const trackingTriageScore = document.getElementById("trackingTriageScore");
  const trackingDecisionReason = document.getElementById("trackingDecisionReason");
  const trackingTriggerReason = document.getElementById("trackingTriggerReason");
  const trackingCuesList = document.getElementById("trackingCuesList");
  const trackingTracksCount = document.getElementById("trackingTracksCount");
  const trackingTracksTableBody = document.getElementById("trackingTracksTableBody");

  let activeCatalogClasses = [];
  let currentFilterClass = null;
  let currentFilterCategory = null;
  let currentSearchQuery = "";
  let currentStatusFilter = "all"; // 'all' | 'confirmed_only' | 'uncertain_only'
  let selectedTrackForDetail = null;
  let selectedTrackObservationIndex = 0;
  let activeResolvedQuery = null;

  // ------------------------------------------------------------------------
  // State
  // ------------------------------------------------------------------------
  let currentJobId = null;
  let pollInterval = null;
  let activeVideoUrl = null;
  let activeFileName = null;
  let currentAnalysis = null;
  let abortController = null;
  let isCancelled = false;

  let currentSelectedClipIndex = 0;
  let animationFrameId = null;

  const ACCEPTED_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".webm"];
  const MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024 * 1024; // 25 GB (supports multi-hour CCTV)

  // ------------------------------------------------------------------------
  // Screen Routing & Transitions
  // ------------------------------------------------------------------------
  function showScreen(screen) {
    [uploadScreen, processingScreen, resultsScreen].forEach((s) => {
      s.classList.remove("active");
    });
    screen.classList.add("active");
    document.body.classList.toggle("results-mode", screen === resultsScreen);
    if (screen === uploadScreen && typeof updateTopNav === "function") {
      updateTopNav("analyze");
    }
  }

  // ------------------------------------------------------------------------
  // Tab Switching (Overview, Flagged Clips, Detected Objects & Tracking)
  // ------------------------------------------------------------------------
  const viewTabs = [
    { btn: tabOverview, panel: viewOverview },
    { btn: tabFlaggedClips, panel: viewFlaggedClips },
    { btn: tabTracking, panel: viewTracking },
  ];

  function switchView(targetName) {
    viewTabs.forEach(({ btn, panel }) => {
      const match = btn.dataset.view === targetName;
      btn.classList.toggle("active", match);
      btn.setAttribute("aria-selected", match ? "true" : "false");
      if (match) {
        panel.style.display = "block";
        panel.classList.add("active");
      } else {
        panel.style.display = "none";
        panel.classList.remove("active");
      }
    });

    if (typeof updateTopNav === "function") {
      if (targetName === "overview") updateTopNav("overview");
      if (targetName === "flagged-clips") updateTopNav("clips");
      if (targetName === "tracking") updateTopNav("tracking");
    }

    if (targetName === "tracking") {
      // Pause main video if playing, render tracking canvas
      resultsVideoPlayer.pause();
      renderSelectedClipTracking();
    } else {
      // Pause tracking video if playing
      trackingVideoPlayer.pause();
    }
  }

  tabOverview.addEventListener("click", () => switchView("overview"));
  tabFlaggedClips.addEventListener("click", () => switchView("flagged-clips"));
  tabTracking.addEventListener("click", () => switchView("tracking"));

  // ------------------------------------------------------------------------
  // Helpers
  // ------------------------------------------------------------------------
  function formatBytes(bytes, decimals = 1) {
    if (!bytes || bytes === 0) return "0 B";
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ["B", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + " " + sizes[i];
  }

  function formatTime(seconds, forceDecimal = false) {
    if (isNaN(seconds) || seconds < 0) return "00:00";
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    if (forceDecimal || (seconds > 0 && seconds < 1.0)) {
      return `${mins.toString().padStart(2, "0")}:${secs.toFixed(1).padStart(4, "0")}`;
    }
    const roundedSecs = Math.round(secs);
    const adjMins = mins + Math.floor(roundedSecs / 60);
    const finalSecs = roundedSecs % 60;
    return `${adjMins.toString().padStart(2, "0")}:${finalSecs.toString().padStart(2, "0")}`;
  }

  function showError(message) {
    uploadErrorText.textContent = message;
    uploadErrorBanner.style.display = "flex";
  }

  function clearError() {
    uploadErrorBanner.style.display = "none";
    uploadErrorText.textContent = "";
  }

  dismissErrorBtn.addEventListener("click", clearError);

  // ------------------------------------------------------------------------
  // Screen 1: File Validation & Drag-and-Drop
  // ------------------------------------------------------------------------
  function validateFile(file) {
    if (!file) {
      showError("Please select a video file.");
      return false;
    }

    const ext = "." + file.name.split(".").pop().toLowerCase();
    if (!ACCEPTED_EXTENSIONS.includes(ext)) {
      showError(`Unsupported file format (${ext}). Supported formats: MP4, AVI, MOV, MKV.`);
      return false;
    }

    if (file.size > MAX_FILE_SIZE_BYTES) {
      showError(`File size exceeds 25 GB limit (${formatBytes(file.size)}).`);
      return false;
    }

    clearError();
    return true;
  }

  // Browse Button Trigger
  browseTriggerBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  dropZone.addEventListener("click", () => {
    fileInput.click();
  });

  dropZone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });

  // Drag and drop events
  ["dragenter", "dragover"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add("dragover");
    });
  });

  ["dragleave", "dragend"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove("dragover");
    });
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.remove("dragover");

    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      handleFileSelected(files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  // Load Report JSON Trigger
  uploadLoadReportBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    uploadReportInput.click();
  });

  loadReportBtn.addEventListener("click", () => {
    loadReportInput.click();
  });

  [uploadReportInput, loadReportInput].forEach((inputEl) => {
    inputEl.addEventListener("change", (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleLoadJsonReport(e.target.files[0]);
      }
    });
  });

  function handleLoadJsonReport(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const rawJson = JSON.parse(e.target.result);
        normalizeAndDisplayJsonReport(rawJson, file.name);
      } catch (err) {
        alert("Failed to parse JSON report: " + err.message);
      }
    };
    reader.readAsText(file);
  }

  window.normalizeAndDisplayJsonReport = normalizeAndDisplayJsonReport;

  function normalizeAndDisplayJsonReport(data, fileName) {
    // Determine structure: could be direct API result, or wrapped exported report, or phase2 report
    let reportData = data;
    if (data.results) {
      reportData = data.results;
    } else if (data.clip_evaluations && Array.isArray(data.clip_evaluations) && data.clip_evaluations.length > 0) {
      // Pick first clip evaluation or summary
      const firstClip = data.clip_evaluations[0];
      reportData = {
        verdict: firstClip.predicted_class === "Normal" ? "Normal" : "Anomaly detected",
        top_class: firstClip.predicted_class,
        confidence: firstClip.confidence || 0.95,
        severity: firstClip.severity || "MEDIUM",
        summary: `Imported report evaluation for ${firstClip.file || fileName}.`,
        clips: firstClip.stealing_analysis_details ? firstClip.stealing_analysis_details.map((sa, idx) => ({
          clip_id: `INC-00${idx + 1}`,
          start: sa.analyzed_interval?.start_time || 0.0,
          end: sa.analyzed_interval?.end_time || 10.0,
          duration: (sa.analyzed_interval?.end_time || 10.0) - (sa.analyzed_interval?.start_time || 0.0),
          class: firstClip.predicted_class,
          confidence: firstClip.confidence || 0.95,
          severity: firstClip.severity || "MEDIUM",
          stealing_analysis: sa,
        })) : [],
        timeline: [],
        probabilities_14_class: {},
      };
    }

    currentJobId = reportData.job_id || "imported";
    activeVideoUrl = reportData.video_url || "";
    activeFileName = fileName.replace(".json", "");

    // Normalize clips
    const clips = reportData.clips || reportData.detected_clips || reportData.incidents || [];
    reportData.clips = clips;

    showScreen(resultsScreen);
    switchView("overview");
    renderResults(reportData, activeVideoUrl, activeFileName);
  }

  // ------------------------------------------------------------------------
  // Screen 2: Processing Flow
  // ------------------------------------------------------------------------
  function handleFileSelected(file) {
    if (!validateFile(file)) return;

    isCancelled = false;
    activeFileName = file.name;
    processingFileName.textContent = file.name;
    processingFileSize.textContent = formatBytes(file.size);

    // Extract local thumbnail preview
    generateClientThumbnail(file);

    // Reset progress UI
    updateProgress(10, "Uploading");
    resetStages();

    showScreen(processingScreen);

    // Start upload
    startUpload(file);
  }

  function generateClientThumbnail(file) {
    const ctx = previewThumbnailCanvas.getContext("2d");
    ctx.fillStyle = "#0f172a";
    ctx.fillRect(0, 0, previewThumbnailCanvas.width, previewThumbnailCanvas.height);
    ctx.fillStyle = "#64748b";
    ctx.font = "14px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Generating video preview...", previewThumbnailCanvas.width / 2, previewThumbnailCanvas.height / 2);

    try {
      const fileUrl = URL.createObjectURL(file);
      clientPreviewVideo.src = fileUrl;

      const renderFrame = () => {
        try {
          if (clientPreviewVideo.videoWidth > 0 && clientPreviewVideo.videoHeight > 0) {
            ctx.drawImage(clientPreviewVideo, 0, 0, previewThumbnailCanvas.width, previewThumbnailCanvas.height);
          }
        } catch (err) {
          console.warn("Could not draw frame canvas:", err);
        }
      };

      clientPreviewVideo.addEventListener("loadeddata", () => {
        renderFrame();
        if (clientPreviewVideo.duration > 0.5) {
          clientPreviewVideo.currentTime = 0.5;
        }
      }, { once: true });

      clientPreviewVideo.addEventListener("seeked", () => {
        renderFrame();
        URL.revokeObjectURL(fileUrl);
      }, { once: true });

      clientPreviewVideo.load();
    } catch (e) {
      console.warn("Client preview generation error:", e);
    }
  }

  function resetStages() {
    [stageUploading, stageExtracting, stageRunning, stageSummary].forEach((el) => {
      el.className = "stage-step";
    });
  }

  function updateProgress(percent, stageName) {
    progressBarFill.style.width = `${percent}%`;
    progressBarTrack.setAttribute("aria-valuenow", percent);
    progressPercentageText.textContent = `${Math.round(percent)}%`;
    currentStageText.textContent = stageName;

    if (stageName.toLowerCase().includes("upload")) {
      setStageActive(stageUploading);
    } else if (stageName.toLowerCase().includes("extract")) {
      setStageComplete(stageUploading);
      setStageActive(stageExtracting);
    } else if (stageName.toLowerCase().includes("videomae") || stageName.toLowerCase().includes("running")) {
      setStageComplete(stageUploading);
      setStageComplete(stageExtracting);
      setStageActive(stageRunning);
    } else if (stageName.toLowerCase().includes("summary") || percent >= 95) {
      setStageComplete(stageUploading);
      setStageComplete(stageExtracting);
      setStageComplete(stageRunning);
      setStageActive(stageSummary);
    }
  }

  function setStageActive(stageEl) {
    stageEl.className = "stage-step active";
  }

  function setStageComplete(stageEl) {
    stageEl.className = "stage-step complete";
  }

  // File Upload via XHR with real upload progress tracking
  function startUpload(file) {
    const formData = new FormData();
    formData.append("file", file);

    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/upload", true);

    xhr.upload.onprogress = (evt) => {
      if (evt.lengthComputable && !isCancelled) {
        const percent = Math.min(25, Math.round(10 + (evt.loaded / evt.total) * 15));
        const loadedStr = formatBytes(evt.loaded);
        const totalStr = formatBytes(evt.total);
        updateProgress(percent, `Uploading (${loadedStr} / ${totalStr})`);
      }
    };

    xhr.onload = () => {
      if (isCancelled) return;
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const data = JSON.parse(xhr.responseText);
          currentJobId = data.job_id;
          activeVideoUrl = data.url || data.video_url || URL.createObjectURL(file);
          updateProgress(25, "Extracting frames");
          pollStatus(currentJobId);
        } catch (e) {
          showError("Invalid server response format.");
          resetToUpload();
        }
      } else {
        let msg = `Upload failed with HTTP ${xhr.status}`;
        try {
          const errData = JSON.parse(xhr.responseText);
          if (errData.detail) msg = errData.detail;
        } catch (_) {}
        showError(msg);
        resetToUpload();
      }
    };

    xhr.onerror = () => {
      if (!isCancelled) {
        showError("Network error during video upload.");
        resetToUpload();
      }
    };

    xhr.send(formData);
  }

  // Poll Job Status
  function pollStatus(jobId) {
    if (pollInterval) clearInterval(pollInterval);

    pollInterval = setInterval(async () => {
      if (isCancelled) {
        clearInterval(pollInterval);
        return;
      }

      try {
        const response = await fetch(`/status/${jobId}`);
        if (!response.ok) return;

        const data = await response.json();

        if (data.status === "processing") {
          updateProgress(data.progress || 25, data.stage || "Processing video");
        } else if (data.status === "completed" || data.status === "done") {
          clearInterval(pollInterval);
          updateProgress(100, "Complete");
          setStageComplete(stageSummary);

          currentAnalysis = data.results;
          setTimeout(() => {
            showScreen(resultsScreen);
            switchView("overview");
            renderResults(data.results, activeVideoUrl, activeFileName);
          }, 450);
        } else if (data.status === "failed" || data.status === "error") {
          clearInterval(pollInterval);
          showError(data.error || "Analysis encountered an error.");
          resetToUpload();
        }
      } catch (err) {
        console.warn("Status poll error:", err);
      }
    }, 1500);
  }

  // Cancel Button Action
  cancelProcessBtn.addEventListener("click", async () => {
    isCancelled = true;
    if (pollInterval) clearInterval(pollInterval);
    if (abortController) abortController.abort();

    if (currentJobId) {
      try {
        await fetch(`/cancel/${currentJobId}`, { method: "POST" });
      } catch (e) {
        console.warn("Cancel request error:", e);
      }
    }

    resetToUpload();
  });

  function resetToUpload() {
    isCancelled = true;
    currentJobId = null;
    if (pollInterval) clearInterval(pollInterval);
    fileInput.value = "";
    resultsVideoPlayer.pause();
    resultsVideoPlayer.src = "";
    trackingVideoPlayer.pause();
    trackingVideoPlayer.src = "";
    showScreen(uploadScreen);
  }

  // ------------------------------------------------------------------------
  // Screen 3: Results Rendering (3 Views)
  // ------------------------------------------------------------------------
  function renderResults(results, videoUrl, filename) {
    if (!results) return;
    currentAnalysis = results;

    const clips = results.clips || results.detected_clips || results.incidents || [];
    const incidents = results.incidents || [];

    // Setup Video Player
    if (videoUrl) {
      resultsVideoPlayer.src = videoUrl;
      resultsVideoPlayer.load();
    }

    // ======================================================================
    // 1. VIEW 1: OVERVIEW RENDERING
    // ======================================================================
    const isUnassessable = Boolean(results.is_unassessable || (results.verdict && results.verdict.toLowerCase().includes("unassessable")));
    const isAnomaly = !isUnassessable && results.verdict === "Anomaly detected";

    if (isUnassessable) {
      verdictBadge.textContent = "UNASSESSABLE";
      verdictBadge.className = "verdict-badge unassessable";
      topClassTitle.textContent = "Insufficient Visual Information";
      confidenceValue.textContent = "--%";
      severityBadge.textContent = "NONE";
      severityBadge.className = "severity-badge none";
      summaryDescription.innerHTML = `
        <div class="quality-warning-box">
          <strong>Quality Warning:</strong> ${results.quality_warning || results.summary}
        </div>
      `;
    } else {
      verdictBadge.textContent = results.verdict || "not available";
      verdictBadge.className = `verdict-badge ${isAnomaly ? "anomaly" : "normal"}`;
      topClassTitle.textContent = results.top_class || (isAnomaly ? "Security Incident" : "Normal Activity");
      confidenceValue.textContent = results.confidence !== undefined ? `${Math.round(results.confidence * 100)}%` : "--%";
      const sev = (results.severity || "NONE").toUpperCase();
      severityBadge.textContent = sev;
      severityBadge.className = `severity-badge ${sev.toLowerCase()}`;
      summaryDescription.textContent = results.summary || "Surveillance footage analyzed.";
    }

    // Overview Key Metrics Grid
    overviewVideoName.textContent = filename || results.metadata?.video_name || "not available";
    overviewVerdict.textContent = results.verdict || "not available";
    overviewIncidentsCount.textContent = `${incidents.length || clips.length || 0}`;

    // Total flagged duration
    let totalFlaggedSec = 0;
    clips.forEach((c) => {
      const dur = c.duration || (c.end - c.start) || 0;
      totalFlaggedSec += dur;
    });
    overviewFlaggedDuration.textContent = `${totalFlaggedSec.toFixed(1)}s`;
    overviewFlaggedClipsCount.textContent = `${clips.length}`;

    // Sent to object analysis count
    const sentToObjectAnalysisCount = clips.filter((c) => {
      const sa = c.stealing_analysis;
      return sa && sa.status && sa.status !== "skipped_not_candidate";
    }).length;
    overviewSentToObjectAnalysis.textContent = `${sentToObjectAnalysisCount} clip${sentToObjectAnalysisCount === 1 ? '' : 's'}`;

    // Processing time
    const procTime = results.total_processing_time_s || results.processing_time_s || results.metadata?.processing_time_s;
    overviewProcessingTime.textContent = procTime ? `${procTime}s` : "not available";

    // Model and device line (only if provided by API)
    if (results.model_used || results.device || results.engine_mode) {
      const parts = [];
      if (results.model_used) parts.push(`Model: ${results.model_used}`);
      if (results.engine_mode) parts.push(`Engine: ${results.engine_mode}`);
      if (results.device) parts.push(`Device: ${results.device}`);
      overviewModelDevice.textContent = parts.join(" | ");
    } else {
      overviewModelDevice.textContent = "not available";
    }

    // Overview Video Duration
    const overviewVideoDuration = document.getElementById("overviewVideoDuration");
    if (overviewVideoDuration) {
      const vidDur = results.metadata?.duration || results.video_duration_s || (results.timeline && results.timeline.length > 0 ? results.timeline[results.timeline.length - 1].end : null);
      overviewVideoDuration.textContent = vidDur ? formatTime(vidDur) : "not available";
    }

    const playerStatusTag = document.getElementById("overviewPlayerStatusTag");
    if (playerStatusTag) {
      playerStatusTag.textContent = videoUrl ? "STREAM ACTIVE" : (filename ? "METADATA AUDIT" : "READY");
    }

    // Render Incidents List on Overview
    const overviewIncidentsList = document.getElementById("overviewIncidentsList");
    const overviewIncidentsHeaderBadge = document.getElementById("overviewIncidentsHeaderBadge");
    if (overviewIncidentsHeaderBadge) {
      overviewIncidentsHeaderBadge.textContent = `${clips.length}`;
    }
    if (overviewIncidentsList) {
      overviewIncidentsList.innerHTML = "";
      if (clips.length === 0) {
        overviewIncidentsList.innerHTML = `
          <div class="no-incidents-state">
            <span class="state-icon">✓</span>
            <p class="state-msg">No security incidents flagged across the duration of this recording.</p>
          </div>
        `;
      } else {
        clips.forEach((clip, idx) => {
          const item = document.createElement("div");
          item.className = "overview-incident-item";
          const start = clip.start || 0;
          const end = clip.end || 0;
          const dur = clip.duration || (end - start) || 0;
          const conf = clip.confidence ? `${Math.round(clip.confidence * 100)}%` : "100%";
          const sev = (clip.severity || "MEDIUM").toUpperCase();
          const sevClass = sev.toLowerCase();
          
          item.innerHTML = `
            <div class="incident-item-header">
              <span class="incident-id">${clip.clip_id || `INC-00${idx + 1}`}</span>
              <span class="incident-class-tag">${clip.class || "Security Incident"}</span>
              <span class="severity-badge ${sevClass}">${sev}</span>
              <span class="incident-conf">${conf}</span>
            </div>
            <div class="incident-item-time">
              <span>⏱ ${formatTime(start)} – ${formatTime(end)} (${dur.toFixed(1)}s)</span>
            </div>
            <div class="incident-item-actions">
              <button type="button" class="incident-action-btn seek-btn" data-start="${start}">
                ▶ Play at ${formatTime(start)}
              </button>
              <button type="button" class="incident-action-btn tab-clip-btn" data-index="${idx}">
                Evidence Clip →
              </button>
              <button type="button" class="incident-action-btn tab-track-btn" data-index="${idx}">
                Tracking →
              </button>
            </div>
          `;

          // Event listeners
          item.querySelector(".seek-btn")?.addEventListener("click", (e) => {
            e.stopPropagation();
            if (resultsVideoPlayer) {
              resultsVideoPlayer.currentTime = start;
              resultsVideoPlayer.play();
            }
          });
          item.querySelector(".tab-clip-btn")?.addEventListener("click", (e) => {
            e.stopPropagation();
            tabFlaggedClips.click();
          });
          item.querySelector(".tab-track-btn")?.addEventListener("click", (e) => {
            e.stopPropagation();
            tabTracking.click();
          });
          item.addEventListener("click", () => {
            if (resultsVideoPlayer) {
              resultsVideoPlayer.currentTime = start;
              resultsVideoPlayer.play();
            }
          });

          overviewIncidentsList.appendChild(item);
        });
      }
    }

    // Render Timeline Bar
    renderTimelineBar(results.timeline, results.metadata ? results.metadata.duration : 0);

    // Render Probability Breakdown
    renderProbabilityBreakdown(results.probabilities_14_class || {});

    // Update Tab Badge
    tabClipsBadge.textContent = `${clips.length}`;

    // ======================================================================
    // 2. VIEW 2: FLAGGED CLIPS RENDERING
    // ======================================================================
    populateClassFilter(clips);
    renderFlaggedClips(clips);

    // ======================================================================
    // 3. VIEW 3: TRACKING & OBJECTS RENDERING
    // ======================================================================
    populateTrackingSelector(clips);
    currentSelectedClipIndex = 0;
    renderSelectedClipTracking();
  }

  // ------------------------------------------------------------------------
  // Timeline Synchronization
  // ------------------------------------------------------------------------
  resultsVideoPlayer.addEventListener("timeupdate", () => {
    const cur = resultsVideoPlayer.currentTime;
    const dur = resultsVideoPlayer.duration || 1;

    timelineCurrentLabel.textContent = formatTime(cur);
    const pct = Math.min(100, Math.max(0, (cur / dur) * 100));
    timelinePlayhead.style.left = `${pct}%`;
  });

  resultsVideoPlayer.addEventListener("loadedmetadata", () => {
    timelineEndLabel.textContent = formatTime(resultsVideoPlayer.duration);
    timelineStartLabel.textContent = "00:00";
  });

  function renderTimelineBar(timeline, totalDuration) {
    timelineSegments.innerHTML = "";
    if (!timeline || timeline.length === 0) return;

    const dur = totalDuration || (timeline[timeline.length - 1].end || 1);
    timelineEndLabel.textContent = formatTime(dur);

    timeline.forEach((pt) => {
      const seg = document.createElement("div");
      seg.className = "timeline-segment";

      const startPct = (pt.start / dur) * 100;
      const endPct = (pt.end / dur) * 100;
      const widthPct = Math.max(endPct - startPct, 1.2);

      seg.style.width = `${widthPct}%`;
      seg.style.backgroundColor = pt.is_anomaly ? "#ef4444" : "#cbd5e1";
      seg.title = `${formatTime(pt.start)} – ${formatTime(pt.end)}: ${pt.class} (${Math.round((pt.score || 0) * 100)}%)`;

      seg.addEventListener("click", (e) => {
        e.stopPropagation();
        resultsVideoPlayer.currentTime = pt.start;
        resultsVideoPlayer.play();
      });

      timelineSegments.appendChild(seg);
    });
  }

  timelineBar.addEventListener("click", (e) => {
    const rect = timelineBar.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const pct = Math.max(0, Math.min(1, clickX / rect.width));

    if (resultsVideoPlayer.duration) {
      resultsVideoPlayer.currentTime = pct * resultsVideoPlayer.duration;
      resultsVideoPlayer.play();
    }
  });

  // ------------------------------------------------------------------------
  // View 2: Flagged Clips Functions
  // ------------------------------------------------------------------------
  function populateClassFilter(clips) {
    clipsClassFilter.innerHTML = '<option value="ALL">All Classes</option>';
    const classes = Array.from(new Set(clips.map((c) => c.class).filter(Boolean))).sort();
    classes.forEach((cls) => {
      const opt = document.createElement("option");
      opt.value = cls;
      opt.textContent = cls;
      clipsClassFilter.appendChild(opt);
    });
  }

  clipsClassFilter.addEventListener("change", () => {
    if (currentAnalysis) {
      const clips = currentAnalysis.clips || currentAnalysis.detected_clips || [];
      renderFlaggedClips(clips);
    }
  });

  clipsSortSelect.addEventListener("change", () => {
    if (currentAnalysis) {
      const clips = currentAnalysis.clips || currentAnalysis.detected_clips || [];
      renderFlaggedClips(clips);
    }
  });

  function renderFlaggedClips(clips) {
    clipsGrid.innerHTML = "";
    if (!clips || clips.length === 0) {
      clipsCountBadge.textContent = "0 clips";
      const emptyCard = document.createElement("div");
      emptyCard.className = "empty-clips-card";
      emptyCard.textContent = "No anomalies detected across the timeline.";
      clipsGrid.appendChild(emptyCard);
      return;
    }

    // Filter by class
    const selectedClass = clipsClassFilter.value;
    let filtered = selectedClass === "ALL" ? [...clips] : clips.filter((c) => c.class === selectedClass);

    // Sort
    const sortVal = clipsSortSelect.value;
    filtered.sort((a, b) => {
      const aStart = a.clip_start !== undefined ? a.clip_start : a.start;
      const bStart = b.clip_start !== undefined ? b.clip_start : b.start;
      const aDur = a.duration || (a.end - a.start) || 0;
      const bDur = b.duration || (b.end - b.start) || 0;
      const aConf = a.confidence || 0;
      const bConf = b.confidence || 0;

      if (sortVal === "time_asc") return aStart - bStart;
      if (sortVal === "time_desc") return bStart - aStart;
      if (sortVal === "conf_desc") return bConf - aConf;
      if (sortVal === "conf_asc") return aConf - bConf;
      if (sortVal === "duration") return bDur - aDur;
      return 0;
    });

    clipsCountBadge.textContent = `${filtered.length} clip${filtered.length === 1 ? '' : 's'}`;

    filtered.forEach((clip) => {
      const card = document.createElement("article");
      card.className = "clip-card clickable-card";

      const clipStart = clip.clip_start !== undefined ? clip.clip_start : clip.start;
      const clipEnd = clip.clip_end !== undefined ? clip.clip_end : clip.end;
      const clipDur = clip.duration !== undefined ? clip.duration : Math.round(clipEnd - clipStart);

      const thumbImg = clip.thumbnail_url
        ? `<img src="${clip.thumbnail_url}" alt="Incident frame" class="clip-thumbnail" loading="lazy">`
        : `<div class="clip-thumbnail placeholder-thumb"><span>${clip.class}</span></div>`;

      // Status badge for stealing-analysis
      const sa = clip.stealing_analysis;
      let statusBadgeHtml = '<span class="status-badge status-not-run">Not Run</span>';
      if (sa && sa.status) {
        if (sa.status === "completed") {
          statusBadgeHtml = '<span class="status-badge status-completed">Completed</span>';
        } else if (sa.status === "persons_only") {
          statusBadgeHtml = '<span class="status-badge status-persons">Persons Only</span>';
        } else if (sa.status === "no_objects_detected" || sa.status === "no_relevant_entities") {
          statusBadgeHtml = '<span class="status-badge status-none">No Objects</span>';
        } else if (sa.status === "error") {
          statusBadgeHtml = '<span class="status-badge status-error">Error</span>';
        }
      }

      card.innerHTML = `
        <div class="clip-thumbnail-wrapper">
          ${thumbImg}
        </div>
        <div class="clip-info">
          <div class="clip-header-row">
            <span class="clip-time-badge">${formatTime(clipStart, ((clipDur < 1.0) || (Math.floor(clipStart) === Math.floor(clipEnd) && clipEnd > clipStart)))} – ${formatTime(clipEnd, ((clipDur < 1.0) || (Math.floor(clipStart) === Math.floor(clipEnd) && clipEnd > clipStart)))} (${clipDur}s)</span>
            <span class="clip-class-label">${clip.class}</span>
            <span class="clip-confidence">(${Math.round((clip.confidence || 0) * 100)}% conf)</span>
            ${statusBadgeHtml}
          </div>
          <p class="clip-desc">${clip.description || `${clip.class} identified in this segment.`}</p>
        </div>
        <div class="clip-actions">
          <button type="button" class="btn btn-secondary btn-sm play-clip-btn" data-start="${clipStart}">
            ▶ Play
          </button>
          <button type="button" class="btn btn-outline btn-sm inspect-tracking-btn">
            Inspect Tracking →
          </button>
        </div>
      `;

      // Play button action
      const playBtn = card.querySelector(".play-clip-btn");
      playBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        resultsVideoPlayer.currentTime = clipStart;
        resultsVideoPlayer.play();
        switchView("overview");
        resultsVideoPlayer.scrollIntoView({ behavior: "smooth", block: "center" });
      });

      // Clicking card or "Inspect Tracking" opens in View 3
      const openTracking = () => {
        const fullClips = currentAnalysis.clips || currentAnalysis.detected_clips || [];
        const indexInFull = fullClips.indexOf(clip);
        if (indexInFull !== -1) {
          currentSelectedClipIndex = indexInFull;
          trackingClipSelect.value = indexInFull;
        }
        switchView("tracking");
      };

      const inspectBtn = card.querySelector(".inspect-tracking-btn");
      inspectBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        openTracking();
      });

      card.addEventListener("click", openTracking);

      clipsGrid.appendChild(card);
    });
  }

  // ------------------------------------------------------------------------
  // View 3: Detected Objects & Tracking Functions (Refined Object Explorer)
  // ------------------------------------------------------------------------

  // Category mapping helper
  const CLASS_CATEGORY_MAP = {
    // Weapons
    "Pistol": "Weapons", "Rifle": "Weapons", "Knife": "Weapons",
    // People
    "person": "People",
    // Personal Belongings
    "backpack": "Personal Belongings", "umbrella": "Personal Belongings",
    "handbag": "Personal Belongings", "tie": "Personal Belongings",
    "suitcase": "Personal Belongings", "cell phone": "Personal Belongings",
    // Vehicles
    "bicycle": "Vehicles", "car": "Vehicles", "motorcycle": "Vehicles",
    "airplane": "Vehicles", "bus": "Vehicles", "train": "Vehicles",
    "truck": "Vehicles", "boat": "Vehicles",
    // Electronics & Appliances
    "tv": "Electronics & Appliances", "laptop": "Electronics & Appliances",
    "mouse": "Electronics & Appliances", "remote": "Electronics & Appliances",
    "keyboard": "Electronics & Appliances", "microwave": "Electronics & Appliances",
    "oven": "Electronics & Appliances", "toaster": "Electronics & Appliances",
    "sink": "Electronics & Appliances", "refrigerator": "Electronics & Appliances",
    // Kitchen & Dining
    "bottle": "Kitchen & Dining", "wine glass": "Kitchen & Dining",
    "cup": "Kitchen & Dining", "fork": "Kitchen & Dining",
    "knife": "Kitchen & Dining", "spoon": "Kitchen & Dining", "bowl": "Kitchen & Dining",
    // Furniture & Interior
    "chair": "Furniture & Interior", "couch": "Furniture & Interior",
    "potted plant": "Furniture & Interior", "bed": "Furniture & Interior",
    "dining table": "Furniture & Interior", "toilet": "Furniture & Interior",
    // Sports Equipment
    "frisbee": "Sports Equipment", "skis": "Sports Equipment",
    "snowboard": "Sports Equipment", "sports ball": "Sports Equipment",
    "kite": "Sports Equipment", "baseball bat": "Sports Equipment",
    "baseball glove": "Sports Equipment", "skateboard": "Sports Equipment",
    "surfboard": "Sports Equipment", "tennis racket": "Sports Equipment",
    // Animals
    "bird": "Animals", "cat": "Animals", "dog": "Animals",
    "horse": "Animals", "sheep": "Animals", "cow": "Animals",
    "elephant": "Animals", "bear": "Animals", "zebra": "Animals", "giraffe": "Animals"
  };

  const CATEGORY_COLOR_MAP = {
    "Weapons": "#ef4444",
    "People": "#06b6d4",
    "Personal Belongings": "#a855f7",
    "Vehicles": "#f59e0b",
    "Electronics & Appliances": "#3b82f6",
    "Kitchen & Dining": "#10b981",
    "Furniture & Interior": "#8b5cf6",
    "Sports Equipment": "#ec4899",
    "Animals": "#84cc16",
    "Household & Accessories": "#0284c7",
    "Other Objects": "#94a3b8"
  };

  function getObjectCategory(className) {
    if (!className) return "Other Objects";
    if (CLASS_CATEGORY_MAP[className]) return CLASS_CATEGORY_MAP[className];
    const lower = className.toLowerCase();
    for (const [k, v] of Object.entries(CLASS_CATEGORY_MAP)) {
      if (k.toLowerCase() === lower) return v;
    }
    return "Other Objects";
  }

  function getCategoryColor(category, className) {
    if (className === "Pistol" || className === "Rifle" || className === "Knife" || category === "Weapons") {
      return "#ef4444";
    }
    return CATEGORY_COLOR_MAP[category] || "#94a3b8";
  }

  // Load backend catalog classes for search & targeted scan
  async function loadCatalogClasses() {
    try {
      const resp = await fetch("/api/tracking/classes");
      if (resp.ok) {
        const data = await resp.json();
        activeCatalogClasses = data.classes || [];
        if (targetedScanClassSelect && activeCatalogClasses.length > 0) {
          targetedScanClassSelect.innerHTML = "";
          // Group by category
          const groups = {};
          activeCatalogClasses.forEach((cls) => {
            const cat = cls.category || "Other";
            if (!groups[cat]) groups[cat] = [];
            groups[cat].push(cls);
          });
          for (const [cat, items] of Object.entries(groups)) {
            const optgroup = document.createElement("optgroup");
            optgroup.label = cat;
            items.forEach((item) => {
              const opt = document.createElement("option");
              opt.value = item.name;
              opt.textContent = `${item.name} (${item.source})`;
              optgroup.appendChild(opt);
            });
            targetedScanClassSelect.appendChild(optgroup);
          }
        }
      }
    } catch (e) {
      console.warn("Could not load tracking classes catalog:", e);
    }
  }

  function populateTrackingSelector(clips) {
    trackingClipSelect.innerHTML = "";
    if (!clips || clips.length === 0) {
      trackingClipSelect.innerHTML = '<option value="">No clips available</option>';
      return;
    }

    clips.forEach((clip, idx) => {
      const opt = document.createElement("option");
      opt.value = idx;
      const start = clip.clip_start !== undefined ? clip.clip_start : clip.start;
      const end = clip.clip_end !== undefined ? clip.clip_end : clip.end;
      opt.textContent = `${clip.clip_id || `INC-${idx + 1}`} (${formatTime(start)} - ${formatTime(end)}) — ${clip.class} (${Math.round((clip.confidence || 0) * 100)}%)`;
      trackingClipSelect.appendChild(opt);
    });

    if (activeCatalogClasses.length === 0) {
      loadCatalogClasses();
    }
  }

  trackingClipSelect.addEventListener("change", (e) => {
    currentSelectedClipIndex = parseInt(e.target.value, 10) || 0;
    renderSelectedClipTracking();
  });

  // HTML escaping helper
  function escapeHtml(str) {
    if (!str) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // Universal Custom Object Search (Section 1)
  async function executeUniversalSearch(queryText) {
    currentSearchQuery = (queryText || "").trim();
    if (trackingSearchInput && trackingSearchInput.value !== currentSearchQuery) {
      trackingSearchInput.value = currentSearchQuery;
    }

    if (!currentSearchQuery) {
      activeResolvedQuery = null;
      if (trackingSearchStatusBanner) trackingSearchStatusBanner.style.display = "none";
      if (trackingResetFilterBtn) trackingResetFilterBtn.style.display = "none";
      renderSelectedClipTracking();
      return;
    }

    if (trackingResetFilterBtn) {
      trackingResetFilterBtn.style.display = "inline-flex";
    }

    try {
      const resp = await fetch(`/api/tracking/resolve-query?q=${encodeURIComponent(currentSearchQuery)}`);
      if (resp.ok) {
        activeResolvedQuery = await resp.json();
      } else {
        const qLower = currentSearchQuery.toLowerCase();
        activeResolvedQuery = {
          query: currentSearchQuery,
          canonical_classes: [qLower],
          matched_aliases: [],
          is_supported: activeCatalogClasses.some(c => c.name.toLowerCase() === qLower),
          explanation: `Direct query: ${currentSearchQuery}`
        };
      }
    } catch (err) {
      console.warn("Query resolution fallback:", err);
      const qLower = currentSearchQuery.toLowerCase();
      activeResolvedQuery = {
        query: currentSearchQuery,
        canonical_classes: [qLower],
        matched_aliases: [],
        is_supported: activeCatalogClasses.some(c => c.name.toLowerCase() === qLower),
        explanation: `Direct query: ${currentSearchQuery}`
      };
    }

    renderSelectedClipTracking();
  }

  // Event Listeners for Search
  trackingSearchInput?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      executeUniversalSearch(trackingSearchInput.value);
    }
  });

  trackingSearchBtn?.addEventListener("click", () => {
    executeUniversalSearch(trackingSearchInput ? trackingSearchInput.value : "");
  });

  let searchDebounceTimeout = null;
  trackingSearchInput?.addEventListener("input", (e) => {
    const val = (e.target.value || "").trim();
    if (!val) {
      executeUniversalSearch("");
      return;
    }
    clearTimeout(searchDebounceTimeout);
    searchDebounceTimeout = setTimeout(() => {
      executeUniversalSearch(val);
    }, 350);
  });

  // Quick Query Pills
  document.querySelectorAll(".quick-query-pill").forEach((pill) => {
    pill.addEventListener("click", () => {
      const q = pill.getAttribute("data-query");
      executeUniversalSearch(q);
    });
  });

  // Status Filter Pills (All, Confirmed, Uncertain)
  document.querySelectorAll(".status-filter-pill").forEach((pill) => {
    pill.addEventListener("click", () => {
      document.querySelectorAll(".status-filter-pill").forEach((p) => p.classList.remove("active"));
      pill.classList.add("active");
      currentStatusFilter = pill.getAttribute("data-filter") || "all";
      renderSelectedClipTracking();
    });
  });

  trackingResetFilterBtn?.addEventListener("click", () => {
    currentFilterClass = null;
    currentFilterCategory = null;
    currentSearchQuery = "";
    currentStatusFilter = "all";
    activeResolvedQuery = null;
    if (trackingSearchInput) trackingSearchInput.value = "";
    if (trackingCategoryFilter) trackingCategoryFilter.value = "";
    document.querySelectorAll(".status-filter-pill").forEach((p) => {
      p.classList.toggle("active", p.getAttribute("data-filter") === "all");
    });
    if (trackingResetFilterBtn) trackingResetFilterBtn.style.display = "none";
    if (trackingSearchStatusBanner) trackingSearchStatusBanner.style.display = "none";
    renderSelectedClipTracking();
  });

  // Mode B Scan trigger button from search banner
  triggerModeBScanBtn?.addEventListener("click", () => {
    if (trackingAdvancedSettings) {
      trackingAdvancedSettings.open = true;
      if (activeResolvedQuery && activeResolvedQuery.canonical_classes && activeResolvedQuery.canonical_classes.length > 0) {
        if (targetedScanClassSelect) {
          targetedScanClassSelect.value = activeResolvedQuery.canonical_classes[0];
        }
      }
      trackingAdvancedSettings.scrollIntoView({ behavior: "smooth" });
    }
  });

  function updateFilterStatusAndRender() {
    const isFiltered = Boolean(currentFilterClass || currentFilterCategory || currentSearchQuery || currentStatusFilter !== "all");
    if (trackingResetFilterBtn) {
      trackingResetFilterBtn.style.display = isFiltered ? "inline-flex" : "none";
    }
    renderSelectedClipTracking();
  }

  function renderSelectedClipTracking() {
    if (!currentAnalysis) return;
    const clips = currentAnalysis.clips || currentAnalysis.detected_clips || currentAnalysis.incidents || [];
    if (!clips || clips.length === 0 || !clips[currentSelectedClipIndex]) {
      trackingEmptyBanner.style.display = "flex";
      trackingEmptyBannerText.textContent = "No incident clips available for tracking inspection.";
      return;
    }

    const clip = clips[currentSelectedClipIndex];
    const sa = clip.stealing_analysis || clip.tracking_analysis || {};
    const rawTrackSummaries = sa.track_summaries || [];
    const clipStartS = sa.clip_start_s !== undefined ? sa.clip_start_s : (clip.clip_start !== undefined ? clip.clip_start : (clip.start || 0.0));

    // 1. Setup Tracking Video Player
    const isClipTrimmedVideo = (clip.clip_is_trimmed !== undefined)
      ? Boolean(clip.clip_is_trimmed)
      : (sa.clip_is_trimmed !== undefined
          ? Boolean(sa.clip_is_trimmed)
          : Boolean(clip.clip_url));

    const videoSrc = (isClipTrimmedVideo && clip.clip_url) ? clip.clip_url : activeVideoUrl;

    if (videoSrc && !trackingVideoPlayer.src.endsWith(videoSrc) && trackingVideoPlayer.src !== videoSrc) {
      trackingVideoPlayer.src = videoSrc;
      trackingVideoPlayer.load();
    }

    if (!isClipTrimmedVideo && clipStartS > 0) {
      trackingVideoPlayer.currentTime = clipStartS;
    }

    // 2. Filter Tracks by Search Query, Status Filter, Category, and Toggles
    const filteredTracks = filterTracks(rawTrackSummaries);

    // 3. Empty State Banner Check
    if (rawTrackSummaries.length === 0) {
      trackingEmptyBanner.style.display = "flex";
      trackingEmptyBannerText.textContent = "No objects or persons were detected in this candidate interval.";
    } else if (filteredTracks.length === 0 && (currentSearchQuery || currentStatusFilter !== "all" || currentFilterClass || currentFilterCategory)) {
      trackingEmptyBanner.style.display = "flex";
      trackingEmptyBannerText.textContent = "No objects match the current search query or active filter in this interval.";
    } else {
      trackingEmptyBanner.style.display = "none";
    }

    // 5. Update Search Status Banner
    updateSearchStatusFeedback(rawTrackSummaries, filteredTracks);

    // 6. Triage Card
    const wa = clip.weapon_analysis || {};
    const waCues = wa.cues || [];
    const saCues = sa.interaction_cues || [];
    const cues = [...waCues, ...saCues];

    if (wa.weapons_detected && wa.weapons_detected.length > 0) {
      trackingRecommendationTitle.textContent = `${wa.review_recommendation || "PRIORITY_HUMAN_REVIEW"} [${wa.weapons_detected.join(", ")}]`;
      trackingRecommendationTitle.style.color = "#ef4444";
      trackingDecisionReason.textContent = `${wa.decision_reason || "Weapon detected"} (AI triage cue — spatial proximity does not prove possession).`;
      trackingTriageScore.textContent = `${Math.round((wa.confidence || 0.9) * 100)}%`;
      trackingTriggerReason.textContent = `Dedicated weapon detection triggered: ${wa.weapons_detected.join(", ")}`;
    } else {
      const scoreVal = sa.uncalibrated_interaction_score !== undefined ? sa.uncalibrated_interaction_score : 0.0;
      trackingTriageScore.textContent = `${Math.round(scoreVal * 100)}%`;
      trackingRecommendationTitle.textContent = sa.review_recommendation || (sa.status === "completed" ? "PRIORITY_HUMAN_REVIEW" : (sa.status === "no_relevant_entities" ? "INSUFFICIENT_VISUAL_EVIDENCE" : "STANDARD_REVIEW"));
      trackingRecommendationTitle.style.color = "var(--gold-primary)";
      trackingDecisionReason.textContent = sa.decision_reason || "Grounded object and multi-entity tracking active across interval.";
      trackingTriggerReason.textContent = sa.candidate_trigger_reason || "Incident interval tracking active.";
    }

    // 7. Interaction Cues List
    trackingCuesList.innerHTML = "";
    if (cues && cues.length > 0) {
      cues.forEach((cue) => {
        const item = document.createElement("div");
        item.className = "cue-item";
        const cueStart = cue.start_s !== undefined ? cue.start_s : (cue.timestamp || 0.0);
        item.innerHTML = `
          <span class="cue-time-pill">${cueStart.toFixed(1)}s</span>
          <span class="cue-desc">${cue.description || cue.cue_type || "Interaction cue"}</span>
        `;

        item.addEventListener("click", () => {
          if (isClipTrimmedVideo) {
            trackingVideoPlayer.currentTime = Math.max(0, cueStart - clipStartS);
          } else {
            trackingVideoPlayer.currentTime = cueStart;
          }
          trackingVideoPlayer.play();
        });

        trackingCuesList.appendChild(item);
      });
    } else {
      trackingCuesList.innerHTML = '<div class="empty-cue-text">No threat or interaction cues triggered for this interval.</div>';
    }

    // 8. Render Search Results & Observed Appearance Intervals
    renderSearchResults(filteredTracks);

    // 9. Draw Overlay for Current Time
    drawTrackingOverlay();
  }

  function filterTracks(tracks) {
    const showPersons = togglePersons?.checked ?? true;
    const showObjects = toggleObjects?.checked ?? true;
    const showWeapons = toggleWeapons?.checked ?? true;
    const showVehicles = toggleVehicles?.checked ?? true;
    const showOther = toggleOtherObjects?.checked ?? true;

    return tracks.filter((t) => {
      const cname = t.class_name || "";
      const cat = t.category || getObjectCategory(cname);
      const isWeapon = t.is_weapon || t.entity_type === "weapon" || cat === "Weapons";
      const isPerson = t.is_person || cname === "person" || cat === "People";
      const isBelonging = cat === "Personal Belongings";
      const isVehicle = cat === "Vehicles";

      // Category toggle checks
      if (isWeapon && !showWeapons) return false;
      if (isPerson && !showPersons) return false;
      if (isBelonging && !showObjects) return false;
      if (isVehicle && !showVehicles) return false;
      if (!isWeapon && !isPerson && !isBelonging && !isVehicle && !showOther) return false;

      // Status filter check (all, confirmed_only, uncertain_only)
      const isConfirmed = t.detection_status === "confirmed_track" || 
        (t.detection_status === undefined && ((t.detections_count || t.bbox_trajectory?.length || 1) >= 2 || (t.max_confidence || 0) >= 0.70));
      
      if (currentStatusFilter === "confirmed_only" && !isConfirmed) {
        return false;
      }
      if (currentStatusFilter === "uncertain_only" && isConfirmed) {
        return false;
      }

      // Clicked Class filter
      if (currentFilterClass && cname.toLowerCase() !== currentFilterClass.toLowerCase()) {
        return false;
      }

      // Selected Category filter
      if (currentFilterCategory && cat.toLowerCase() !== currentFilterCategory.toLowerCase()) {
        return false;
      }

      // Search Query filter (matches class_name, canonical classes, category, or track_id)
      if (currentSearchQuery) {
        const queryLower = currentSearchQuery.toLowerCase();
        let queryMatches = false;

        // 1. Direct class name match
        if (cname.toLowerCase().includes(queryLower)) queryMatches = true;

        // 2. Canonical classes match from query resolution
        if (activeResolvedQuery && activeResolvedQuery.canonical_classes) {
          if (activeResolvedQuery.canonical_classes.some(c => c.toLowerCase() === cname.toLowerCase())) {
            queryMatches = true;
          }
        }

        // 3. Category match
        if (cat.toLowerCase().includes(queryLower)) queryMatches = true;

        // 4. Track ID match
        if (String(t.track_id).toLowerCase().includes(queryLower)) queryMatches = true;

        if (!queryMatches) return false;
      }

      return true;
    });
  }

  function updateSearchStatusFeedback(allTracks, filteredTracks) {
    if (!trackingSearchStatusBanner) return;

    if (!currentSearchQuery) {
      trackingSearchStatusBanner.style.display = "none";
      if (triggerModeBScanBtn) triggerModeBScanBtn.style.display = "none";
      return;
    }

    trackingSearchStatusBanner.style.display = "flex";
    const q = currentSearchQuery;
    const count = filteredTracks.length;

    const isResolvedSupported = activeResolvedQuery ? activeResolvedQuery.is_supported : false;
    const canonicalList = activeResolvedQuery && activeResolvedQuery.canonical_classes && activeResolvedQuery.canonical_classes.length > 0
      ? activeResolvedQuery.canonical_classes.join(", ")
      : q;

    if (count > 0) {
      const aliasNote = activeResolvedQuery && activeResolvedQuery.matched_aliases && activeResolvedQuery.matched_aliases.length > 0
        ? ` (alias for ${canonicalList})`
        : "";
      trackingSearchStatusText.innerHTML = `Showing <strong>${count}</strong> matching track(s) for "<strong>${escapeHtml(q)}</strong>"${aliasNote}.`;
      trackingSearchResultsBadge.textContent = `${count} match${count === 1 ? '' : 'es'}`;
      trackingSearchResultsBadge.style.color = "var(--gold-primary)";
      if (triggerModeBScanBtn) triggerModeBScanBtn.style.display = "none";
    } else if (isResolvedSupported) {
      trackingSearchStatusText.innerHTML = `Class "<strong>${escapeHtml(canonicalList)}</strong>" is supported by active models, but was <strong>not detected</strong> in this video interval.`;
      trackingSearchResultsBadge.textContent = `0 detections`;
      trackingSearchResultsBadge.style.color = "var(--silver)";
      if (triggerModeBScanBtn) {
        triggerModeBScanBtn.style.display = "inline-flex";
        triggerModeBScanBtn.textContent = `Initiate Targeted Scan for "${canonicalList}" (Mode B)`;
      }
    } else {
      trackingSearchStatusText.innerHTML = `Class "<strong>${escapeHtml(q)}</strong>" is not recognized in YOLOv8 COCO (80 classes) or WeaponShield (3 classes) catalogs.`;
      trackingSearchResultsBadge.textContent = `unsupported class`;
      trackingSearchResultsBadge.style.color = "#ef4444";
      if (triggerModeBScanBtn) triggerModeBScanBtn.style.display = "none";
    }
  }

  function renderSearchResults(tracks) {
    if (!trackingSearchResultsList) return;
    trackingSearchResultsList.innerHTML = "";

    if (trackingResultsCount) {
      trackingResultsCount.textContent = `${tracks.length} track${tracks.length === 1 ? '' : 's'} matching`;
    }

    if (tracks.length === 0) {
      const emptyDiv = document.createElement("div");
      emptyDiv.className = "results-empty-placeholder";
      if (currentSearchQuery) {
        const canonical = activeResolvedQuery && activeResolvedQuery.canonical_classes && activeResolvedQuery.canonical_classes.length > 0
          ? activeResolvedQuery.canonical_classes.join(", ")
          : currentSearchQuery;
        if (activeResolvedQuery && activeResolvedQuery.is_supported) {
          emptyDiv.innerHTML = `
            <div>No matching tracks for <strong>"${escapeHtml(currentSearchQuery)}"</strong> in this clip.</div>
            <div style="font-size: 11px; color: var(--silver); margin-top: 6px;">
              Try adjusting your status filter or run a <strong>Mode B Targeted Scan</strong> in Advanced Settings.
            </div>
          `;
        } else {
          emptyDiv.innerHTML = `
            <div>Class <strong>"${escapeHtml(currentSearchQuery)}"</strong> was not found in supported detection catalogs.</div>
            <div style="font-size: 11px; color: var(--silver); margin-top: 6px;">
              Supported catalogs include YOLOv8 COCO (person, car, phone, backpack, umbrella, etc.) and WeaponShield (Pistol, Rifle, Knife).
            </div>
          `;
        }
      } else {
        emptyDiv.innerHTML = `
          <div>No tracks match the current filter selection in this clip.</div>
        `;
      }
      trackingSearchResultsList.appendChild(emptyDiv);
      return;
    }

    tracks.forEach((track) => {
      const card = document.createElement("div");
      const isSelected = selectedTrackForDetail && selectedTrackForDetail.track_id === track.track_id;
      card.className = `search-result-card ${isSelected ? 'selected-track' : ''}`;
      card.dataset.trackId = track.track_id;

      const tid = track.track_id !== undefined ? track.track_id : "N/A";
      const cname = track.class_name || "unknown";
      const cat = track.category || getObjectCategory(cname);
      const color = getCategoryColor(cat, cname);
      const modelSource = track.model_source || (track.is_weapon ? "weapon_shield_ai" : "coco_yolov8n");
      const detCount = track.detections_count || (track.bbox_trajectory ? track.bbox_trajectory.length : 1);
      const maxConf = track.max_confidence !== undefined ? `${Math.round(track.max_confidence * 100)}%` : "N/A";
      const avgConf = track.avg_confidence !== undefined ? `${Math.round(track.avg_confidence * 100)}%` : "N/A";
      const fseen = track.first_seen_s !== undefined ? track.first_seen_s.toFixed(2) : "0.00";
      const lseen = track.last_seen_s !== undefined ? track.last_seen_s.toFixed(2) : "0.00";
      const dur = track.duration_s !== undefined ? track.duration_s.toFixed(2) : "0.00";

      const isConfirmed = track.detection_status === "confirmed_track" || 
        (track.detection_status === undefined && (detCount >= 2 || (track.max_confidence || 0) >= 0.70));
      const statusBadgeClass = isConfirmed ? "confirmed" : "uncertain";
      const statusBadgeText = isConfirmed ? "Confirmed Track" : "Uncertain Candidate";
      const calThreshold = track.calibrated_threshold !== undefined ? `${Math.round(track.calibrated_threshold * 100)}%` : "35%";

      // Compute intervals if missing
      const intervals = (track.appearance_intervals && track.appearance_intervals.length > 0)
        ? track.appearance_intervals
        : [{
            start_s: track.first_seen_s || 0.0,
            end_s: track.last_seen_s || 0.0,
            duration_s: track.duration_s || 0.0,
            observation_count: detCount,
            is_single_frame: detCount === 1,
            label: detCount === 1 ? `[ ${formatTime(track.first_seen_s || 0.0)} ] (1 frame)` : `[ ${formatTime(track.first_seen_s || 0.0)} — ${formatTime(track.last_seen_s || 0.0)} ] (${detCount} frames)`
          }];

      let intervalsHtml = intervals.map((inv) => {
        const timeDisplay = inv.is_single_frame
          ? `[ ${formatTime(inv.start_s)} ] (1 frame)`
          : `[ ${formatTime(inv.start_s)} — ${formatTime(inv.end_s)} ] (${inv.observation_count} frames, ${inv.duration_s.toFixed(1)}s)`;
        return `
          <button type="button" class="interval-pill-btn" data-start="${inv.start_s.toFixed(2)}" data-end="${inv.end_s.toFixed(2)}" data-tid="${tid}" title="Click to seek video to ${inv.start_s.toFixed(2)}s and highlight object">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:11px; height:11px;">
              <circle cx="12" cy="12" r="10"></circle>
              <polyline points="12 6 12 12 16 14"></polyline>
            </svg>
            ${timeDisplay}
          </button>
        `;
      }).join("");

      card.innerHTML = `
        <div class="result-card-header">
          <div class="result-header-left">
            <span class="badge-dot" style="background-color: ${color}; width: 10px; height: 10px;"></span>
            <strong style="font-size: 13px; letter-spacing: 0.02em;">#${tid} ${cname.toUpperCase()}</strong>
            <span style="font-size: 11px; color: var(--silver); margin-left: 2px;">(${cat})</span>
            <span class="status-badge ${statusBadgeClass}">${statusBadgeText}</span>
          </div>
          <div style="display: flex; align-items: center; gap: 8px;">
            <span style="font-family: var(--font-mono); font-size: 10.5px; color: var(--silver);">${modelSource}</span>
            <span style="font-family: var(--font-mono); font-size: 11px; color: var(--gold-primary); font-weight: bold;">Conf: ${maxConf}</span>
          </div>
        </div>

        <div class="result-meta-row">
          <div class="result-meta-item">First: <strong>${fseen}s</strong></div>
          <div class="result-meta-item">Last: <strong>${lseen}s</strong></div>
          <div class="result-meta-item">Span: <strong>${dur}s</strong></div>
          <div class="result-meta-item">Frames: <strong>${detCount}</strong></div>
          <div class="result-meta-item">Threshold: <strong>${calThreshold}</strong></div>
          <div class="result-meta-item">Avg Conf: <strong>${avgConf}</strong></div>
        </div>

        <div class="appearance-intervals-wrap">
          <div class="intervals-label">Appearance Intervals (${intervals.length} separate interval${intervals.length === 1 ? '' : 's'}):</div>
          <div class="intervals-pills-row">
            ${intervalsHtml}
          </div>
        </div>

        <div style="display: flex; align-items: center; justify-content: flex-end; gap: 8px; margin-top: 2px;">
          <button type="button" class="btn btn-secondary btn-xs seek-first-btn" data-time="${fseen}">
            Seek First Seen (${fseen}s)
          </button>
          <button type="button" class="btn btn-secondary btn-xs seek-last-btn" data-time="${lseen}">
            Seek Last Seen (${lseen}s)
          </button>
          <button type="button" class="btn btn-primary btn-xs inspect-btn">
            INSPECT TRACK
          </button>
        </div>
      `;

      // Event listeners for interval pills
      card.querySelectorAll(".interval-pill-btn").forEach((pBtn) => {
        pBtn.addEventListener("click", (e) => {
          e.stopPropagation();
          const sTime = parseFloat(pBtn.getAttribute("data-start")) || 0;
          selectedTrackForDetail = track;
          seekTrackingVideoToTime(sTime);
          highlightSelectedCard(track.track_id);
          drawTrackingOverlay();
        });
      });

      // Seek buttons
      card.querySelector(".seek-first-btn")?.addEventListener("click", (e) => {
        e.stopPropagation();
        selectedTrackForDetail = track;
        seekTrackingVideoToTime(parseFloat(fseen) || 0);
        highlightSelectedCard(track.track_id);
        drawTrackingOverlay();
      });

      card.querySelector(".seek-last-btn")?.addEventListener("click", (e) => {
        e.stopPropagation();
        selectedTrackForDetail = track;
        seekTrackingVideoToTime(parseFloat(lseen) || 0);
        highlightSelectedCard(track.track_id);
        drawTrackingOverlay();
      });

      // Inspect button
      card.querySelector(".inspect-btn")?.addEventListener("click", (e) => {
        e.stopPropagation();
        openTrackDetailModal(track);
      });

      // Clicking card selects it
      card.addEventListener("click", () => {
        selectedTrackForDetail = track;
        highlightSelectedCard(track.track_id);
        drawTrackingOverlay();
      });

      trackingSearchResultsList.appendChild(card);
    });
  }

  function highlightSelectedCard(trackId) {
    document.querySelectorAll(".search-result-card").forEach(c => {
      c.classList.toggle("selected-track", c.dataset.trackId === String(trackId));
    });
  }

  // Universal Track Inspector Modal (Section 5)
  function openTrackDetailModal(track) {
    if (!trackDetailModal) return;
    selectedTrackForDetail = track;
    selectedTrackObservationIndex = 0;

    const cname = track.class_name || "unknown";
    const cat = track.category || getObjectCategory(cname);
    const color = getCategoryColor(cat, cname);
    const modelSource = track.model_source || (track.is_weapon ? "weapon_shield_ai" : "coco_yolov8n");
    const detCount = track.detections_count || (track.bbox_trajectory ? track.bbox_trajectory.length : 1);
    const avgConf = track.avg_confidence !== undefined ? `${Math.round(track.avg_confidence * 100)}%` : "N/A";
    const maxConf = track.max_confidence !== undefined ? `${Math.round(track.max_confidence * 100)}%` : "N/A";
    const isConfirmed = track.detection_status === "confirmed_track" || 
      (track.detection_status === undefined && (detCount >= 2 || (track.max_confidence || 0) >= 0.70));
    const statusText = isConfirmed ? "CONFIRMED TRACK" : "UNCERTAIN CANDIDATE";
    const statusColor = isConfirmed ? "#10b981" : "#f59e0b";
    const calThresh = track.calibrated_threshold !== undefined ? `${Math.round(track.calibrated_threshold * 100)}%` : "35%";
    const reasonText = track.candidate_reason || (isConfirmed ? "Persistent track with multiple verified observations." : "Single-frame observation or borderline confidence.");

    if (trackDetailTitle) {
      trackDetailTitle.innerHTML = `
        <span class="badge-dot" style="background-color: ${color}; width:10px; height:10px;"></span>
        Track #${track.track_id} [${cname.toUpperCase()}]
        <span style="font-size: 11px; color: var(--silver); font-weight: normal; margin-left: 8px;">(${cat})</span>
        <span class="status-badge ${isConfirmed ? 'confirmed' : 'uncertain'}" style="margin-left: 8px;">${statusText}</span>
      `;
    }

    if (trackDetailGrid) {
      trackDetailGrid.innerHTML = `
        <div class="telemetry-cell">
          <div class="cell-label">TRACK ID</div>
          <div class="cell-val">#${track.track_id}</div>
        </div>
        <div class="telemetry-cell">
          <div class="cell-label">OBJECT CLASS</div>
          <div class="cell-val">${cname}</div>
        </div>
        <div class="telemetry-cell">
          <div class="cell-label">CATEGORY</div>
          <div class="cell-val">${cat}</div>
        </div>
        <div class="telemetry-cell">
          <div class="cell-label">DETECTOR ENGINE</div>
          <div class="cell-val" style="font-family: var(--font-mono); font-size: 11px;">${modelSource}</div>
        </div>
        <div class="telemetry-cell">
          <div class="cell-label">DETECTION STATUS</div>
          <div class="cell-val" style="color: ${statusColor}; font-weight: bold;">${statusText}</div>
        </div>
        <div class="telemetry-cell">
          <div class="cell-label">CALIBRATED THRESHOLD</div>
          <div class="cell-val">${calThresh}</div>
        </div>
        <div class="telemetry-cell">
          <div class="cell-label">FIRST SEEN</div>
          <div class="cell-val">${track.first_seen_s !== undefined ? track.first_seen_s.toFixed(2) + 's' : 'N/A'}</div>
        </div>
        <div class="telemetry-cell">
          <div class="cell-label">LAST SEEN</div>
          <div class="cell-val">${track.last_seen_s !== undefined ? track.last_seen_s.toFixed(2) + 's' : 'N/A'}</div>
        </div>
        <div class="telemetry-cell">
          <div class="cell-label">ACTIVE DURATION</div>
          <div class="cell-val">${track.duration_s !== undefined ? track.duration_s.toFixed(2) + 's' : 'N/A'}</div>
        </div>
        <div class="telemetry-cell">
          <div class="cell-label">OBSERVATIONS (FRAMES)</div>
          <div class="cell-val">${detCount}</div>
        </div>
        <div class="telemetry-cell">
          <div class="cell-label">CONFIDENCE (AVG / MAX)</div>
          <div class="cell-val">${avgConf} / ${maxConf}</div>
        </div>
        <div class="telemetry-cell" style="grid-column: 1 / -1;">
          <div class="cell-label">STATUS REASON &amp; TRIAGE CRITERIA</div>
          <div class="cell-val" style="font-size: 11px; font-weight: normal; color: var(--silver);">${reasonText}</div>
        </div>
      `;
    }

    // Populate intervals list in modal
    if (trackDetailIntervalsList) {
      trackDetailIntervalsList.innerHTML = "";
      const intervals = (track.appearance_intervals && track.appearance_intervals.length > 0)
        ? track.appearance_intervals
        : [{
            start_s: track.first_seen_s || 0.0,
            end_s: track.last_seen_s || 0.0,
            duration_s: track.duration_s || 0.0,
            observation_count: detCount,
            is_single_frame: detCount === 1,
            label: detCount === 1 ? `[ ${formatTime(track.first_seen_s || 0.0)} ] (1 frame)` : `[ ${formatTime(track.first_seen_s || 0.0)} — ${formatTime(track.last_seen_s || 0.0)} ] (${detCount} frames)`
          }];

      intervals.forEach((inv, idx) => {
        const invRow = document.createElement("div");
        invRow.className = "interval-detail-row";
        invRow.style.cssText = "display: flex; align-items: center; justify-content: space-between; padding: 6px 10px; background: rgba(255,255,255,0.03); border-radius: 4px; margin-bottom: 4px; border: 1px solid rgba(255,255,255,0.06);";
        const timeStr = inv.is_single_frame
          ? `Exact frame at ${inv.start_s.toFixed(2)}s`
          : `${inv.start_s.toFixed(2)}s — ${inv.end_s.toFixed(2)}s (${inv.duration_s.toFixed(2)}s duration)`;

        invRow.innerHTML = `
          <div style="font-family: var(--font-mono); font-size: 11px;">
            <strong style="color: var(--cyan-primary);">Interval #${idx + 1}:</strong> ${timeStr}
            <span style="color: var(--silver); margin-left: 6px;">[${inv.observation_count} obs]</span>
          </div>
          <button type="button" class="btn btn-secondary btn-xs seek-inv-btn" data-time="${inv.start_s}">
            Seek Interval
          </button>
        `;

        invRow.querySelector(".seek-inv-btn")?.addEventListener("click", () => {
          seekTrackingVideoToTime(inv.start_s);
        });

        trackDetailIntervalsList.appendChild(invRow);
      });
    }

    // Wire seeker and stepper buttons
    if (seekFirstSeenBtn) {
      seekFirstSeenBtn.textContent = `Seek First Seen (${(track.first_seen_s || 0).toFixed(1)}s)`;
      seekFirstSeenBtn.onclick = () => {
        selectedTrackObservationIndex = 0;
        seekTrackingVideoToTime(track.first_seen_s || 0);
      };
    }

    if (seekLastSeenBtn) {
      seekLastSeenBtn.textContent = `Seek Last Seen (${(track.last_seen_s || 0).toFixed(1)}s)`;
      seekLastSeenBtn.onclick = () => {
        const traj = track.bbox_trajectory || [];
        selectedTrackObservationIndex = traj.length > 0 ? traj.length - 1 : 0;
        seekTrackingVideoToTime(track.last_seen_s || 0);
      };
    }

    if (prevObservationBtn) {
      prevObservationBtn.onclick = () => {
        stepObservation(track, -1);
      };
    }

    if (nextObservationBtn) {
      nextObservationBtn.onclick = () => {
        stepObservation(track, +1);
      };
    }

    trackDetailModal.classList.add("active");
    highlightSelectedCard(track.track_id);
    drawTrackingOverlay();
  }

  function stepObservation(track, delta) {
    const traj = track.bbox_trajectory || [];
    if (traj.length === 0) return;
    selectedTrackObservationIndex = Math.max(0, Math.min(traj.length - 1, selectedTrackObservationIndex + delta));
    const pt = traj[selectedTrackObservationIndex];
    if (pt && pt.t !== undefined) {
      seekTrackingVideoToTime(pt.t);
      if (prevObservationBtn) {
        prevObservationBtn.textContent = `◀ Prev Obs (${selectedTrackObservationIndex + 1}/${traj.length})`;
      }
      if (nextObservationBtn) {
        nextObservationBtn.textContent = `Next Obs (${selectedTrackObservationIndex + 1}/${traj.length}) ▶`;
      }
    }
  }

  function seekTrackingVideoToTime(sourceTime) {
    if (!currentAnalysis) return;
    const clips = currentAnalysis.clips || currentAnalysis.detected_clips || currentAnalysis.incidents || [];
    const clip = clips[currentSelectedClipIndex];
    if (!clip) return;

    const sa = clip.stealing_analysis || clip.tracking_analysis || {};
    const clipStartS = sa.clip_start_s !== undefined ? sa.clip_start_s : (clip.clip_start !== undefined ? clip.clip_start : (clip.start || 0.0));
    const isClipTrimmedVideo = (clip.clip_is_trimmed !== undefined)
      ? Boolean(clip.clip_is_trimmed)
      : (sa.clip_is_trimmed !== undefined
          ? Boolean(sa.clip_is_trimmed)
          : Boolean(clip.clip_url && (!activeVideoUrl || trackingVideoPlayer.src.endsWith(clip.clip_url) || trackingVideoPlayer.src.includes("/static/clips/"))));

    if (isClipTrimmedVideo) {
      trackingVideoPlayer.currentTime = Math.max(0, sourceTime - clipStartS);
    } else {
      trackingVideoPlayer.currentTime = sourceTime;
    }
    trackingVideoPlayer.play();
  }

  closeTrackDetailBtn?.addEventListener("click", () => {
    trackDetailModal?.classList.remove("active");
    selectedTrackForDetail = null;
    drawTrackingOverlay();
  });

  trackDetailModal?.addEventListener("click", (e) => {
    if (e.target === trackDetailModal) {
      trackDetailModal.classList.remove("active");
      selectedTrackForDetail = null;
      drawTrackingOverlay();
    }
  });

  // Targeted Scan Runner (Mode B)
  targetedScanRunBtn?.addEventListener("click", async () => {
    const targetClass = targetedScanClassSelect?.value || "person";
    const startS = parseFloat(targetedScanStart?.value) || 0.0;
    const endS = parseFloat(targetedScanEnd?.value) || 10.0;
    const fps = parseFloat(targetedScanFps?.value) || 10.0;

    if (!activeFileName && !activeVideoUrl) {
      if (targetedScanFeedback) {
        targetedScanFeedback.style.display = "block";
        targetedScanFeedback.style.color = "#ef4444";
        targetedScanFeedback.textContent = "Error: Please upload and analyze a video before initiating a targeted scan.";
      }
      return;
    }

    if (targetedScanFeedback) {
      targetedScanFeedback.style.display = "block";
      targetedScanFeedback.style.color = "var(--gold-primary)";
      targetedScanFeedback.innerHTML = `<span class="spinner" style="display:inline-block; width:12px; height:12px; border:2px solid var(--gold-primary); border-top-color:transparent; border-radius:50%; animation:spin 1s linear infinite; vertical-align:middle; margin-right:6px;"></span> Scanning ${targetClass} from ${startS.toFixed(1)}s to ${endS.toFixed(1)}s at ${fps} FPS...`;
    }
    targetedScanRunBtn.disabled = true;

    try {
      const resp = await fetch("/api/tracking/scan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          video_path: activeFileName || activeVideoUrl,
          target_classes: [targetClass],
          start_s: startS,
          end_s: endS,
          fps: fps
        })
      });

      const res = await resp.json();
      if (resp.ok && res.status === "completed") {
        targetedScanFeedback.style.color = "#10b981";
        targetedScanFeedback.innerHTML = `✓ Scan completed in ${res.scan_time_s.toFixed(2)}s: Examined <strong>${res.frames_examined}</strong> frames (${res.coverage_percentage.toFixed(1)}% coverage). Detected <strong>${res.detections_found}</strong> instances across <strong>${res.tracks_found}</strong> tracks.`;
      } else {
        targetedScanFeedback.style.color = "#ef4444";
        targetedScanFeedback.textContent = `Scan failed: ${res.detail || res.message || "Unknown error"}`;
      }
    } catch (err) {
      targetedScanFeedback.style.color = "#ef4444";
      targetedScanFeedback.textContent = `Scan error: ${err.message}`;
    } finally {
      targetedScanRunBtn.disabled = false;
    }
  });

  // Export Handlers (JSON & CSV)
  exportTrackingJsonBtn?.addEventListener("click", () => {
    const qParams = new URLSearchParams();
    if (currentJobId) qParams.set("job_id", currentJobId);
    if (currentSearchQuery) qParams.set("q", currentSearchQuery);
    if (currentStatusFilter && currentStatusFilter !== "all") qParams.set("status_filter", currentStatusFilter);
    qParams.set("format", "json");

    if (currentJobId) {
      window.open(`/api/tracking/export?${qParams.toString()}`, "_blank");
    } else if (currentAnalysis) {
      const blob = new Blob([JSON.stringify(currentAnalysis, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `argus_tracking_${Date.now()}.json`;
      a.click();
      URL.revokeObjectURL(url);
    }
  });

  exportTrackingCsvBtn?.addEventListener("click", () => {
    const qParams = new URLSearchParams();
    if (currentJobId) qParams.set("job_id", currentJobId);
    if (currentSearchQuery) qParams.set("q", currentSearchQuery);
    if (currentStatusFilter && currentStatusFilter !== "all") qParams.set("status_filter", currentStatusFilter);
    qParams.set("format", "csv");

    if (currentJobId) {
      window.open(`/api/tracking/export?${qParams.toString()}`, "_blank");
    } else if (currentAnalysis) {
      // Direct CSV export fallback
      const clips = currentAnalysis.clips || currentAnalysis.detected_clips || currentAnalysis.incidents || [];
      const rows = [["clip_id", "track_id", "class_name", "category", "first_seen_s", "last_seen_s", "duration_s", "max_confidence", "is_weapon", "status", "intervals"]];
      clips.forEach((clip, cIdx) => {
        const sa = clip.stealing_analysis || clip.tracking_analysis || {};
        const tracks = sa.track_summaries || [];
        tracks.forEach((t) => {
          const invStr = (t.appearance_intervals || []).map(i => `${i.start_s}-${i.end_s}`).join(";");
          rows.push([
            clip.clip_id || `INC-${cIdx + 1}`,
            t.track_id,
            `"${t.class_name}"`,
            `"${t.category || getObjectCategory(t.class_name)}"`,
            t.first_seen_s,
            t.last_seen_s,
            t.duration_s,
            t.max_confidence,
            t.is_weapon ? 1 : 0,
            `"${t.detection_status || 'confirmed_track'}"`,
            `"${invStr}"`
          ]);
        });
      });
      const csvStr = rows.map((r) => r.join(",")).join("\n");
      const blob = new Blob([csvStr], { type: "text/csv" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `argus_tracking_${Date.now()}.csv`;
      a.click();
      URL.revokeObjectURL(url);
    }
  });

  // ------------------------------------------------------------------------
  // Canvas Bounding Box Overlay Drawing & Time Sync
  // ------------------------------------------------------------------------
  function getActiveTrackingSourceTime() {
    if (!currentAnalysis) return 0.0;
    const clips = currentAnalysis.clips || currentAnalysis.detected_clips || currentAnalysis.incidents || [];
    const clip = clips[currentSelectedClipIndex];
    if (!clip) return trackingVideoPlayer.currentTime;

    const sa = clip.stealing_analysis || clip.tracking_analysis || {};
    const clipStartS = sa.clip_start_s !== undefined ? sa.clip_start_s : (clip.clip_start !== undefined ? clip.clip_start : (clip.start || 0.0));
    
    const isClipTrimmedVideo = (clip.clip_is_trimmed !== undefined)
      ? Boolean(clip.clip_is_trimmed)
      : (sa.clip_is_trimmed !== undefined
          ? Boolean(sa.clip_is_trimmed)
          : Boolean(clip.clip_url && (!activeVideoUrl || trackingVideoPlayer.src.endsWith(clip.clip_url) || trackingVideoPlayer.src.includes("/static/clips/"))));

    if (isClipTrimmedVideo) {
      return trackingVideoPlayer.currentTime + clipStartS;
    }
    return trackingVideoPlayer.currentTime;
  }

  function drawTrackingOverlay() {
    if (!trackingCanvas || !trackingVideoPlayer) return;

    const ctx = trackingCanvas.getContext("2d");
    const vidW = trackingVideoPlayer.clientWidth;
    const vidH = trackingVideoPlayer.clientHeight;

    if (trackingCanvas.width !== vidW || trackingCanvas.height !== vidH) {
      trackingCanvas.width = vidW || 640;
      trackingCanvas.height = vidH || 360;
    }

    ctx.clearRect(0, 0, trackingCanvas.width, trackingCanvas.height);

    const sourceTime = getActiveTrackingSourceTime();
    if (trackingTimeIndicator) {
      trackingTimeIndicator.textContent = `Source Time: ${formatTime(sourceTime)} (${sourceTime.toFixed(1)}s)`;
    }

    if (!toggleOverlay?.checked) return;

    const clips = currentAnalysis?.clips || currentAnalysis?.detected_clips || currentAnalysis?.incidents || [];
    const clip = clips[currentSelectedClipIndex];
    if (!clip) return;

    const sa = clip.stealing_analysis || clip.tracking_analysis || {};
    const allTracks = sa.track_summaries || [];
    const tracksToDraw = filterTracks(allTracks);

    const drawTrajectories = toggleTrajectories?.checked ?? true;

    // 1. Draw Trajectories Polyline if enabled
    if (drawTrajectories) {
      tracksToDraw.forEach((track) => {
        const trajectory = track.bbox_trajectory || [];
        if (!trajectory || trajectory.length < 2) return;

        const cat = track.category || getObjectCategory(track.class_name);
        const color = getCategoryColor(cat, track.class_name);
        const isSelected = selectedTrackForDetail && selectedTrackForDetail.track_id === track.track_id;

        ctx.save();
        ctx.beginPath();
        let started = false;

        for (let i = 0; i < trajectory.length; i++) {
          const pt = trajectory[i];
          // Draw up to sourceTime unless selected
          if (!isSelected && pt.t > sourceTime + 0.3) break;

          const cx = ((pt.bbox_norm[0] + pt.bbox_norm[2]) / 2) * trackingCanvas.width;
          const cy = ((pt.bbox_norm[1] + pt.bbox_norm[3]) / 2) * trackingCanvas.height;

          if (!started) {
            ctx.moveTo(cx, cy);
            started = true;
          } else {
            ctx.lineTo(cx, cy);
          }
        }

        ctx.strokeStyle = color;
        ctx.lineWidth = isSelected ? 3.0 : 1.5;
        ctx.setLineDash(isSelected ? [4, 4] : []);
        ctx.globalAlpha = isSelected ? 0.9 : 0.6;
        ctx.stroke();
        ctx.restore();
      });
    }

    // 2. Draw Active Bounding Boxes at sourceTime
    tracksToDraw.forEach((track) => {
      const isSelected = selectedTrackForDetail && selectedTrackForDetail.track_id === track.track_id;
      const fseen = track.first_seen_s !== undefined ? track.first_seen_s : -Infinity;
      const lseen = track.last_seen_s !== undefined ? track.last_seen_s : Infinity;

      // Visibility window
      if (sourceTime < (fseen - 0.25) || sourceTime > (lseen + 0.25)) {
        return;
      }

      const trajectory = track.bbox_trajectory || [];
      if (!trajectory || trajectory.length === 0) return;

      // Find closest trajectory point in time
      let closestPt = null;
      let minDiff = Infinity;
      for (let i = 0; i < trajectory.length; i++) {
        const diff = Math.abs(trajectory[i].t - sourceTime);
        if (diff < minDiff) {
          minDiff = diff;
          closestPt = trajectory[i];
        }
      }

      if (!closestPt || minDiff > 0.45 || !closestPt.bbox_norm) return;

      const norm = closestPt.bbox_norm;
      const x = norm[0] * trackingCanvas.width;
      const y = norm[1] * trackingCanvas.height;
      const w = (norm[2] - norm[0]) * trackingCanvas.width;
      const h = (norm[3] - norm[1]) * trackingCanvas.height;

      const cat = track.category || getObjectCategory(track.class_name);
      const color = getCategoryColor(cat, track.class_name);
      const isWeapon = track.is_weapon || cat === "Weapons";

      // Draw Bounding Box
      ctx.save();
      ctx.lineWidth = isSelected ? 3.5 : 2.5;
      ctx.strokeStyle = color;
      if (isSelected) {
        ctx.shadowColor = color;
        ctx.shadowBlur = 10;
      }
      ctx.strokeRect(x, y, w, h);
      ctx.restore();

      // Draw Label Badge
      const confVal = closestPt.confidence !== undefined ? Math.round(closestPt.confidence * 100) : Math.round(track.max_confidence * 100);
      const label = isWeapon 
        ? `[${track.class_name.toUpperCase()}] #${track.track_id} ${confVal}%` 
        : `#${track.track_id} ${track.class_name} ${confVal}%`;

      ctx.font = "bold 11px Inter, sans-serif";
      const textMetrics = ctx.measureText(label);
      const labelW = textMetrics.width + 10;
      const labelH = 18;

      ctx.fillStyle = color;
      ctx.fillRect(x, Math.max(0, y - labelH), labelW, labelH);

      ctx.fillStyle = "#ffffff";
      ctx.textBaseline = "middle";
      ctx.fillText(label, x + 5, Math.max(0, y - labelH) + labelH / 2);
    });
  }

  // Hook sync listeners
  trackingVideoPlayer?.addEventListener("timeupdate", drawTrackingOverlay);
  trackingVideoPlayer?.addEventListener("seeked", drawTrackingOverlay);
  trackingVideoPlayer?.addEventListener("play", () => {
    function loop() {
      if (!trackingVideoPlayer.paused && !trackingVideoPlayer.ended) {
        drawTrackingOverlay();
        animationFrameId = requestAnimationFrame(loop);
      }
    }
    loop();
  });
  trackingVideoPlayer?.addEventListener("pause", () => {
    if (animationFrameId) cancelAnimationFrame(animationFrameId);
    drawTrackingOverlay();
  });
  trackingVideoPlayer?.addEventListener("error", () => {
    trackingEmptyBanner.style.display = "flex";
    trackingEmptyBannerText.textContent = "Evidence video playback error: browser was unable to decode the media stream. Original recording remains available.";
  });

  window.addEventListener("resize", drawTrackingOverlay);
  toggleOverlay?.addEventListener("change", drawTrackingOverlay);
  togglePersons?.addEventListener("change", updateFilterStatusAndRender);
  toggleObjects?.addEventListener("change", updateFilterStatusAndRender);
  toggleWeapons?.addEventListener("change", updateFilterStatusAndRender);
  toggleVehicles?.addEventListener("change", updateFilterStatusAndRender);
  toggleOtherObjects?.addEventListener("change", updateFilterStatusAndRender);
  toggleTrajectories?.addEventListener("change", drawTrackingOverlay);

  // ------------------------------------------------------------------------
  // Probability Breakdown Accordion
  // ------------------------------------------------------------------------
  function renderProbabilityBreakdown(probs) {
    breakdownContent.innerHTML = "";
    const entries = Object.entries(probs).sort((a, b) => b[1] - a[1]);

    entries.forEach(([className, probVal]) => {
      const pct = (probVal * 100).toFixed(1);
      const row = document.createElement("div");
      row.className = "breakdown-row";
      row.innerHTML = `
        <span class="breakdown-class-name" title="${className}">${className}</span>
        <div class="breakdown-bar-track">
          <div class="breakdown-bar-fill" style="width: ${pct}%;"></div>
        </div>
        <span class="breakdown-pct">${pct}%</span>
      `;
      breakdownContent.appendChild(row);
    });
  }

  // ------------------------------------------------------------------------
  // Action Buttons & Navigation Wiring
  // ------------------------------------------------------------------------
  function updateTopNav(activeItem) {
    const navBtnAnalyze = document.getElementById("navBtnAnalyze");
    const navBtnOverview = document.getElementById("navBtnOverview");
    const navBtnClips = document.getElementById("navBtnClips");
    const navBtnTracking = document.getElementById("navBtnTracking");
    [navBtnAnalyze, navBtnOverview, navBtnClips, navBtnTracking].forEach(b => {
      if (b) b.classList.remove("active");
    });
    if (activeItem === "analyze" && navBtnAnalyze) navBtnAnalyze.classList.add("active");
    if (activeItem === "overview" && navBtnOverview) navBtnOverview.classList.add("active");
    if (activeItem === "clips" && navBtnClips) navBtnClips.classList.add("active");
    if (activeItem === "tracking" && navBtnTracking) navBtnTracking.classList.add("active");
  }

  // Centered Floating Top Navigation
  const navBrandLogo = document.getElementById("navBrandLogo");
  const navBtnAnalyze = document.getElementById("navBtnAnalyze");
  const navBtnOverview = document.getElementById("navBtnOverview");
  const navBtnClips = document.getElementById("navBtnClips");
  const navBtnTracking = document.getElementById("navBtnTracking");
  const navBtnTelemetry = document.getElementById("navBtnTelemetry");
  const telemetryModal = document.getElementById("telemetryModal");
  const closeTelemetryBtn = document.getElementById("closeTelemetryBtn");
  const closeTelemetryBtn2 = document.getElementById("closeTelemetryBtn2");

  if (navBrandLogo) {
    navBrandLogo.addEventListener("click", () => {
      resetToUpload();
      updateTopNav("analyze");
    });
  }

  if (navBtnAnalyze) {
    navBtnAnalyze.addEventListener("click", () => {
      resetToUpload();
      updateTopNav("analyze");
    });
  }

  if (navBtnOverview) {
    navBtnOverview.addEventListener("click", () => {
      if (currentAnalysis) {
        showScreen(resultsScreen);
        switchView("overview");
        updateTopNav("overview");
      } else {
        const drop = document.getElementById("dropZone");
        if (drop) drop.scrollIntoView({ behavior: "smooth" });
      }
    });
  }

  if (navBtnClips) {
    navBtnClips.addEventListener("click", () => {
      if (currentAnalysis) {
        showScreen(resultsScreen);
        switchView("flagged-clips");
        updateTopNav("clips");
      } else {
        const drop = document.getElementById("dropZone");
        if (drop) drop.scrollIntoView({ behavior: "smooth" });
      }
    });
  }

  if (navBtnTracking) {
    navBtnTracking.addEventListener("click", () => {
      if (currentAnalysis) {
        showScreen(resultsScreen);
        switchView("tracking");
        updateTopNav("tracking");
      } else {
        const drop = document.getElementById("dropZone");
        if (drop) drop.scrollIntoView({ behavior: "smooth" });
      }
    });
  }

  // Telemetry Modal
  if (navBtnTelemetry && telemetryModal) {
    navBtnTelemetry.addEventListener("click", () => {
      telemetryModal.classList.add("active");
    });
  }
  if (closeTelemetryBtn && telemetryModal) {
    closeTelemetryBtn.addEventListener("click", () => {
      telemetryModal.classList.remove("active");
    });
  }
  if (closeTelemetryBtn2 && telemetryModal) {
    closeTelemetryBtn2.addEventListener("click", () => {
      telemetryModal.classList.remove("active");
    });
  }

  // Hero Analyze CTA button
  const heroAnalyzeActionBtn = document.getElementById("heroAnalyzeActionBtn");
  if (heroAnalyzeActionBtn) {
    heroAnalyzeActionBtn.addEventListener("click", () => {
      const drop = document.getElementById("dropZone");
      if (drop) {
        drop.scrollIntoView({ behavior: "smooth" });
        drop.focus();
      }
    });
  }

  // Benchmark Quick Sample Triggers
  async function triggerSampleAnalysis(sampleFilename) {
    try {
      showError("");
      uploadErrorBanner.style.display = "none";
      const resp = await fetch(`/static/samples/${sampleFilename}`);
      if (!resp.ok) {
        throw new Error(`Failed to load benchmark sample video: ${resp.statusText}`);
      }
      const blob = await resp.blob();
      const file = new File([blob], sampleFilename, { type: blob.type || "video/mp4" });
      handleFileSelect(file);
    } catch (err) {
      console.error("Error triggering sample:", err);
      showError(`Could not load benchmark sample ${sampleFilename}: ${err.message}`);
    }
  }

  const sampleFightingBtn = document.getElementById("sampleFightingBtn");
  const sampleNormalBtn = document.getElementById("sampleNormalBtn");
  const sampleStealingBtn = document.getElementById("sampleStealingBtn");

  if (sampleFightingBtn) {
    sampleFightingBtn.addEventListener("click", () => triggerSampleAnalysis("upload_b7c54c67.mp4"));
  }
  if (sampleNormalBtn) {
    sampleNormalBtn.addEventListener("click", () => triggerSampleAnalysis("Normal_Videos_247_x264.mp4"));
  }
  if (sampleStealingBtn) {
    sampleStealingBtn.addEventListener("click", () => triggerSampleAnalysis("Stealing050_x264.mp4"));
  }

  analyzeAnotherBtn.addEventListener("click", () => {
    resetToUpload();
    updateTopNav("analyze");
  });

  exportReportBtn.addEventListener("click", () => {
    if (!currentAnalysis) return;

    const report = {
      report_title: "ARGUS Surveillance Anomaly Intelligence Audit",
      platform: "ARGUS Panoptes-Core v2.4",
      timestamp: new Date().toISOString(),
      job_id: currentJobId,
      video_url: activeVideoUrl,
      video_name: activeFileName,
      verdict: currentAnalysis.verdict,
      top_class: currentAnalysis.top_class,
      confidence: currentAnalysis.confidence,
      severity: currentAnalysis.severity,
      summary: currentAnalysis.summary,
      metadata: currentAnalysis.metadata,
      detected_clips: currentAnalysis.clips || currentAnalysis.detected_clips,
      probabilities_14_class: currentAnalysis.probabilities_14_class,
      timeline: currentAnalysis.timeline,
      model_used: currentAnalysis.model_used,
      engine_mode: currentAnalysis.engine_mode,
      device: currentAnalysis.device,
    };

    const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `argus_audit_report_${activeFileName || 'video'}_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  });

  // Dynamic Telemetry Annotation update from /api/status
  async function fetchSystemTelemetry() {
    try {
      const res = await fetch("/api/status");
      if (!res.ok) return;
      const data = await res.json();
      const cornerElem = document.querySelector(".corner-top-right");
      if (cornerElem && data) {
        const devName = data.device_name || (data.device ? data.device.toUpperCase() : "CPU");
        const devStr = data.device ? data.device.toUpperCase() : "CPU";
        cornerElem.textContent = `TELEMETRY // ${devStr} • ${devName}`;
      }
    } catch (e) {
      console.warn("Telemetry fetch error:", e);
    }
  }
  fetchSystemTelemetry();

  // Handle URL query parameters for direct audit review: ?sample=stealing050&view=overview|clips|tracking
  const urlParams = new URLSearchParams(window.location.search);
  const sampleParam = urlParams.get("sample");
  const viewParam = urlParams.get("view");
  if (sampleParam === "stealing050" || sampleParam === "sample") {
    fetch("/static/samples/stealing050_api_result.json")
      .then(res => res.json())
      .then(data => {
        normalizeAndDisplayJsonReport(data, "Stealing050_x264.mp4");
        if (viewParam) {
          setTimeout(() => {
            const mappedView = viewParam === "clips" ? "flagged-clips" : viewParam;
            switchView(mappedView);
            updateTopNav(viewParam);
          }, 100);
        }
      })
      .catch(err => console.error("Could not autoload sample report:", err));
  }
});
