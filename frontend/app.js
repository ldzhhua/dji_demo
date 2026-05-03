const state = {
  health: null,
  overview: null,
  jobs: [],
  selectedAction: "dataset",
  pollTimer: null,
};

const els = {
  apiState: document.querySelector("#api-state"),
  jobCount: document.querySelector("#job-count"),
  completedCount: document.querySelector("#completed-count"),
  runningCount: document.querySelector("#running-count"),
  failedCount: document.querySelector("#failed-count"),
  opencdBadge: document.querySelector("#opencd-badge"),
  jobList: document.querySelector("#job-list"),
  emptyJobs: document.querySelector("#empty-jobs"),
  toast: document.querySelector("#toast"),
  tabs: document.querySelectorAll(".tab-button"),
  forms: document.querySelectorAll(".action-form"),
  demoDatasetButton: document.querySelector("#demo-dataset-button"),
  demoPipelineButton: document.querySelector("#demo-pipeline-button"),
  clearJobsButton: document.querySelector("#clear-jobs-button"),
  cancelClearButton: document.querySelector("#cancel-clear-button"),
  confirmClearButton: document.querySelector("#confirm-clear-button"),
  confirmDialog: document.querySelector("#confirm-modal"),
};

const actionConfig = {
  dataset: {
    endpoint: "/api/datasets",
    success: "数据集拆分任务已完成",
    fields: (form) => ({
      input_dir: form.input_dir.value.trim(),
      output_dir: form.output_dir.value.trim(),
      val_ratio: Number(form.val_ratio.value),
      seed: form.seed.value ? Number(form.seed.value) : null,
    }),
  },
  training: {
    endpoint: "/api/training",
    success: "训练任务已提交",
    fields: (form) => ({
      config_path: form.config_path.value.trim(),
      work_dir: form.work_dir.value.trim(),
    }),
  },
  inference: {
    endpoint: "/api/inference",
    success: "推理任务已提交",
    fields: (form) => ({
      model_path: form.model_path.value.trim(),
      input_dir: form.input_dir.value.trim(),
      output_dir: form.output_dir.value.trim(),
    }),
  },
};

function formatDate(value) {
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

function setApiState(online) {
  els.apiState.textContent = online ? "API 在线" : "API 离线";
  els.apiState.classList.toggle("offline", !online);
}

function renderOverview() {
  const metrics = state.overview?.metrics || {};
  els.jobCount.textContent = metrics.total_jobs ?? 0;
  els.completedCount.textContent = metrics.completed_jobs ?? 0;
  els.runningCount.textContent = metrics.running_jobs ?? 0;
  els.failedCount.textContent = metrics.failed_jobs ?? 0;

  const available = Boolean(state.health?.opencd_available);
  els.opencdBadge.textContent = available ? "OpenCD 已连接" : "OpenCD 未安装";
  els.opencdBadge.classList.toggle("warning", !available);
}

function renderJobs() {
  els.emptyJobs.hidden = state.jobs.length > 0;
  els.jobList.innerHTML = state.jobs
    .map(
      (job) => `
        <article class="job-card">
          <div>
            <div class="job-heading">
              <span class="status-dot ${job.status}"></span>
              <h3>${job.title}</h3>
              <span class="job-kind">${kindLabel(job.kind)}</span>
              <span class="badge ${job.status}">${statusLabel(job.status)}</span>
            </div>
            <p>${job.message}</p>
            ${job.result ? `<pre>${JSON.stringify(job.result, null, 2)}</pre>` : ""}
          </div>
          <div class="job-side">
            <strong>${job.progress}%</strong>
            <span>${formatDate(job.updated_at)}</span>
            <button class="icon-button" type="button" data-delete-job="${job.id}">删除</button>
          </div>
        </article>
      `,
    )
    .join("");
}

function kindLabel(kind) {
  return {
    dataset: "数据集",
    training: "训练",
    inference: "推理",
    pipeline: "完整流水线",
  }[kind] || kind;
}

function statusLabel(status) {
  return {
    queued: "排队",
    running: "运行中",
    completed: "完成",
    failed: "失败",
  }[status] || status;
}

function showToast(message, isError = false) {
  els.toast.textContent = message;
  els.toast.classList.toggle("error", isError);
  els.toast.hidden = false;
  els.toast.classList.add("show");
  window.setTimeout(() => {
    els.toast.classList.remove("show");
    els.toast.hidden = true;
  }, 3200);
}

async function requestJson(url, options) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    let message = `请求失败：${response.status}`;
    try {
      const error = await response.json();
      message = error.detail || message;
    } catch {
      // Keep the generic HTTP message when the server does not return JSON.
    }
    throw new Error(message);
  }
  return response.json();
}

async function refresh() {
  try {
    const [health, overview, jobs] = await Promise.all([
      requestJson("/api/health"),
      requestJson("/api/overview"),
      requestJson("/api/jobs"),
    ]);
    state.health = health;
    state.overview = overview;
    state.jobs = jobs;
    setApiState(true);
    renderOverview();
    renderJobs();
    updatePolling();
  } catch (error) {
    setApiState(false);
    showToast(error.message, true);
  }
}

function updatePolling() {
  const hasActiveJobs = state.jobs.some((job) =>
    ["queued", "running"].includes(job.status),
  );
  if (hasActiveJobs && !state.pollTimer) {
    state.pollTimer = window.setInterval(refresh, 1600);
  }
  if (!hasActiveJobs && state.pollTimer) {
    window.clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
}

function switchAction(action) {
  state.selectedAction = action;
  els.tabs.forEach((button) => {
    button.classList.toggle("active", button.dataset.action === action);
  });
  els.forms.forEach((form) => {
    form.hidden = form.dataset.action !== action;
  });
}

async function submitAction(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const action = form.dataset.action;
  const config = actionConfig[action];
  const button = form.querySelector("button[type='submit']");
  button.disabled = true;
  button.textContent = "处理中...";

  try {
    const payload = config.fields(form);
    const job = await requestJson(config.endpoint, {
      method: "POST",
      body: JSON.stringify(payload),
    });
    showToast(job.status === "failed" ? job.message : config.success, job.status === "failed");
    await refresh();
  } catch (error) {
    showToast(error.message, true);
  } finally {
    button.disabled = false;
    button.textContent = button.dataset.label;
  }
}

async function prepareDemoDataset() {
  els.demoDatasetButton.disabled = true;
  try {
    const demo = await requestJson("/api/demo-dataset", {
      method: "POST",
      body: JSON.stringify({ image_count: 8, val_ratio: 0.25, seed: 42 }),
    });
    const form = document.querySelector(".action-form[data-action='dataset']");
    form.input_dir.value = demo.input_dir;
    form.output_dir.value = demo.output_dir;
    form.val_ratio.value = demo.val_ratio;
    form.seed.value = demo.seed ?? "";
    switchAction("dataset");
    showToast("演示数据已生成，可直接创建数据集");
  } catch (error) {
    showToast(error.message, true);
  } finally {
    els.demoDatasetButton.disabled = false;
  }
}

async function runDemoPipeline() {
  els.demoPipelineButton.disabled = true;
  els.demoPipelineButton.textContent = "运行中...";
  try {
    await requestJson("/api/demo-pipeline", {
      method: "POST",
      body: JSON.stringify({ image_count: 8, val_ratio: 0.25, seed: 42 }),
    });
    showToast("完整演示流水线已启动");
    await refresh();
  } catch (error) {
    showToast(error.message, true);
  } finally {
    els.demoPipelineButton.disabled = false;
    els.demoPipelineButton.textContent = "一键完整跑通";
  }
}

async function clearJobs() {
  await requestJson("/api/jobs", { method: "DELETE" });
  closeConfirmDialog();
  showToast("任务记录已清空");
  await refresh();
}

async function deleteJob(jobId) {
  await requestJson(`/api/jobs/${jobId}`, { method: "DELETE" });
  showToast("任务已删除");
  await refresh();
}

function openConfirmDialog() {
  els.confirmDialog.hidden = false;
}

function closeConfirmDialog() {
  els.confirmDialog.hidden = true;
}

els.tabs.forEach((button) => {
  button.addEventListener("click", () => switchAction(button.dataset.action));
});

els.forms.forEach((form) => {
  const button = form.querySelector("button[type='submit']");
  button.dataset.label = button.textContent;
  form.addEventListener("submit", submitAction);
});

document.querySelector("#refresh-button").addEventListener("click", refresh);
els.demoDatasetButton.addEventListener("click", prepareDemoDataset);
els.demoPipelineButton.addEventListener("click", runDemoPipeline);
els.clearJobsButton.addEventListener("click", openConfirmDialog);
els.cancelClearButton.addEventListener("click", closeConfirmDialog);
els.confirmClearButton.addEventListener("click", clearJobs);
els.confirmDialog.addEventListener("click", (event) => {
  if (event.target === els.confirmDialog) {
    closeConfirmDialog();
  }
});
els.jobList.addEventListener("click", (event) => {
  const button = event.target.closest("[data-delete-job]");
  if (button) {
    deleteJob(button.dataset.deleteJob);
  }
});

switchAction(state.selectedAction);
refresh();
