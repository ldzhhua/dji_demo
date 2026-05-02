const state = {
  health: null,
  overview: null,
  jobs: [],
  selectedAction: "dataset",
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
            </div>
            <p>${job.message}</p>
            ${job.result ? `<pre>${JSON.stringify(job.result, null, 2)}</pre>` : ""}
          </div>
          <div class="job-side">
            <strong>${job.progress}%</strong>
            <span>${formatDate(job.updated_at)}</span>
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
  }[kind] || kind;
}

function showToast(message, isError = false) {
  els.toast.textContent = message;
  els.toast.classList.toggle("error", isError);
  els.toast.hidden = false;
  window.setTimeout(() => {
    els.toast.hidden = true;
  }, 3200);
}

async function requestJson(url, options) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    throw new Error(`请求失败：${response.status}`);
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
  } catch (error) {
    setApiState(false);
    showToast(error.message, true);
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

els.tabs.forEach((button) => {
  button.addEventListener("click", () => switchAction(button.dataset.action));
});

els.forms.forEach((form) => {
  const button = form.querySelector("button[type='submit']");
  button.dataset.label = button.textContent;
  form.addEventListener("submit", submitAction);
});

document.querySelector("#refresh-button").addEventListener("click", refresh);

switchAction(state.selectedAction);
refresh();
