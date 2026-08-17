document.addEventListener("DOMContentLoaded", function () {
  setupUploadForm();

  requestAnimationFrame(() => {
    drawVehicleCountChart();
  });
});

function setupUploadForm() {
  const form = document.getElementById('uploadForm');
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('videoInput');
  const fileNameLabel = document.getElementById('fileNameLabel');
  const overlay = document.getElementById('loadingOverlay');

  if (!form) return;

  function showFileName() {
    if (fileInput.files && fileInput.files.length > 0) {
      fileNameLabel.textContent = 'Selected: ' + fileInput.files[0].name;
    }
  }

  fileInput.addEventListener('change', showFileName);

  ['dragenter', 'dragover'].forEach(evt => {
    dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(evt => {
    dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove('dragover');
    });
  });

  dropZone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      fileInput.files = files;
      showFileName();
    }
  });

  form.addEventListener('submit', (e) => {
    if (!fileInput.files || fileInput.files.length === 0) {
      e.preventDefault();
      fileNameLabel.textContent = 'Please choose a video file first.';
      return;
    }
    overlay.classList.remove('hidden');
  });
}

function drawDensityChart() {
  const canvas = document.getElementById('densityChart');
  if (!canvas || typeof frameTrend === 'undefined' || frameTrend.length === 0) return;

  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const cssWidth = canvas.clientWidth || 900;
  const cssHeight = 260;
  canvas.width = cssWidth * dpr;
  canvas.height = cssHeight * dpr;
  ctx.scale(dpr, dpr);

  const padding = { top: 16, right: 20, bottom: 30, left: 40 };
  const plotW = cssWidth - padding.left - padding.right;
  const plotH = cssHeight - padding.top - padding.bottom;

  const counts = frameTrend.map(p => p[1]);
  const maxCount = Math.max(1, ...counts);

  ctx.clearRect(0, 0, cssWidth, cssHeight);

  // Axes
  ctx.strokeStyle = 'rgba(20,21,26,0.28)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padding.left, padding.top);
  ctx.lineTo(padding.left, padding.top + plotH);
  ctx.lineTo(padding.left + plotW, padding.top + plotH);
  ctx.stroke();

  // Y-axis labels (0, mid, max)
  ctx.fillStyle = '#45473F';
  ctx.font = '11px "DM Mono", monospace';
  ctx.textAlign = 'right';
  [0, 0.5, 1].forEach(frac => {
    const val = Math.round(maxCount * frac);
    const y = padding.top + plotH - frac * plotH;
    ctx.fillText(String(val), padding.left - 8, y + 4);
  });

  // Line
  ctx.strokeStyle = '#C68A1E';
  ctx.lineWidth = 2;
  ctx.beginPath();
  frameTrend.forEach((point, i) => {
    const x = padding.left + (i / Math.max(1, frameTrend.length - 1)) * plotW;
    const y = padding.top + plotH - (point[1] / maxCount) * plotH;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Fill under line
  ctx.lineTo(padding.left + plotW, padding.top + plotH);
  ctx.lineTo(padding.left, padding.top + plotH);
  ctx.closePath();
  ctx.fillStyle = 'rgba(198,138,30,0.10)';
  ctx.fill();

  // X-axis label
  ctx.fillStyle = '#45473F';
  ctx.textAlign = 'center';
  ctx.fillText('Frame progression →', padding.left + plotW / 2, cssHeight - 6);
}


function drawVehicleCountChart() {
  const canvas = document.getElementById("vehicleCountChart");

  if (!canvas) {
    console.log("Vehicle count chart canvas not found.");
    return;
  }

  if (
    typeof frameTrend === "undefined" ||
    !Array.isArray(frameTrend) ||
    frameTrend.length === 0
  ) {
    console.log("No vehicle count data available.");
    return;
  }

  const ctx = canvas.getContext("2d");

  const width = canvas.clientWidth || 900;
  const height = 320;
  const dpr = window.devicePixelRatio || 1;

  canvas.width = width * dpr;
  canvas.height = height * dpr;

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const padding = {
    top: 25,
    right: 25,
    bottom: 45,
    left: 50
  };

  const chartWidth =
    width - padding.left - padding.right;

  const chartHeight =
    height - padding.top - padding.bottom;

  // frameTrend format:
  // [[frame_number, vehicle_count], ...]

  const data = frameTrend.map(point => ({
    frame: Number(point[0]),
    vehicles: Number(point[1])
  }));

  const maxVehicles = Math.max(
    1,
    ...data.map(p => p.vehicles)
  );

  // Clear
  ctx.clearRect(0, 0, width, height);

  // Background
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  // -----------------------------
  // Grid
  // -----------------------------

  ctx.strokeStyle = "rgba(20,21,26,0.10)";
  ctx.lineWidth = 1;

  const gridCount = 5;

  for (let i = 0; i <= gridCount; i++) {

    const y =
      padding.top +
      chartHeight -
      (i / gridCount) * chartHeight;

    ctx.beginPath();

    ctx.moveTo(
      padding.left,
      y
    );

    ctx.lineTo(
      padding.left + chartWidth,
      y
    );

    ctx.stroke();
  }

  // -----------------------------
  // Y axis
  // -----------------------------

  ctx.fillStyle = "#45473F";
  ctx.font = '11px "DM Mono", monospace';
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";

  for (let i = 0; i <= gridCount; i++) {

    const value = Math.round(
      (maxVehicles * i) / gridCount
    );

    const y =
      padding.top +
      chartHeight -
      (i / gridCount) * chartHeight;

    ctx.fillText(
      value.toString(),
      padding.left - 8,
      y
    );
  }

  // -----------------------------
  // X axis
  // -----------------------------

  ctx.strokeStyle =
    "rgba(20,21,26,0.35)";

  ctx.beginPath();

  ctx.moveTo(
    padding.left,
    padding.top
  );

  ctx.lineTo(
    padding.left,
    padding.top + chartHeight
  );

  ctx.lineTo(
    padding.left + chartWidth,
    padding.top + chartHeight
  );

  ctx.stroke();

  // -----------------------------
  // Vehicle count line
  // -----------------------------

  ctx.beginPath();

  data.forEach((point, index) => {

    const x =
      padding.left +
      (index / Math.max(1, data.length - 1)) *
        chartWidth;

    const y =
      padding.top +
      chartHeight -
      (point.vehicles / maxVehicles) *
        chartHeight;

    if (index === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });

  ctx.strokeStyle = "#C68A1E";
  ctx.lineWidth = 3;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";

  ctx.stroke();

  // -----------------------------
  // X labels
  // -----------------------------

  ctx.fillStyle = "#45473F";
  ctx.font = '10px "DM Mono", monospace';
  ctx.textAlign = "center";
  ctx.textBaseline = "top";

  const labels = Math.min(6, data.length);

  for (let i = 0; i < labels; i++) {

    const index =
      Math.round(
        (i / Math.max(1, labels - 1)) *
        (data.length - 1)
      );

    const x =
      padding.left +
      (index / Math.max(1, data.length - 1)) *
        chartWidth;

    ctx.fillText(
      "Frame " + data[index].frame,
      x,
      padding.top + chartHeight + 10
    );
  }

  // -----------------------------
  // Axis titles
  // -----------------------------

  ctx.fillStyle = "#45473F";
  ctx.font = '11px "DM Mono", monospace';

  ctx.textAlign = "center";

  ctx.fillText(
    "Video Frame",
    padding.left + chartWidth / 2,
    height - 8
  );

  ctx.save();

  ctx.translate(
    13,
    padding.top + chartHeight / 2
  );

  ctx.rotate(-Math.PI / 2);

  ctx.fillText(
    "Vehicle Count",
    0,
    0
  );

  ctx.restore();

  console.log(
    "Vehicle Count Over Time chart rendered:",
    data.length,
    "points"
  );
}