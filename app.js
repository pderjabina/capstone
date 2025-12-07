// ------------------------------
// Global State
// ------------------------------
let model = null;
let tokenizerConfig = null;
let wordIndex = {};
let maxLen = 0;
let vocabSize = null;
let oovIndex = 1;

// Chart instances (to avoid duplicate rendering)
let classChart, lengthChart, missingChart;

// ------------------------------
// Initialization
// ------------------------------
document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    initJobForm();
    initApp();
});

async function initApp() {
    const modelStatus = document.getElementById("model-status");
    try {
        modelStatus.textContent = "Loading model and tokenizer…";

        await Promise.all([loadModel(), loadTokenizer()]);

        modelStatus.textContent = "Model and tokenizer loaded – ready!";
        modelStatus.classList.remove("status-loading");
        modelStatus.classList.add("status-ready");

        document.getElementById("predict-btn").disabled = false;
    } catch (err) {
        console.error("Initialization error:", err);
        modelStatus.textContent = "Error loading model/tokenizer. See console.";
        modelStatus.classList.remove("status-loading");
        modelStatus.classList.add("status-error");
    }

    // EDA init is independent; if it упадёт – не ломаем JobCheck
    initEDA();
}

// ------------------------------
// Tabs
// ------------------------------
function initTabs() {
    const buttons = document.querySelectorAll(".tab-button");
    const contents = {
        "job-check": document.getElementById("tab-job-check"),
        "eda": document.getElementById("tab-eda"),
    };

    buttons.forEach((btn) => {
        btn.addEventListener("click", () => {
            const tab = btn.dataset.tab;

            buttons.forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");

            Object.entries(contents).forEach(([name, el]) => {
                if (name === tab) {
                    el.classList.add("active");
                } else {
                    el.classList.remove("active");
                }
            });
        });
    });
}

// ------------------------------
// Model & Tokenizer loading
// ------------------------------
async function loadModel() {
    // Path relative to index.html / GitHub Pages root
    model = await tf.loadLayersModel("model/model.json");
    console.log("Model loaded");
}

async function loadTokenizer() {
    const res = await fetch("model/frontend_config.json");
    if (!res.ok) {
        throw new Error("Cannot load frontend_config.json");
    }
    const cfg = await res.json();

    tokenizerConfig = cfg;
    wordIndex = cfg.word_index || {};
    maxLen = cfg.max_len || 200;
    vocabSize = cfg.vocab_size || null;

    // ВАЖНО: используем oov_index из конфига, если он есть
    if (typeof cfg.oov_index === "number") {
        oovIndex = cfg.oov_index;
    } else {
        // fallback: 1 — стандартный индекс OOV в Keras
        oovIndex = 1;
    }

    console.log("Tokenizer loaded", { maxLen, vocabSize, oovIndex });
}

// ------------------------------
// Text preprocessing / tokenization
// ------------------------------
function buildFullTextFromForm(formData) {
    const fields = [
        "title",
        "company_profile",
        "description",
        "requirements",
        "benefits",
        "location",
        "salary_range",
        "employment_type",
        "industry",
    ];

    const parts = fields.map((name) => (formData.get(name) || "").trim());
    const fullText = parts.join(" ").replace(/\s+/g, " ").trim();

    return fullText;
}

function normalizeText(text) {
    // lower-case, remove non-alphanumeric (simple English-focused preprocessing)
    return text
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, " ")
        .replace(/\s+/g, " ")
        .trim();
}

function textToSequence(text) {
    const norm = normalizeText(text);
    if (!norm) {
        return new Array(maxLen).fill(0);
    }

    const tokens = norm.split(" ");
    const seq = [];

    for (const t of tokens) {
        let idx = wordIndex[t];
        if (!idx || (vocabSize && idx >= vocabSize)) {
            idx = oovIndex;
        }
        seq.push(idx);
    }

    // post-padding / truncation
    if (seq.length > maxLen) {
        return seq.slice(0, maxLen);
    } else if (seq.length < maxLen) {
        const padded = seq.slice();
        while (padded.length < maxLen) {
            padded.push(0);
        }
        return padded;
    }
    return seq;
}

// ------------------------------
// Job Check form
// ------------------------------
function initJobForm() {
    const form = document.getElementById("job-form");
    const predictBtn = document.getElementById("predict-btn");
    const clearBtn = document.getElementById("clear-btn");

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        if (!model || !tokenizerConfig) {
            alert("Model is not ready yet. Please wait a bit.");
            return;
        }

        try {
            predictBtn.disabled = true;
            predictBtn.textContent = "Predicting…";

            const formData = new FormData(form);
            const fullText = buildFullTextFromForm(formData);

            if (!fullText) {
                alert("Please fill at least some fields of the job posting.");
                return;
            }

            const sequence = textToSequence(fullText);
            const inputTensor = tf.tensor2d([sequence], [1, maxLen]);
            const outputTensor = model.predict(inputTensor);
            const data = await outputTensor.data();
            const prob = data[0];

            inputTensor.dispose();
            outputTensor.dispose();

            renderPrediction(prob);
        } catch (err) {
            console.error("Prediction error:", err);
            alert("Prediction error. See console for details.");
        } finally {
            predictBtn.disabled = false;
            predictBtn.textContent = "Predict";
        }
    });

    clearBtn.addEventListener("click", () => {
        form.reset();
        const result = document.getElementById("prediction-result");
        result.classList.add("hidden");
    });
}

function renderPrediction(prob) {
    const result = document.getElementById("prediction-result");
    const probSpan = document.getElementById("probability-value");
    const label = document.getElementById("prediction-label");
    const explainer = document.getElementById("prediction-explainer");

    const probPercent = (prob * 100).toFixed(1) + " %";
    probSpan.textContent = probPercent;

    const isFake = prob >= 0.5;

    label.className = "badge " + (isFake ? "badge-fake" : "badge-real");
    label.textContent = isFake ? "Suspicious posting" : "Likely real posting";

    explainer.textContent = isFake
        ? "Модель оценивает эту вакансию как потенциально фейковую. Будьте осторожны: проверьте контакты, домен, требования к оплате и наличие подозрительных ссылок."
        : "Модель оценивает эту вакансию как скорее настоящую. Тем не менее, всегда проверяйте детали самостоятельно (домен, контакты, реалистичность условий).";

    result.classList.remove("hidden");
}

// ------------------------------
// EDA
// ------------------------------
async function initEDA() {
    const status = document.getElementById("eda-status");
    try {
        status.textContent = "Loading EDA data…";

        const [edaData, csvInfo] = await Promise.all([
            loadEdaJson(),
            loadCsvFromZip(),
        ]);
console.log("EDA JSON keys:", Object.keys(edaData));

        if (csvInfo) {
            const infoDiv = document.getElementById("csv-info");
            infoDiv.textContent =
                `Rows in combined_dataset_clean.csv: ${csvInfo.rows}` +
                (csvInfo.fakeCount != null
                    ? ` | Fraudulent = ${csvInfo.fakeCount}, Real = ${csvInfo.rows - csvInfo.fakeCount}`
                    : "");
        }

        renderEdaChartsAndWordclouds(edaData);

        status.textContent = "EDA data loaded.";
        status.classList.remove("status-loading");
        status.classList.add("status-ready");
    } catch (err) {
        console.error("EDA error:", err);
        status.textContent = "Error loading EDA data. See console.";
        status.classList.remove("status-loading");
        status.classList.add("status-error");
    }
}

/**
 * Ожидаемый формат файла data/eda_data.json (пример):
 *
 * {
 *   "class_distribution": { "real": 17000, "fake": 1800 },
 *   "text_length_distribution": {
 *      "bins": [0, 50, 100, ...],
 *      "real": [count, count, ...],
 *      "fake": [count, count, ...]
 *   },
 *   "text_length_boxplot": {
 *      "real": { "min": 20, "q1": 120, "median": 230, "q3": 400, "max": 1200 },
 *      "fake": { "min": 10, "q1": 80,  "median": 160, "q3": 300, "max": 900  }
 *   },
 *   "missing_values": {
 *      "fields": ["title", "company_profile", ...],
 *      "real":   [0.01, 0.2, ...],  // доли или проценты
 *      "fake":   [0.05, 0.4, ...]
 *   },
 *   "wordcloud": {
 *      "real": [ { "text": "engineer", "size": 40 }, ... ],
 *      "fake": [ { "text": "click",    "size": 50 }, ... ]
 *   }
 * }
 */
async function loadEdaJson() {
    const res = await fetch("data/eda_data.json");
    if (!res.ok) {
        throw new Error("Cannot load eda_data.json");
    }
    return res.json();
}

async function loadCsvFromZip() {
    const res = await fetch("data/combined_dataset_clean.zip");
    if (!res.ok) {
        throw new Error("Cannot load combined_dataset_clean.zip");
    }
    const buffer = await res.arrayBuffer();
    const zip = await JSZip.loadAsync(buffer);

    // Find first CSV file in the archive
    let csvFile = null;
    zip.forEach((relativePath, file) => {
        if (!csvFile && /\.csv$/i.test(relativePath)) {
            csvFile = file;
        }
    });

    if (!csvFile) {
        return { rows: 0 };
    }

    const csvText = await csvFile.async("string");
    const parsed = Papa.parse(csvText, {
        header: true,
        skipEmptyLines: true,
    });

    const rows = parsed.data.length;

    // Try to detect fraudulent column if present
    let fakeCount = null;
    if (parsed.meta && parsed.meta.fields && parsed.meta.fields.includes("fraudulent")) {
        fakeCount = parsed.data.filter((row) => {
            const v = row["fraudulent"];
            return v === "1" || v === 1 || v === true || v === "true";
        }).length;
    }

    return { rows, fakeCount };
}

function renderEdaChartsAndWordclouds(eda) {
    if (!eda || typeof eda !== "object") {
        console.warn("EDA: empty or invalid data", eda);
        return;
    }

    console.log("EDA JSON keys:", Object.keys(eda));

    // ---------- 1. CLASS DISTRIBUTION (class_counts) ----------
    if (eda.class_counts) {
        const src = eda.class_counts;

        // Пытаемся аккуратно вытащить real / fake
        let real = 0;
        let fake = 0;

        if ("real" in src || "fake" in src) {
            real = src.real ?? 0;
            fake = src.fake ?? 0;
        } else if ("0" in src || "1" in src) {
            // частый вариант: {"0": real, "1": fake}
            real = src["0"] ?? 0;
            fake = src["1"] ?? 0;
        } else {
            // на крайний случай: берём первые два значения
            const vals = Object.values(src);
            if (vals.length >= 2) {
                real = vals[0];
                fake = vals[1];
            }
        }

        renderClassDistributionChart({ real, fake });
    } else {
        console.warn("EDA: no class_counts in JSON");
    }

    // ---------- 2. TEXT LENGTH DISTRIBUTION (lengths) ----------
    if (eda.lengths) {
        const src = eda.lengths;

        // ожидаемый формат:
        // { bins: [...], real: [...], fake: [...] }
        const bins =
            src.bins ||
            src.edges ||
            src.x ||
            [];
        const real =
            src.real ||
            src.real_counts ||
            src.real_hist ||
            [];
        const fake =
            src.fake ||
            src.fake_counts ||
            src.fake_hist ||
            [];

        renderLengthDistributionChart({
            bins,
            real,
            fake,
        });
    } else {
        console.warn("EDA: no lengths in JSON");
    }

    // ---------- 3. MISSING VALUES (missing) ----------
    if (eda.missing) {
        const src = eda.missing;

        // ожидаемый формат:
        // { fields: [...], real: [...], fake: [...] }
        const fields =
            src.fields ||
            src.columns ||
            [];
        const real =
            src.real ||
            src.real_share ||
            [];
        const fake =
            src.fake ||
            src.fake_share ||
            [];

        renderMissingValuesChart({
            fields,
            real,
            fake,
        });
    } else {
        console.warn("EDA: no missing in JSON");
    }

    // ---------- 4. BOXLOT + WORDCLOUD ----------
    // В твоём JSON нет явных ключей для boxplot и wordcloud,
    // поэтому аккуратно пропускаем эти визуализации.
    // (Если потом добавите в eda_data.json отдельные поля — легко подключим.)
}


// ------------------------------
// Charts
// ------------------------------
function renderClassDistributionChart(dist) {
    const canvas = document.getElementById("class-distribution-chart");
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    if (classChart) {
        classChart.destroy();
    }

    const real = dist.real ?? 0;
    const fake = dist.fake ?? 0;

    classChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: ["Real", "Fake"],
            datasets: [
                {
                    label: "Count",
                    data: [real, fake],
                },
            ],
        },
        options: {
            responsive: true,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => `Count: ${ctx.raw}`,
                    },
                },
            },
            scales: {
                x: {
                    ticks: { color: "#9ca3af" },
                    grid: { display: false },
                },
                y: {
                    ticks: { color: "#9ca3af" },
                    grid: { color: "rgba(55,65,81,0.6)" },
                },
            },
        },
    });
}

function renderLengthDistributionChart(dist) {
    const ctx = document.getElementById("length-distribution-chart");
    if (!ctx) return;

    if (lengthChart) {
        lengthChart.destroy();
    }

    const binsRaw = dist.bins || dist.edges || dist.x || [];
    const realRaw = dist.real || dist.real_counts || dist.real_hist || [];
    const fakeRaw = dist.fake || dist.fake_counts || dist.fake_hist || [];

    // На всякий случай лог — если что, посмотрим в консоли
    console.log("Length dist raw:", {
        bins: binsRaw.length,
        real: realRaw.length,
        fake: fakeRaw.length,
        sampleReal: realRaw.slice(0, 5),
        sampleFake: fakeRaw.slice(0, 5),
    });

    // Приводим всё к числам (мало ли, вдруг строки)
    const bins = binsRaw.map((v) => Number(v));
    const real = realRaw.map((v) => Number(v));
    const fake = fakeRaw.map((v) => Number(v));

    const minLen = Math.min(bins.length, real.length, fake.length);

    if (!minLen) {
        console.warn("Length distribution: no usable data", {
            binsLen: bins.length,
            realLen: real.length,
            fakeLen: fake.length,
        });
        return;
    }

    const labels = bins.slice(0, minLen);
    const realData = real.slice(0, minLen);
    const fakeData = fake.slice(0, minLen);

    lengthChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels,
            datasets: [
                {
                    label: "Real",
                    data: realData,
                    backgroundColor: "rgba(59,130,246,0.7)",
                    borderColor: "rgba(191,219,254,1)",
                    borderWidth: 1,
                },
                {
                    label: "Fake",
                    data: fakeData,
                    backgroundColor: "rgba(248,113,113,0.7)",
                    borderColor: "rgba(254,202,202,1)",
                    borderWidth: 1,
                },
            ],
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    labels: { color: "#e5e7eb" },
                },
                tooltip: {
                    callbacks: {
                        title: (items) =>
                            "Text length ≤ " + items[0].label,
                    },
                },
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: "Text length bin",
                        color: "#9ca3af",
                    },
                    ticks: {
                        color: "#9ca3af",
                        maxRotation: 0,
                        autoSkip: true,
                        maxTicksLimit: 10,
                    },
                    grid: { display: false },
                },
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: "Count",
                        color: "#9ca3af",
                    },
                    ticks: { color: "#9ca3af" },
                    grid: { color: "rgba(55,65,81,0.6)" },
                },
            },
        },
    });
}


function renderMissingValuesChart(missing) {
    const canvas = document.getElementById("missing-values-chart");
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    if (missingChart) {
        missingChart.destroy();
    }

    const fields = missing.fields || [];
    const real = missing.real || [];
    const fake = missing.fake || [];

    missingChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: fields,
            datasets: [
                {
                    label: "Real",
                    data: real,
                },
                {
                    label: "Fake",
                    data: fake,
                },
            ],
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    labels: { color: "#e5e7eb" },
                },
                tooltip: {
                    callbacks: {
                        label: (ctx) => {
                            const v = ctx.raw;
                            if (v <= 1) return `Share: ${(v * 100).toFixed(1)} %`;
                            return `Value: ${v}`;
                        },
                    },
                },
            },
            scales: {
                x: {
                    ticks: { color: "#9ca3af" },
                    grid: { display: false },
                },
                y: {
                    ticks: { color: "#9ca3af" },
                    grid: { color: "rgba(55,65,81,0.6)" },
                },
            },
        },
    });
}

// ------------------------------
// Simple custom boxplot (canvas only)
// ------------------------------
function renderBoxplot(boxData) {
    const canvas = document.getElementById("boxplot-canvas");
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;

    // Clear
    ctx.clearRect(0, 0, width, height);

    const real = boxData.real || {};
    const fake = boxData.fake || {};

    const allValues = [
        real.min,
        real.max,
        fake.min,
        fake.max,
    ].filter((v) => typeof v === "number");

    if (!allValues.length) return;

    const minY = Math.min(...allValues);
    const maxY = Math.max(...allValues);
    const padding = 30;

    const scaleY = (value) => {
        if (maxY === minY) return height / 2;
        const t = (value - minY) / (maxY - minY);
        return height - padding - t * (height - 2 * padding);
    };

    // Axis
    ctx.strokeStyle = "rgba(75,85,99,0.7)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding, padding / 2);
    ctx.lineTo(padding, height - padding);
    ctx.stroke();

    ctx.fillStyle = "#9ca3af";
    ctx.font = "10px system-ui";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";

    const ticks = 4;
    for (let i = 0; i <= ticks; i++) {
        const v = minY + ((maxY - minY) * i) / ticks;
        const y = scaleY(v);
        ctx.fillText(Math.round(v), padding - 4, y);
        ctx.strokeStyle = "rgba(55,65,81,0.4)";
        ctx.beginPath();
        ctx.moveTo(padding, y);
        ctx.lineTo(width - padding, y);
        ctx.stroke();
    }

    // Boxes positions
    const realX = width / 3;
    const fakeX = (2 * width) / 3;

    drawSingleBoxplot(ctx, real, realX, scaleY, "Real");
    drawSingleBoxplot(ctx, fake, fakeX, scaleY, "Fake");
}

function drawSingleBoxplot(ctx, stats, centerX, scaleY, label) {
    const { min, q1, median, q3, max } = stats;

    if (
        [min, q1, median, q3, max].some(
            (v) => typeof v !== "number"
        )
    ) {
        return;
    }

    const boxWidth = 40;

    const yMin = scaleY(min);
    const yQ1 = scaleY(q1);
    const yMed = scaleY(median);
    const yQ3 = scaleY(q3);
    const yMax = scaleY(max);

    // Whiskers
    ctx.strokeStyle = "#60a5fa";
    ctx.lineWidth = 2;

    ctx.beginPath();
    ctx.moveTo(centerX, yMin);
    ctx.lineTo(centerX, yQ1);
    ctx.moveTo(centerX, yQ3);
    ctx.lineTo(centerX, yMax);
    ctx.stroke();

    // Horizontal whisker lines
    ctx.beginPath();
    ctx.moveTo(centerX - boxWidth / 4, yMin);
    ctx.lineTo(centerX + boxWidth / 4, yMin);
    ctx.moveTo(centerX - boxWidth / 4, yMax);
    ctx.lineTo(centerX + boxWidth / 4, yMax);
    ctx.stroke();

    // Box (Q1–Q3)
    ctx.fillStyle = "rgba(56,189,248,0.15)";
    ctx.strokeStyle = "rgba(56,189,248,0.9)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.rect(centerX - boxWidth / 2, yQ3, boxWidth, yQ1 - yQ3);
    ctx.fill();
    ctx.stroke();

    // Median
    ctx.strokeStyle = "#f97373";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(centerX - boxWidth / 2, yMed);
    ctx.lineTo(centerX + boxWidth / 2, yMed);
    ctx.stroke();

    // Label
    ctx.fillStyle = "#e5e7eb";
    ctx.font = "11px system-ui";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText(label, centerX, scaleY(min) + 12);
}

// ------------------------------
// WordCloud via d3-cloud
// ------------------------------
function drawWordCloud(containerId, words) {
    const container = document.getElementById(containerId);
    if (!container || !Array.isArray(words) || words.length === 0) return;

    const width = container.clientWidth || 300;
    const height = container.clientHeight || 220;

    container.innerHTML = "";

    const maxSize = d3.max(words, (d) => d.size || d.freq || d.count || 1) || 1;
    const minSize = d3.min(words, (d) => d.size || d.freq || d.count || 1) || 1;

    const scale = d3
        .scaleLinear()
        .domain([minSize, maxSize])
        .range([12, 42]);

    const layoutWords = words.map((d) => ({
        text: d.text || d.word,
        size: scale(d.size || d.freq || d.count || 1),
    }));

    const layout = d3.layout
        .cloud()
        .size([width, height])
        .words(layoutWords)
        .padding(3)
        .rotate(() => (Math.random() > 0.8 ? 90 : 0))
        .font("system-ui")
        .fontSize((d) => d.size)
        .on("end", (drawWords) => {
            const svg = d3
                .select(container)
                .append("svg")
                .attr("width", width)
                .attr("height", height);

            const group = svg
                .append("g")
                .attr("transform", `translate(${width / 2}, ${height / 2})`);

            group
                .selectAll("text")
                .data(drawWords)
                .enter()
                .append("text")
                .style("font-size", (d) => d.size + "px")
                .style("font-family", "system-ui, sans-serif")
                .style("fill", () => "#f9fafb")
                .attr("text-anchor", "middle")
                .attr("transform", (d) => `translate(${d.x},${d.y})rotate(${d.rotate})`)
                .text((d) => d.text);
        });

    layout.start();
}
