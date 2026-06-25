import {
  Tensor,
  isWebGPUSupported,
  loadAndCompile,
  loadLiteRt,
  unloadLiteRt,
} from "@litertjs/core";

const elements = {
  backend: document.getElementById("backend"),
  modelUrl: document.getElementById("modelUrl"),
  imageUrl: document.getElementById("imageUrl"),
  imageSize: document.getElementById("imageSize"),
  warmupRuns: document.getElementById("warmupRuns"),
  benchmarkRuns: document.getElementById("benchmarkRuns"),
  runButton: document.getElementById("runButton"),
  score: document.getElementById("score"),
  mean: document.getElementById("mean"),
  median: document.getElementById("median"),
  minMax: document.getElementById("minMax"),
  accelerated: document.getElementById("accelerated"),
  log: document.getElementById("log"),
};

const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

let liteRtLoaded = false;

function log(message) {
  const timestamp = new Date().toLocaleTimeString();
  elements.log.textContent += `[${timestamp}] ${message}\n`;
  elements.log.scrollTop = elements.log.scrollHeight;
}

function setMetric(element, value) {
  element.textContent = value;
}

function formatMs(value) {
  return `${value.toFixed(3)} ms`;
}

function percentile(sortedValues, ratio) {
  if (sortedValues.length === 0) return Number.NaN;
  const index = Math.min(sortedValues.length - 1, Math.max(0, Math.floor(sortedValues.length * ratio)));
  return sortedValues[index];
}

function summarize(times) {
  const sorted = [...times].sort((a, b) => a - b);
  const sum = times.reduce((acc, value) => acc + value, 0);
  return {
    mean: sum / times.length,
    median: percentile(sorted, 0.5),
    min: sorted[0],
    max: sorted[sorted.length - 1],
    p90: percentile(sorted, 0.9),
    p95: percentile(sorted, 0.95),
  };
}

function parsePositiveInt(value, name) {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(`${name} must be a positive integer.`);
  }
  return parsed;
}

async function loadImage(url) {
  const image = new Image();
  image.decoding = "async";
  image.src = url;
  await image.decode();
  return image;
}

function resizeShortEdge(width, height, size) {
  if (width <= height) {
    return [size, Math.trunc((size * height) / width)];
  }
  return [Math.trunc((size * width) / height), size];
}

async function preprocessImage(imageUrl, imageSize) {
  const image = await loadImage(imageUrl);
  const [resizedWidth, resizedHeight] = resizeShortEdge(image.naturalWidth, image.naturalHeight, imageSize);

  const resizeCanvas = document.createElement("canvas");
  resizeCanvas.width = resizedWidth;
  resizeCanvas.height = resizedHeight;
  const resizeContext = resizeCanvas.getContext("2d", { willReadFrequently: true });
  resizeContext.imageSmoothingEnabled = true;
  resizeContext.imageSmoothingQuality = "medium";
  resizeContext.drawImage(image, 0, 0, resizedWidth, resizedHeight);

  const cropCanvas = document.createElement("canvas");
  cropCanvas.width = imageSize;
  cropCanvas.height = imageSize;
  const cropContext = cropCanvas.getContext("2d", { willReadFrequently: true });
  cropContext.drawImage(
    resizeCanvas,
    Math.round((resizedWidth - imageSize) / 2),
    Math.round((resizedHeight - imageSize) / 2),
    imageSize,
    imageSize,
    0,
    0,
    imageSize,
    imageSize,
  );

  const pixels = cropContext.getImageData(0, 0, imageSize, imageSize).data;
  const input = new Float32Array(1 * 3 * imageSize * imageSize);
  const planeSize = imageSize * imageSize;

  for (let index = 0; index < planeSize; index += 1) {
    const pixelIndex = index * 4;
    input[index] = (pixels[pixelIndex] / 255 - MEAN[0]) / STD[0];
    input[planeSize + index] = (pixels[pixelIndex + 1] / 255 - MEAN[1]) / STD[1];
    input[planeSize * 2 + index] = (pixels[pixelIndex + 2] / 255 - MEAN[2]) / STD[2];
  }

  return input;
}

async function ensureLiteRtLoaded() {
  if (liteRtLoaded) return;
  await loadLiteRt("/litert-wasm/", { threads: false });
  liteRtLoaded = true;
}

async function runOnce(model, inputTensor) {
  const startedAt = performance.now();
  const outputs = await model.run(inputTensor);
  const firstOutput = Array.isArray(outputs) ? outputs[0] : Object.values(outputs)[0];
  const outputData = await firstOutput.data();
  const elapsed = performance.now() - startedAt;
  const score = outputData[0];

  for (const output of Array.isArray(outputs) ? outputs : Object.values(outputs)) {
    output.delete();
  }

  return { elapsed, score };
}

async function benchmark() {
  elements.runButton.disabled = true;
  elements.log.textContent = "";
  setMetric(elements.score, "-");
  setMetric(elements.mean, "-");
  setMetric(elements.median, "-");
  setMetric(elements.minMax, "-");
  setMetric(elements.accelerated, "-");

  let model;
  let inputTensor;

  try {
    const backend = elements.backend.value;
    const modelUrl = elements.modelUrl.value.trim();
    const imageUrl = elements.imageUrl.value.trim();
    const imageSize = parsePositiveInt(elements.imageSize.value, "image_size");
    const warmupRuns = Number.parseInt(elements.warmupRuns.value, 10);
    const benchmarkRuns = parsePositiveInt(elements.benchmarkRuns.value, "benchmark_runs");

    if (backend === "webgpu" && !isWebGPUSupported()) {
      throw new Error("WebGPU is not available in this browser. Use Chrome/Edge over HTTPS or select wasm.");
    }
    if (!Number.isFinite(warmupRuns) || warmupRuns < 0) {
      throw new Error("warmup_runs must be zero or a positive integer.");
    }

    log(`Backend: ${backend}`);
    log(`Model: ${modelUrl}`);
    log(`Image: ${imageUrl}`);
    log("Loading LiteRT.js...");
    await ensureLiteRtLoaded();

    log("Preprocessing image...");
    const inputData = await preprocessImage(imageUrl, imageSize);
    inputTensor = new Tensor(inputData, [1, 3, imageSize, imageSize]);

    log("Compiling TFLite model...");
    const compileOptions = {
      accelerator: backend,
      gpuOptions: { precision: "fp32" },
      cpuOptions: { numThreads: Math.max(1, navigator.hardwareConcurrency || 1) },
    };
    const compileStartedAt = performance.now();
    model = await loadAndCompile(modelUrl, compileOptions);
    const compileElapsed = performance.now() - compileStartedAt;
    log(`Compile time: ${formatMs(compileElapsed)}`);
    log(`isFullyAccelerated: ${model.isFullyAccelerated}`);
    setMetric(elements.accelerated, String(model.isFullyAccelerated));

    const inputDetails = model.getInputDetails();
    const outputDetails = model.getOutputDetails();
    log(`Input: ${inputDetails.map((detail) => `${detail.name} [${Array.from(detail.shape).join(", ")}] ${detail.dtype}`).join("; ")}`);
    log(`Output: ${outputDetails.map((detail) => `${detail.name} [${Array.from(detail.shape).join(", ")}] ${detail.dtype}`).join("; ")}`);

    for (let i = 0; i < warmupRuns; i += 1) {
      await runOnce(model, inputTensor);
    }
    log(`Warmup completed: ${warmupRuns} runs`);

    const times = [];
    let score = Number.NaN;
    for (let i = 0; i < benchmarkRuns; i += 1) {
      const result = await runOnce(model, inputTensor);
      times.push(result.elapsed);
      score = result.score;
    }

    const stats = summarize(times);
    setMetric(elements.score, score.toFixed(6));
    setMetric(elements.mean, formatMs(stats.mean));
    setMetric(elements.median, formatMs(stats.median));
    setMetric(elements.minMax, `${formatMs(stats.min)} / ${formatMs(stats.max)}`);
    log(`Runs: ${benchmarkRuns}`);
    log(`Mean: ${formatMs(stats.mean)}`);
    log(`Median: ${formatMs(stats.median)}`);
    log(`P90: ${formatMs(stats.p90)}`);
    log(`P95: ${formatMs(stats.p95)}`);
    log(`Min: ${formatMs(stats.min)}`);
    log(`Max: ${formatMs(stats.max)}`);
    log(`Score: ${score.toFixed(8)}`);
  } catch (error) {
    log(`Error: ${error instanceof Error ? error.message : String(error)}`);
    console.error(error);
  } finally {
    inputTensor?.delete();
    model?.delete();
    elements.runButton.disabled = false;
  }
}

window.addEventListener("beforeunload", () => {
  if (liteRtLoaded) {
    unloadLiteRt();
  }
});

elements.runButton.addEventListener("click", benchmark);

log(`WebGPU supported: ${isWebGPUSupported()}`);
log("Select a backend and press Run benchmark.");
