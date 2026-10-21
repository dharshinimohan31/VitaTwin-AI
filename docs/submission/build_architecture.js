const pptxgen = require("pptxgenjs");

const COLOR = {
  bg: "FFFFFF",
  navy: "0E3B4D",
  teal: "1B7F8C",
  tealLight: "E4F1F3",
  ink: "1A2226",
  sub: "5B6B70",
  low: "2E7D32",
  mod: "B8860B",
  high: "B3261E",
  line: "9FB4B9",
};

function box(slide, x, y, w, h, title, lines, opts = {}) {
  const fill = opts.fill || COLOR.tealLight;
  const titleColor = opts.titleColor || COLOR.navy;
  slide.addShape("roundRect", {
    x, y, w, h,
    rectRadius: 0.08,
    fill: { color: fill },
    line: { color: opts.lineColor || COLOR.teal, width: 1.25 },
    shadow: { type: "outer", color: "1A2226", opacity: 0.18, blur: 4, offset: 2, angle: 90 },
  });
  slide.addText(title, {
    x: x + 0.1, y: y + 0.06, w: w - 0.2, h: 0.28,
    fontFace: "Georgia", fontSize: opts.titleSize || 12, bold: true,
    color: titleColor, align: "left", isTextBox: true, margin: 0,
  });
  if (lines && lines.length) {
    slide.addText(
      lines.map((t, i) => ({ text: t, options: { bullet: { code: "2022" }, breakLine: i !== lines.length - 1 } })),
      {
        x: x + 0.1, y: y + 0.34, w: w - 0.2, h: h - 0.4,
        fontFace: "Calibri", fontSize: opts.bodySize || 9, color: COLOR.ink,
        align: "left", isTextBox: true, margin: 0, valign: "top",
        paraSpaceAfter: 2,
      }
    );
  }
}

function arrow(slide, x1, y1, x2, y2) {
  slide.addShape("line", {
    x: Math.min(x1, x2), y: Math.min(y1, y2),
    w: Math.abs(x2 - x1), h: Math.abs(y2 - y1),
    line: { color: COLOR.line, width: 1.75, endArrowType: "triangle" },
  });
}

async function build() {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE"; // 13.3 x 7.5
  const slide = pres.addSlide();
  slide.background = { color: COLOR.bg };

  slide.addText("VitaTwin AI — System Architecture", {
    x: 0.5, y: 0.22, w: 12.3, h: 0.5,
    fontFace: "Georgia", fontSize: 24, bold: true, color: COLOR.navy,
    isTextBox: true, margin: 0,
  });
  slide.addText("Predict. Simulate. Prevent.  |  Happiest Health Digital Twin Challenge 2026", {
    x: 0.5, y: 0.68, w: 12.3, h: 0.3,
    fontFace: "Calibri", fontSize: 12, italic: true, color: COLOR.sub,
    isTextBox: true, margin: 0,
  });

  // Row 1: Data sources
  box(slide, 0.5, 1.25, 3.6, 1.15, "Static / Historical Data", [
    "Synthetic T2D-relevant EHR",
    "Demographics, HbA1c, diagnosis,",
    "duration, medications",
  ]);
  box(slide, 4.35, 1.25, 3.6, 1.15, "Dynamic / Real-Time Data", [
    "Wearable + CGM stream (5-min)",
    "Glucose, HR, HRV, steps, sleep",
    "Rule-based physiological simulator",
  ]);
  box(slide, 9.2, 1.25, 3.6, 1.15, "Data Provenance Tag", [
    "Every row: data_source column",
    "synthetic vs. real_proxy (future)",
    "See docs/dataset.md",
  ], { fill: "FFF6E0", lineColor: COLOR.mod, titleColor: COLOR.mod });

  arrow(slide, 2.3, 2.4, 2.3, 2.75);
  arrow(slide, 6.15, 2.4, 6.15, 2.75);

  // Row 2: Pipeline
  box(slide, 2.0, 2.78, 8.4, 0.75, "Data Pipeline", [
    "Validation → cleaning → timestamp alignment → patient mapping → missing-value handling → feature engineering",
  ], { fill: "EAF2F1", bodySize: 9.5 });

  arrow(slide, 6.15, 3.53, 6.15, 3.85);

  // Row 3: Digital Twin core
  slide.addShape("roundRect", {
    x: 0.5, y: 3.88, w: 12.3, h: 2.05,
    rectRadius: 0.08,
    fill: { color: "FFFFFF" },
    line: { color: COLOR.navy, width: 1.75 },
  });
  slide.addText("Digital Twin Engine  (per-patient, stateful — VitaTwinState)", {
    x: 0.65, y: 3.95, w: 12.0, h: 0.28,
    fontFace: "Georgia", fontSize: 13, bold: true, color: COLOR.navy,
    isTextBox: true, margin: 0,
  });

  box(slide, 0.75, 4.3, 2.9, 1.5, "Fused Twin State", [
    "demographics · historical",
    "dynamic · derived features",
    "personal baseline deviation",
  ], { fill: "F2F7F8" });
  box(slide, 3.85, 4.3, 2.85, 1.5, "Prediction Engine", [
    "Regression: glucose @ +2h",
    "Classification: excursion risk",
    "RandomForest (beat linear baseline)",
    "Test MAE 11.0 | ROC-AUC 0.92",
  ], { fill: "F2F7F8" });
  box(slide, 6.9, 4.3, 2.85, 1.5, "Risk + Explainability", [
    "Risk: LOW / MODERATE / HIGH",
    "Trend + confidence",
    "SHAP contributing factors",
  ], { fill: "F2F7F8" });
  box(slide, 9.95, 4.3, 2.5, 1.5, "What-If Simulator", [
    "Forked state (no mutation)",
    "Baseline vs. simulated",
    "Labelled hypothetical only",
  ], { fill: "FFF6E0", lineColor: COLOR.mod, titleColor: COLOR.mod });

  arrow(slide, 6.65, 5.93, 6.65, 6.2);

  // Row 4: API + DB + Dashboard
  box(slide, 0.5, 6.22, 3.5, 1.0, "SQLite Database", [
    "patients · ehr_records · sensor_readings",
    "twin_states · predictions · simulation_runs",
  ], { fill: "EAF2F1", bodySize: 8.5 });
  box(slide, 4.3, 6.22, 3.9, 1.0, "FastAPI Backend", [
    "/patients/{id}/twin · /prediction · /risk",
    "/sensor-update · /simulate",
  ], { fill: "EAF2F1", bodySize: 8.5 });
  box(slide, 8.5, 6.22, 4.3, 1.0, "Streamlit Doctor Dashboard", [
    "Overview · Twin · Prediction · Why? · What-If · Live Twin",
  ], { fill: "EAF2F1", bodySize: 8.5 });

  arrow(slide, 4.0, 6.72, 4.3, 6.72);
  arrow(slide, 8.2, 6.72, 8.5, 6.72);

  slide.addText(
    "Research/educational prototype — outputs are experimental model estimates, not medical diagnoses.",
    {
      x: 0.5, y: 7.28, w: 12.3, h: 0.22,
      fontFace: "Calibri", fontSize: 8.5, italic: true, color: COLOR.sub,
      isTextBox: true, margin: 0, align: "center",
    }
  );

  await pres.writeFile({ fileName: "architecture_diagram.pptx" });
  console.log("done");
}

build();
