const pptxgen = require("pptxgenjs");

const C = {
  bg: "FFFFFF",
  navy: "0E3B4D",
  teal: "1B7F8C",
  tealLight: "E4F1F3",
  ink: "1A2226",
  sub: "5B6B70",
  low: "2E7D32",
  lowBg: "E8F4EA",
  mod: "B8860B",
  modBg: "FFF6E0",
  high: "B3261E",
  highBg: "FBEAE9",
  card: "F2F7F8",
};

const FONT_H = "Georgia";
const FONT_B = "Calibri";

function kicker(slide, text) {
  slide.addText(text.toUpperCase(), {
    x: 0.6, y: 0.42, w: 12.1, h: 0.3,
    fontFace: FONT_B, fontSize: 12, bold: true, color: C.teal,
    charSpacing: 2, isTextBox: true, margin: 0,
  });
}
function title(slide, text, y = 0.72) {
  slide.addText(text, {
    x: 0.6, y, w: 12.1, h: 0.7,
    fontFace: FONT_H, fontSize: 28, bold: true, color: C.navy,
    isTextBox: true, margin: 0,
  });
}
function pageNum(slide, n) {
  slide.addText(String(n), {
    x: 12.7, y: 7.1, w: 0.5, h: 0.3,
    fontFace: FONT_B, fontSize: 10, color: C.sub, align: "right",
    isTextBox: true, margin: 0,
  });
}
function footer(slide) {
  slide.addText("VitaTwin AI — Predict. Simulate. Prevent.", {
    x: 0.6, y: 7.1, w: 6, h: 0.3,
    fontFace: FONT_B, fontSize: 9, italic: true, color: C.sub,
    isTextBox: true, margin: 0,
  });
}
function bullets(slide, items, opts = {}) {
  slide.addText(
    items.map((t, i) => ({ text: t, options: { bullet: { code: "2022" }, breakLine: i !== items.length - 1 } })),
    {
      x: opts.x ?? 0.6, y: opts.y ?? 1.6, w: opts.w ?? 11.9, h: opts.h ?? 4.8,
      fontFace: FONT_B, fontSize: opts.size ?? 15, color: C.ink,
      isTextBox: true, margin: 0, valign: "top", paraSpaceAfter: opts.spacing ?? 10,
      lineSpacingMultiple: 1.15,
    }
  );
}
function card(slide, x, y, w, h, heading, body, opts = {}) {
  slide.addShape("roundRect", {
    x, y, w, h, rectRadius: 0.08,
    fill: { color: opts.fill || C.card },
    line: { color: opts.line || C.teal, width: 1 },
  });
  slide.addText(heading, {
    x: x + 0.22, y: y + 0.15, w: w - 0.44, h: 0.35,
    fontFace: FONT_H, fontSize: opts.hSize || 14, bold: true, color: opts.hColor || C.navy,
    isTextBox: true, margin: 0,
  });
  slide.addText(body, {
    x: x + 0.22, y: y + 0.55, w: w - 0.44, h: h - 0.7,
    fontFace: FONT_B, fontSize: opts.bSize || 11, color: C.ink,
    isTextBox: true, margin: 0, valign: "top", lineSpacingMultiple: 1.15,
  });
}

async function build() {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE";
  pres.theme = { headFontFace: FONT_H, bodyFontFace: FONT_B };

  // ---------- 1. Title ----------
  {
    const s = pres.addSlide();
    s.background = { color: C.navy };
    s.addShape("rect", { x: 0, y: 0, w: 13.33, h: 7.5, fill: { color: C.navy } });
    s.addText("VitaTwin AI", {
      x: 0.9, y: 2.5, w: 11.5, h: 1.3,
      fontFace: FONT_H, fontSize: 54, bold: true, color: "FFFFFF",
      isTextBox: true, margin: 0,
    });
    s.addText("Predict.  Simulate.  Prevent.", {
      x: 0.9, y: 3.75, w: 11.5, h: 0.6,
      fontFace: FONT_B, fontSize: 22, color: "9FD6DC", italic: true,
      isTextBox: true, margin: 0,
    });
    s.addShape("line", { x: 0.9, y: 4.55, w: 3.2, h: 0, line: { color: "3E7482", width: 1.5 } });
    s.addText("An AI-powered healthcare Digital Twin proof-of-concept for early\nglycemic-risk awareness in Type 2 Diabetes", {
      x: 0.9, y: 4.75, w: 9.5, h: 0.9,
      fontFace: FONT_B, fontSize: 14, color: "D7E7E9",
      isTextBox: true, margin: 0, lineSpacingMultiple: 1.3,
    });
    s.addText("Happiest Health Digital Twin Challenge 2026", {
      x: 0.9, y: 6.55, w: 8, h: 0.35,
      fontFace: FONT_B, fontSize: 12, bold: true, color: "FFFFFF",
      isTextBox: true, margin: 0,
    });
    s.addText("Team: A Girl :)   |   SRM Madurai College for Engineering and Technology", {
      x: 0.9, y: 6.9, w: 9, h: 0.3,
      fontFace: FONT_B, fontSize: 11, color: "9FD6DC",
      isTextBox: true, margin: 0,
    });
  }

  // ---------- 2. Problem ----------
  {
    const s = pres.addSlide();
    kicker(s, "The Problem");
    title(s, "Healthcare today reacts after the event has already happened");
    card(s, 0.6, 1.85, 3.85, 4.3, "DATA", ["A patient's medical record and", "wearable data exist —", "but sit unused between visits."].join("\n"), { fill: C.card });
    card(s, 4.65, 1.85, 3.85, 4.3, "EVENT", ["A glucose excursion, or other", "adverse event, occurs", "unpredicted."].join("\n"), { fill: C.highBg, line: C.high, hColor: C.high });
    card(s, 8.7, 1.85, 3.85, 4.3, "REACTION", ["Clinical response happens", "only after the fact —", "the costliest point to intervene."].join("\n"), { fill: C.card });
    s.addText("Type 2 Diabetes is one of India's most prevalent chronic, lifestyle-linked conditions — and glycemic instability is a leading driver of avoidable complications.", {
      x: 0.6, y: 6.35, w: 11.9, h: 0.6,
      fontFace: FONT_B, fontSize: 13, italic: true, color: C.sub,
      isTextBox: true, margin: 0,
    });
    footer(s); pageNum(s, 2);
  }

  // ---------- 3. Paradigm shift ----------
  {
    const s = pres.addSlide();
    kicker(s, "Our Approach");
    title(s, "From reacting to a single event, to living alongside a twin");
    const steps = ["DATA", "DIGITAL TWIN", "PREDICTION", "SIMULATION", "EARLY AWARENESS"];
    const w = 2.2, gap = 0.25, startX = 0.6, y = 2.6, h = 1.3;
    steps.forEach((t, i) => {
      const x = startX + i * (w + gap);
      s.addShape("roundRect", {
        x, y, w, h, rectRadius: 0.08,
        fill: { color: i === 0 ? C.card : C.tealLight },
        line: { color: C.teal, width: 1.25 },
      });
      s.addText(t, {
        x, y: y + 0.45, w, h: 0.4,
        fontFace: FONT_H, fontSize: 12.5, bold: true, color: C.navy,
        align: "center", isTextBox: true, margin: 0,
      });
      if (i < steps.length - 1) {
        s.addShape("line", {
          x: x + w, y: y + h / 2, w: gap, h: 0,
          line: { color: C.sub, width: 2, endArrowType: "triangle" },
        });
      }
    });
    s.addText("Traditional healthcare: DATA → EVENT → REACTION", {
      x: 0.6, y: 4.35, w: 11.9, h: 0.35,
      fontFace: FONT_B, fontSize: 14, color: C.high, isTextBox: true, margin: 0,
    });
    s.addText("VitaTwin AI creates a living computational representation of a patient — combining what we already know with what is happening right now — so the AI can estimate what could happen next, and clinicians can explore hypothetical futures before deciding.", {
      x: 0.6, y: 5.0, w: 11.5, h: 1.6,
      fontFace: FONT_B, fontSize: 14, color: C.ink, isTextBox: true, margin: 0, lineSpacingMultiple: 1.3,
    });
    footer(s); pageNum(s, 3);
  }

  // ---------- 4. Architecture (embed the diagram image) ----------
  {
    const s = pres.addSlide();
    kicker(s, "System Design");
    title(s, "Architecture");
    s.addImage({ path: "slide-1.jpg", x: 0.5, y: 1.55, w: 12.33, h: 5.6 });
    footer(s); pageNum(s, 4);
  }

  // ---------- 5. Data strategy ----------
  {
    const s = pres.addSlide();
    kicker(s, "Data Strategy");
    title(s, "Synthetic, disclosed, and schema-compatible with real data");
    card(s, 0.6, 1.85, 5.9, 2.05, "Static / Historical", [
      "Synthetic T2D-relevant patient population",
      "Age 35-72, BMI 23-39, HbA1c 5.8-9.8%",
      "Diagnosis, duration, medication profile",
    ].join("\n"));
    card(s, 6.75, 1.85, 5.9, 2.05, "Dynamic / Real-Time", [
      "5-minute glucose / HR / HRV / steps / sleep",
      "Rule-based physiological simulator",
      "Meal + activity + circadian response modeling",
    ].join("\n"));
    card(s, 0.6, 4.1, 12.05, 2.0,
      "Why synthetic, and why that's the right call",
      "Real CGM+wearable datasets (OhioT1DM, D1NAMO) are Type 1 diabetes research cohorts — there is no strong open dataset combining T2D-specific EHR with continuous wearable+CGM data. Rather than hide this mismatch, every row carries a data_source tag, and the dynamic-stream schema is built to be drop-in compatible with real datasets via a documented loader stub — so real data replaces the simulator later without touching downstream code.",
      { fill: C.modBg, line: C.mod, hColor: C.mod, bSize: 12.5 }
    );
    footer(s); pageNum(s, 5);
  }

  // ---------- 6. Digital Twin engine ----------
  {
    const s = pres.addSlide();
    kicker(s, "Core Innovation");
    title(s, "An explicit Digital Twin state — not just a model");
    bullets(s, [
      "VitaTwinState: a real, persisted per-patient object — demographics, historical record, dynamic signals, derived features, prediction, and explanation — not a label a model spits out",
      "Update loop: new sensor data → validation → timestamp alignment → twin update → feature engineering → prediction → risk → explanation → dashboard refresh",
      "Personal baseline: risk is judged against each patient's own recent glucose baseline, not only a population average — \"current state is 19.7% above this patient's baseline\"",
      "What-If simulation runs on a forked copy of the state — it never mutates the real, persisted twin",
    ], { y: 1.75, size: 15.5, h: 5.0, spacing: 14 });
    footer(s); pageNum(s, 6);
  }

  // ---------- 7. ML methodology ----------
  {
    const s = pres.addSlide();
    kicker(s, "Machine Learning");
    title(s, "Baselines first, temporal validation, no leakage");
    card(s, 0.6, 1.85, 3.85, 4.3, "Dual Prediction Target", [
      "Regression: glucose value",
      "~2 hours ahead",
      "",
      "Classification: probability of a",
      "hyperglycemic excursion",
      "(>180 mg/dL) within 2 hours",
    ].join("\n"));
    card(s, 4.65, 1.85, 3.85, 4.3, "Models — Trained & Evaluated", [
      "Regression: Ridge vs. Random Forest",
      "  -> Random Forest won (val MAE",
      "     10.7 vs. 15.9)",
      "",
      "Classification: LogReg vs. RF",
      "  -> Random Forest won (val ROC-AUC",
      "     0.92 vs. 0.85)",
      "",
      "Real, run, test-set results -- see",
      "docs/model.md",
    ].join("\n"));
    card(s, 8.7, 1.85, 3.85, 4.3, "Validation Discipline", [
      "Per-patient chronological",
      "70/15/15 split (train/val/test)",
      "",
      "No shuffling -- adjacent 5-min",
      "readings are correlated, so a",
      "random split would leak",
      "",
      "Test MAE: 11.0 mg/dL",
      "Test ROC-AUC: 0.92",
    ].join("\n"), { fill: C.modBg, line: C.mod, hColor: C.mod });
    footer(s); pageNum(s, 7);
  }

  // ---------- 8. Explainability ----------
  {
    const s = pres.addSlide();
    kicker(s, "Trust & Transparency");
    title(s, "The clinician never sees a bare risk score");
    s.addShape("roundRect", {
      x: 0.6, y: 1.85, w: 5.6, h: 4.3, rectRadius: 0.08,
      fill: { color: C.highBg }, line: { color: C.high, width: 1.25 },
    });
    s.addText("Risk: HIGH", {
      x: 0.9, y: 2.1, w: 5, h: 0.5,
      fontFace: FONT_H, fontSize: 22, bold: true, color: C.high, isTextBox: true, margin: 0,
    });
    s.addText([
      { text: "↑ glucose trend (rising 3 of last 4 readings)", options: { breakLine: true } },
      { text: "↓ recent activity vs. patient baseline", options: { breakLine: true } },
      { text: "↓ sleep duration last night", options: { breakLine: true } },
      { text: "↑ 22% deviation from personal glucose baseline", options: { breakLine: false } },
    ], {
      x: 0.9, y: 2.8, w: 5.0, h: 2.8,
      fontFace: FONT_B, fontSize: 14, color: C.ink, isTextBox: true, margin: 0, paraSpaceAfter: 10,
    });
    card(s, 6.55, 1.85, 6.15, 4.3, "How", [
      "SHAP (TreeExplainer) on the selected",
      "tree-based model gives signed,",
      "per-prediction feature contributions.",
      "",
      "We show direction and rank of the",
      "top 4-5 factors — never a raw SHAP",
      "plot dumped on a clinician, and never",
      "a causal claim the evidence doesn't",
      "support (e.g. we do not say low sleep",
      "\"caused\" the glucose spike).",
    ].join("\n"), { bSize: 13 });
    footer(s); pageNum(s, 8);
  }

  // ---------- 9. What-If simulator ----------
  {
    const s = pres.addSlide();
    kicker(s, "Key Differentiator — Real Output, Not a Mockup");
    title(s, "What-If Simulator: baseline vs. simulated scenario");
    card(s, 0.6, 1.85, 5.6, 3.55, "Baseline — Patient VT001, actual state", [
      "",
      "Predicted glucose (+2h):  186 mg/dL",
      "Excursion probability:  81%",
      "Risk category:  HIGH",
    ].join("\n"), { hColor: C.high, bSize: 15 });
    card(s, 6.4, 1.85, 6.3, 3.55, "Simulated — +activity, +sleep, -glucose", [
      "",
      "Predicted glucose (+2h):  179 mg/dL",
      "Excursion probability:  70%",
      "Risk category:  HIGH (reduced, not flipped)",
    ].join("\n"), { fill: C.modBg, line: C.mod, hColor: C.mod, bSize: 15 });
    card(s, 0.6, 5.55, 12.1, 1.3,
      "Honest finding, stated plainly",
      "Even with a combined activity+sleep+glucose intervention, risk stayed HIGH -- this baseline model is (correctly) dominated by current glucose over short-horizon lifestyle levers. Documented as a limitation and a concrete next step (richer meal/medication features), not hidden. Every number above came from actually running src/simulation/whatif.py.",
      { fill: C.card, bSize: 11.5 }
    );
    footer(s); pageNum(s, 9);
  }

  // ---------- 10. Demo / live twin ----------
  {
    const s = pres.addSlide();
    kicker(s, "Demonstration");
    title(s, "Live Twin: a real, working replay engine");
    bullets(s, [
      "Built and working: a \u201cLive Twin\u201d tab replays each patient's real last 24 hours, 5 minutes at a time, with Play / Pause / Reset / Scrub and adjustable speed (1x / 4x / 12x)",
      "Each step re-runs the ACTUAL trained model live: Digital Twin updates \u2192 prediction recomputes \u2192 risk re-assesses \u2192 the same risk ring/SHAP panels used everywhere else update in real time",
      "Verified end-to-end: in one real patient's 24h window, risk genuinely moves LOW \u2192 MODERATE \u2192 HIGH \u2192 MODERATE \u2192 LOW as the day progresses \u2014 not scripted, just what the model actually outputs at each real historical point",
      "Jury demo flow: select patient \u2192 open Live Twin \u2192 press Play \u2192 watch risk rise and fall live \u2192 pause on a HIGH moment \u2192 open \u201cWhy this prediction?\u201d \u2192 open What-If Simulator",
    ], { y: 1.75, size: 14.5, h: 4.8, spacing: 13 });
    footer(s); pageNum(s, 10);
  }

  // ---------- 11. Progress & Roadmap ----------
  {
    const s = pres.addSlide();
    kicker(s, "Status");
    title(s, "Progress & Roadmap");
    const done = [
      "Phase 0 — Feasibility report",
      "Phase 1 — Synthetic data generator (11/11 tests)",
      "Phase 2 — Preprocessing & fusion pipeline",
      "Phase 3 — Digital Twin engine (fork/update/snapshot)",
      "Phase 4-5 — Trained models, real test metrics",
      "Phase 6 — SHAP explainability",
      "Phase 7 — What-If simulator + recommendations",
      "Phase 8 — FastAPI backend (12 endpoints)",
      "Phase 9 — Two dashboards: Streamlit + custom web UI",
      "Phase 10 — Live Twin replay (real, working, tested)",
      "Phase 11 — Full test suite: 66/66 passing",
      "Phase 12 — Documentation (7 docs, all real)",
    ];
    const todo = [
      "Phase 13 — Record the actual demo video",
      "  (script ready in docs/demo_script.md,",
      "   every number verified against the",
      "   live running system)",
      "",
      "Team name + college name",
      "  (placeholders in README/deck/LICENSE)",
    ];
    card(s, 0.6, 1.85, 7.3, 4.3, "Complete — real, tested code", done.map(d => "\u2713 " + d).join("\n"), { fill: C.lowBg, line: C.low, hColor: C.low, bSize: 11.5 });
    card(s, 8.1, 1.85, 4.6, 4.3, "Remaining", todo.map(d => d ? "\u25CB " + d : "").join("\n"), { fill: C.card, bSize: 11.5 });
    footer(s); pageNum(s, 11);
  }

  // ---------- 12. Safety / Team / Close ----------
  {
    const s = pres.addSlide();
    s.background = { color: C.navy };
    s.addText("Safety, honestly stated", {
      x: 0.9, y: 0.9, w: 11.5, h: 0.6,
      fontFace: FONT_H, fontSize: 26, bold: true, color: "FFFFFF", isTextBox: true, margin: 0,
    });
    s.addText("VitaTwin AI is a research and educational prototype. Predictions are experimental model outputs and are not medical diagnoses or treatment recommendations. All data is synthetic or disclosed as a physiological proxy — no real patient data is used.", {
      x: 0.9, y: 1.65, w: 10.8, h: 1.1,
      fontFace: FONT_B, fontSize: 14, color: "D7E7E9", isTextBox: true, margin: 0, lineSpacingMultiple: 1.3,
    });
    s.addShape("line", { x: 0.9, y: 3.0, w: 11.5, h: 0, line: { color: "3E7482", width: 1 } });
    s.addText("Team", {
      x: 0.9, y: 3.3, w: 5, h: 0.4,
      fontFace: FONT_H, fontSize: 16, bold: true, color: "9FD6DC", isTextBox: true, margin: 0,
    });
    s.addText("A Girl :)  —  SRM Madurai College for Engineering and Technology\nTeam Leader: Dharshini M", {
      x: 0.9, y: 3.75, w: 8, h: 1.2,
      fontFace: FONT_B, fontSize: 13, color: "FFFFFF", isTextBox: true, margin: 0, lineSpacingMultiple: 1.3,
    });
    s.addText("VitaTwin AI  —  Predict. Simulate. Prevent.", {
      x: 0.9, y: 6.6, w: 10, h: 0.4,
      fontFace: FONT_H, fontSize: 15, italic: true, color: "9FD6DC", isTextBox: true, margin: 0,
    });
  }

  await pres.writeFile({ fileName: "presentation_deck.pptx" });
  console.log("done");
}

build();
