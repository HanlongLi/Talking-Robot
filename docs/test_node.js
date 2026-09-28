const fs = require("fs"), path = require("path");
const A = require("./artoo.js");
const dir = __dirname;
A.setConstants(JSON.parse(fs.readFileSync(path.join(dir, "constants.json"))));
for (const [name, maxMel] of [["released", 400], ["reproduced", 300]]) {
  const buf = fs.readFileSync(path.join(dir, "weights", name + ".wasm"));
  const W = A.loadContainer(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
  const refs = JSON.parse(fs.readFileSync(path.join(dir, "ref", name + ".json")));
  console.log(`=== ${name}: ${Object.keys(W).length} tensors`);
  for (const r of refs) {
    const ids = A.encode(r.text);
    const okIds = JSON.stringify(ids) === JSON.stringify(r.ids);
    const t0 = Date.now(); const out = A.ttsInfer(W, ids, maxMel); const tTts = Date.now() - t0;
    const durOk = JSON.stringify(out.durations) === JSON.stringify(r.durations);
    const Tref = r.mel[0].length; let maxd = 0, sumd = 0;
    for (let m = 0; m < 40; m++) for (let t = 0; t < Math.min(Tref, out.T); t++) { const d = Math.abs(out.mel[m * out.T + t] - r.mel[m][t]); maxd = Math.max(maxd, d); sumd += d; }
    const t1 = Date.now(); const dec = A.asrDecode(W, out.mel, out.T); const tAsr = Date.now() - t1;
    // ASR on the torch reference mel
    const refMel = new Float32Array(40 * Tref); for (let m = 0; m < 40; m++) for (let t = 0; t < Tref; t++) refMel[m * Tref + t] = r.mel[m][t];
    const decRef = A.asrDecode(W, refMel, Tref);
    // PS path
    const wav = A.psSynth(ids); const pm = A.melFromWav(wav); let psSum = 0; for (let i = 0; i < pm.mel.length; i++) psSum += pm.mel[i];
    const col0 = []; for (let m = 0; m < 40; m++) col0.push(pm.mel[m * pm.T]);
    const c0diff = Math.max(...col0.map((v, i) => Math.abs(v - r.ps_mel_col0[i])));
    const decPs = A.asrDecode(W, pm.mel, pm.T);
    // GL round trip + ASR
    const t2 = Date.now(); const glw = A.griffinLim(out.mel, out.T, 32, 0.99, A.mulberry32(1)); const tGl = Date.now() - t2;
    const gm = A.melFromWav(glw); const decGl = A.asrDecode(W, gm.mel, gm.T);
    console.log(`'${r.text}': ids ${okIds ? "ok" : "MISMATCH"} | dur ${durOk ? "ok" : "MISMATCH " + JSON.stringify(out.durations) + " vs " + JSON.stringify(r.durations)} | T=${out.T}/${Tref} | mel maxdiff=${maxd.toFixed(4)} mean=${(sumd / (40 * Math.min(Tref, out.T))).toFixed(5)}`);
    console.log(`    ASR(js mel): '${A.decode(dec)}' ${JSON.stringify(dec) === JSON.stringify(r.decoded) ? "== torch" : "!= torch '" + A.decode(r.decoded) + "'"} | ASR(torch mel): ${JSON.stringify(decRef) === JSON.stringify(r.decoded) ? "== torch" : "!= torch"} | PS mel shape ${pm.T} vs ${r.ps_mel_shape[1]} sum ${psSum.toFixed(1)} vs ${r.ps_mel_sum.toFixed(1)} col0diff ${c0diff.toExponential(2)} | ASR(PS): '${A.decode(decPs)}' ${JSON.stringify(decPs) === JSON.stringify(r.ps_decoded) ? "== torch" : "!= torch '" + A.decode(r.ps_decoded) + "'"}`);
    console.log(`    GL->mel->ASR: '${A.decode(decGl)}' | timings tts ${tTts} ms, asr ${tAsr} ms, GL ${tGl} ms`);
  }
}
