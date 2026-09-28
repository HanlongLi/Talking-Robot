/* Artoo in the browser: FastSpeech-Nano TTS + Griffin-Lim + mel frontend + Conformer-Tiny CTC ASR,
   procedural synthesizer and channel simulation.  Pure JS, no dependencies.  Works in Node and browsers. */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.Artoo = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";
  const SR = 16000, N_FFT = 512, HOP = 160, N_FREQ = 257, N_MELS = 40, D = 128, HEADS = 4, DFF = 256, VOCAB = 128;

  // ---------------------------------------------------------------- tokenizer
  const LETTERS = "abcdefghijklmnopqrstuvwxyz", DIGITS = "0123456789", PUNCT = ".,!?-:;'\"()/@ #_+";
  const CMDS = ["<STOP>", "<GO>", "<FWD>", "<BWD>", "<LEFT>", "<RIGHT>", "<UP>", "<DOWN>", "<FAST>", "<SLOW>", "<HOME>", "<GOTO>",
    "<GRAB>", "<DROP>", "<LIFT>", "<LOWER>", "<PUSH>", "<PULL>", "<OPEN>", "<CLOSE>", "<ROTATE>", "<SCAN>", "<POINT>", "<REACH>",
    "<ACK>", "<NACK>", "<READY>", "<WAIT>", "<ALERT>", "<EMERG>", "<STATUS>", "<REPORT>", "<YES>", "<NO>", "<OK>", "<FAIL>",
    "<SYNC>", "<LEAD>", "<FOLLOW>", "<FORM>", "<IDLE>", "<BUSY>", "<CHARGE>", "<ERROR>"];
  const CMD_ID = {}; CMDS.forEach((c, i) => { CMD_ID[c] = 84 + i; });
  const C2I = {}, I2C = { 0: "", 1: "", 2: "<sos>", 3: "<eos>", 4: " ", 5: "¿" };
  LETTERS.split("").forEach((ch, i) => { C2I[ch] = 6 + i; I2C[6 + i] = ch; });
  DIGITS.split("").forEach((ch, i) => { C2I[ch] = 32 + i; I2C[32 + i] = ch; });
  PUNCT.split("").forEach((ch, i) => { C2I[ch] = 42 + i; I2C[42 + i] = ch; });
  C2I[" "] = 4;
  for (let i = 0; i < 26; i++) I2C[58 + i] = "<R" + i + ">";
  I2C[58] = "+";  // '+' is the 17th punctuation entry and lands on reserved slot 58
  CMDS.forEach((c) => { I2C[CMD_ID[c]] = c; });
  const CMDS_BY_LEN = CMDS.slice().sort((a, b) => b.length - a.length);
  function encode(text) {
    const out = []; let i = 0;
    while (i < text.length) {
      if (text[i] === "<") {
        const hit = CMDS_BY_LEN.find((c) => text.substr(i, c.length) === c);
        if (hit) { out.push(CMD_ID[hit]); i += hit.length; continue; }
      }
      const ch = text[i].toLowerCase();
      out.push(C2I[ch] !== undefined ? C2I[ch] : 5); i++;
    }
    return out;
  }
  function decode(ids) { return ids.filter((t) => t > 3).map((t) => I2C[t] !== undefined ? I2C[t] : "¿").join(""); }
  function editDistance(a, b) {
    const d = new Int32Array(b.length + 1); for (let j = 0; j <= b.length; j++) d[j] = j;
    for (let i = 1; i <= a.length; i++) {
      let prev = d[0]; d[0] = i;
      for (let j = 1; j <= b.length; j++) { const cur = d[j]; d[j] = Math.min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] === b[j - 1] ? 0 : 1)); prev = cur; }
    }
    return d[b.length];
  }

  // ---------------------------------------------------------------- weights
  function f16to32(h) {
    const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    if (e === 0) return s * m * Math.pow(2, -24);
    if (e === 31) return m ? NaN : s * Infinity;
    return s * (1 + m / 1024) * Math.pow(2, e - 15);
  }
  function loadContainer(buf) {
    const dv = new DataView(buf), hlen = dv.getUint32(0, true);
    const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hlen)));
    const start = 4 + hlen, n = (buf.byteLength - start) / 2, u16 = new Uint16Array(buf, start, n), all = new Float32Array(n);
    for (let i = 0; i < n; i++) all[i] = f16to32(u16[i]);
    const W = {};
    for (const k in header) { const [shape, off] = header[k]; const size = shape.reduce((a, b) => a * b, 1); W[k] = { shape, data: all.subarray(off, off + size) }; }
    return W;
  }

  // ---------------------------------------------------------------- math kernels (row-major [T, C])
  const GELU = (x) => 0.5 * x * (1 + erf(x / Math.SQRT2));
  function erf(x) { // Abramowitz-Stegun 7.1.26 (|err| < 1.5e-7)
    const s = x < 0 ? -1 : 1; x = Math.abs(x); const t = 1 / (1 + 0.3275911 * x);
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return s * y;
  }
  function linear(x, T, cin, w, b, cout) { // y[T,cout] = x[T,cin] W^T + b, W [cout,cin]
    const y = new Float32Array(T * cout), wd = w.data, bd = b ? b.data : null;
    for (let t = 0; t < T; t++) { const xo = t * cin, yo = t * cout;
      for (let o = 0; o < cout; o++) { let s = bd ? bd[o] : 0; const wo = o * cin; for (let i = 0; i < cin; i++) s += x[xo + i] * wd[wo + i]; y[yo + o] = s; } }
    return y;
  }
  function layerNorm(x, T, C, g, b, eps = 1e-5) {
    const y = new Float32Array(T * C), gd = g.data, bd = b.data;
    for (let t = 0; t < T; t++) { const o = t * C; let m = 0; for (let c = 0; c < C; c++) m += x[o + c]; m /= C;
      let v = 0; for (let c = 0; c < C; c++) { const d = x[o + c] - m; v += d * d; } v /= C; const inv = 1 / Math.sqrt(v + eps);
      for (let c = 0; c < C; c++) y[o + c] = (x[o + c] - m) * inv * gd[c] + bd[c]; }
    return y;
  }
  function addInPlace(a, b) { for (let i = 0; i < a.length; i++) a[i] += b[i]; return a; }
  function geluInPlace(a) { for (let i = 0; i < a.length; i++) a[i] = GELU(a[i]); return a; }
  function mha(x, T, inW, inB, outW, outB) { // nn.MultiheadAttention, batch_first, no mask
    const qkv = linear(x, T, D, inW, inB, 3 * D), dh = D / HEADS, scale = 1 / Math.sqrt(dh), ctx = new Float32Array(T * D), sc = new Float32Array(T);
    for (let h = 0; h < HEADS; h++) { const qo = h * dh, ko = D + h * dh, vo = 2 * D + h * dh;
      for (let i = 0; i < T; i++) { let mx = -Infinity; const qi = i * 3 * D + qo;
        for (let j = 0; j < T; j++) { let s = 0; const kj = j * 3 * D + ko; for (let d = 0; d < dh; d++) s += qkv[qi + d] * qkv[kj + d]; s *= scale; sc[j] = s; if (s > mx) mx = s; }
        let sum = 0; for (let j = 0; j < T; j++) { sc[j] = Math.exp(sc[j] - mx); sum += sc[j]; }
        const co = i * D + qo; for (let j = 0; j < T; j++) { const p = sc[j] / sum, vj = j * 3 * D + vo; for (let d = 0; d < dh; d++) ctx[co + d] += p * qkv[vj + d]; } } }
    return linear(ctx, T, D, outW, outB, D);
  }
  function posEnc(T) { const pe = new Float32Array(T * D); for (let p = 0; p < T; p++) for (let i = 0; i < D; i += 2) { const div = Math.exp(i * (-Math.log(10000) / D)); pe[p * D + i] = Math.sin(p * div); pe[p * D + i + 1] = Math.cos(p * div); } return pe; }
  function transformerLayer(x, T, W, pfx) { // post-norm nn.TransformerEncoderLayer, GELU
    const a = mha(x, T, W[pfx + "self_attn.in_proj_weight"], W[pfx + "self_attn.in_proj_bias"], W[pfx + "self_attn.out_proj.weight"], W[pfx + "self_attn.out_proj.bias"]);
    let h = layerNorm(addInPlace(a, x), T, D, W[pfx + "norm1.weight"], W[pfx + "norm1.bias"]);
    const f = linear(geluInPlace(linear(h, T, D, W[pfx + "linear1.weight"], W[pfx + "linear1.bias"], DFF)), T, DFF, W[pfx + "linear2.weight"], W[pfx + "linear2.bias"], D);
    return layerNorm(addInPlace(f, h), T, D, W[pfx + "norm2.weight"], W[pfx + "norm2.bias"]);
  }
  function conv1dSame(x, T, cin, w, b, cout, k) { // x [T,cin] (row-major), w [cout,cin,k], pad k//2 -> y [T,cout]
    const y = new Float32Array(T * cout), wd = w.data, bd = b.data, p = k >> 1;
    for (let o = 0; o < cout; o++) for (let t = 0; t < T; t++) { let s = bd[o];
      for (let i = 0; i < cin; i++) { const wo = (o * cin + i) * k; for (let kk = 0; kk < k; kk++) { const tt = t + kk - p; if (tt >= 0 && tt < T) s += x[tt * cin + i] * wd[wo + kk]; } }
      y[t * cout + o] = s; }
    return y;
  }
  function batchNormEval(x, T, C, W, pfx) { const g = W[pfx + "weight"].data, b = W[pfx + "bias"].data, m = W[pfx + "running_mean"].data, v = W[pfx + "running_var"].data;
    for (let t = 0; t < T; t++) for (let c = 0; c < C; c++) x[t * C + c] = (x[t * C + c] - m[c]) / Math.sqrt(v[c] + 1e-5) * g[c] + b[c]; return x; }

  // ---------------------------------------------------------------- TTS
  function ttsInfer(W, ids, maxMelLen) {
    const T = ids.length, emb = W["tts.tok_emb.weight"].data, x = new Float32Array(T * D);
    for (let t = 0; t < T; t++) x.set(emb.subarray(ids[t] * D, ids[t] * D + D), t * D);
    let h = addInPlace(x, posEnc(T));
    for (let l = 0; l < 4; l++) h = transformerLayer(h, T, W, "tts.encoder.layers." + l + ".");
    // duration predictor
    let dp = geluInPlace(conv1dSame(h, T, D, W["tts.dur_pred.conv.0.weight"], W["tts.dur_pred.conv.0.bias"], D, 3));
    dp = batchNormEval(dp, T, D, W, "tts.dur_pred.conv.2.");
    dp = geluInPlace(conv1dSame(dp, T, D, W["tts.dur_pred.conv.3.weight"], W["tts.dur_pred.conv.3.bias"], D, 3));
    dp = batchNormEval(dp, T, D, W, "tts.dur_pred.conv.5.");
    const ldp = linear(dp, T, D, W["tts.dur_pred.proj.weight"], W["tts.dur_pred.proj.bias"], 1);
    const dur = Array.from(ldp, (v) => Math.max(1, Math.round(Math.exp(v) - 1)));
    const total = dur.reduce((a, b) => a + b, 0), Lm = Math.min(total, maxMelLen), Tpad = maxMelLen; // decoder runs over the padded length, as in TTSModel.infer
    const xr = new Float32Array(Tpad * D); let f = 0;
    for (let t = 0; t < T && f < Lm; t++) for (let r = 0; r < dur[t] && f < Lm; r++, f++) xr.set(h.subarray(t * D, t * D + D), f * D);
    let d = addInPlace(xr, posEnc(Tpad));
    for (let l = 0; l < 4; l++) d = transformerLayer(d, Tpad, W, "tts.decoder.layers." + l + ".");
    const melT = linear(d, Tpad, D, W["tts.mel_proj.weight"], W["tts.mel_proj.bias"], N_MELS); // [Tpad, 40]
    const mel = new Float32Array(N_MELS * Lm); // [40, Lm]
    for (let t = 0; t < Lm; t++) for (let m = 0; m < N_MELS; m++) mel[m * Lm + t] = melT[t * N_MELS + m];
    return { mel, T: Lm, durations: dur, logDur: Array.from(ldp) };
  }

  // ---------------------------------------------------------------- FFT / STFT / mel / Griffin-Lim
  const HANN = new Float32Array(N_FFT); for (let n = 0; n < N_FFT; n++) HANN[n] = 0.5 - 0.5 * Math.cos(2 * Math.PI * n / N_FFT); // periodic Hann (torch default)
  const BITREV = new Uint16Array(N_FFT); { let bits = 9; for (let i = 0; i < N_FFT; i++) { let r = 0, v = i; for (let b = 0; b < bits; b++) { r = (r << 1) | (v & 1); v >>= 1; } BITREV[i] = r; } }
  const COS = new Float32Array(N_FFT / 2), SIN = new Float32Array(N_FFT / 2); for (let i = 0; i < N_FFT / 2; i++) { COS[i] = Math.cos(2 * Math.PI * i / N_FFT); SIN[i] = Math.sin(2 * Math.PI * i / N_FFT); }
  function fft(re, im, inverse) { // in-place radix-2, length N_FFT
    for (let i = 0; i < N_FFT; i++) { const j = BITREV[i]; if (j > i) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
    for (let size = 2; size <= N_FFT; size <<= 1) { const half = size >> 1, step = N_FFT / size;
      for (let start = 0; start < N_FFT; start += size) for (let k = 0; k < half; k++) { const wr = COS[k * step], wi = (inverse ? 1 : -1) * SIN[k * step];
        const a = start + k, b = a + half, tr = re[b] * wr - im[b] * wi, ti = re[b] * wi + im[b] * wr; re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti; } }
    if (inverse) for (let i = 0; i < N_FFT; i++) { re[i] /= N_FFT; im[i] /= N_FFT; }
  }
  function reflectPad(x, p) { const n = x.length, y = new Float32Array(n + 2 * p); for (let i = 0; i < p; i++) y[i] = x[p - i]; y.set(x, p); for (let i = 0; i < p; i++) y[p + n + i] = x[n - 2 - i]; return y; }
  function stft(wav) { // center=True reflect -> {re, im} arrays [T][N_FREQ]
    const p = N_FFT >> 1, xp = reflectPad(wav, p), T = 1 + Math.floor(wav.length / HOP), re = new Float32Array(T * N_FREQ), im = new Float32Array(T * N_FREQ), fr = new Float32Array(N_FFT), fi = new Float32Array(N_FFT);
    for (let t = 0; t < T; t++) { const o = t * HOP; for (let n = 0; n < N_FFT; n++) { fr[n] = xp[o + n] * HANN[n]; fi[n] = 0; } fft(fr, fi, false);
      for (let k = 0; k < N_FREQ; k++) { re[t * N_FREQ + k] = fr[k]; im[t * N_FREQ + k] = fi[k]; } }
    return { re, im, T };
  }
  function istft(re, im, T) { // center=True, window-sum-square normalisation, length = HOP*(T-1)
    const L = HOP * (T - 1), p = N_FFT >> 1, y = new Float32Array(L + N_FFT), wsum = new Float32Array(L + N_FFT), fr = new Float32Array(N_FFT), fi = new Float32Array(N_FFT);
    for (let t = 0; t < T; t++) { for (let k = 0; k < N_FREQ; k++) { fr[k] = re[t * N_FREQ + k]; fi[k] = im[t * N_FREQ + k]; }
      for (let k = 1; k < N_FFT / 2; k++) { fr[N_FFT - k] = fr[k]; fi[N_FFT - k] = -fi[k]; } fi[0] = 0; fi[N_FFT / 2] = 0; fft(fr, fi, true);
      const o = t * HOP; for (let n = 0; n < N_FFT; n++) { y[o + n] += fr[n] * HANN[n]; wsum[o + n] += HANN[n] * HANN[n]; } }
    const out = new Float32Array(L); for (let n = 0; n < L; n++) { const w = wsum[n + p]; out[n] = w > 1e-11 ? y[n + p] / w : y[n + p]; } return out;
  }
  let melFB = null, melPinv = null; // set via setConstants
  function melFromWav(wav) { // log1p(mel power) -> [40, T]
    const S = stft(wav), T = S.T, mel = new Float32Array(N_MELS * T);
    for (let t = 0; t < T; t++) { for (let m = 0; m < N_MELS; m++) { let s = 0; for (let k = 0; k < N_FREQ; k++) { const r = S.re[t * N_FREQ + k], i = S.im[t * N_FREQ + k]; s += (r * r + i * i) * melFB[k * N_MELS + m]; } mel[m * T + t] = Math.log1p(s); } }
    return { mel, T };
  }
  function griffinLim(mel, T, nIter = 32, momentum = 0.99, rng = Math.random) { // mel [40,T] log1p -> waveform
    const mag = new Float32Array(T * N_FREQ);
    for (let t = 0; t < T; t++) for (let k = 0; k < N_FREQ; k++) { let s = 0; for (let m = 0; m < N_MELS; m++) s += melPinv[k * N_MELS + m] * Math.expm1(Math.min(mel[m * T + t], 10)); mag[t * N_FREQ + k] = Math.sqrt(Math.max(s, 0)); }
    let re = new Float32Array(T * N_FREQ), im = new Float32Array(T * N_FREQ), pr = new Float32Array(T * N_FREQ), pi = new Float32Array(T * N_FREQ);
    for (let i = 0; i < re.length; i++) { const a = rng() * 2 * Math.PI; re[i] = mag[i] * Math.cos(a); im[i] = mag[i] * Math.sin(a); }
    for (let it = 0; it < nIter; it++) {
      const w = istft(re, im, T), R = stft(w), c = momentum / (1 + momentum);
      for (let i = 0; i < re.length; i++) { let ar = R.re[i] - c * pr[i], ai = R.im[i] - c * pi[i]; const n = Math.hypot(ar, ai) + 1e-16; pr[i] = R.re[i]; pi[i] = R.im[i]; re[i] = mag[i] * ar / n; im[i] = mag[i] * ai / n; }
    }
    return istft(re, im, T);
  }

  // ---------------------------------------------------------------- ASR
  function asrLogits(W, mel, T) { // mel [40,T] -> {logits [T',128], Tsub}
    const H1 = Math.ceil(N_MELS / 2), T1 = Math.ceil(T / 2), H2 = Math.ceil(H1 / 2), T2 = Math.ceil(T1 / 2);
    function conv2d(inp, cin, hIn, tIn, w, b, cout) { const hOut = Math.ceil(hIn / 2), tOut = Math.ceil(tIn / 2), out = new Float32Array(cout * hOut * tOut), wd = w.data, bd = b.data;
      for (let o = 0; o < cout; o++) for (let y = 0; y < hOut; y++) for (let x = 0; x < tOut; x++) { let s = bd[o];
        for (let i = 0; i < cin; i++) for (let ky = 0; ky < 3; ky++) { const yy = 2 * y + ky - 1; if (yy < 0 || yy >= hIn) continue; for (let kx = 0; kx < 3; kx++) { const xx = 2 * x + kx - 1; if (xx < 0 || xx >= tIn) continue; s += inp[(i * hIn + yy) * tIn + xx] * wd[((o * cin + i) * 3 + ky) * 3 + kx]; } }
        out[(o * hOut + y) * tOut + x] = GELU(s); }
      return out; }
    const c1 = conv2d(mel, 1, N_MELS, T, W["asr.subsampling.conv.0.weight"], W["asr.subsampling.conv.0.bias"], 32);
    const c2 = conv2d(c1, 32, H1, T1, W["asr.subsampling.conv.2.weight"], W["asr.subsampling.conv.2.bias"], 32);
    const flat = new Float32Array(T2 * 32 * H2); // [T2, C*Fr] with index c*H2 + f
    for (let t = 0; t < T2; t++) for (let c = 0; c < 32; c++) for (let f = 0; f < H2; f++) flat[t * 32 * H2 + c * H2 + f] = c2[(c * H2 + f) * T2 + t];
    let x = addInPlace(linear(flat, T2, 32 * H2, W["asr.subsampling.proj.weight"], W["asr.subsampling.proj.bias"], D), posEnc(T2));
    for (let l = 0; l < 4; l++) { const p = "asr.blocks." + l + ".";
      const ff = (pf, inp) => linear(geluInPlace(linear(layerNorm(inp, T2, D, W[pf + "0.weight"], W[pf + "0.bias"]), T2, D, W[pf + "1.weight"], W[pf + "1.bias"], DFF)), T2, DFF, W[pf + "4.weight"], W[pf + "4.bias"], D);
      let f1 = ff(p + "ff1.", x); for (let i = 0; i < x.length; i++) x[i] += 0.5 * f1[i];
      const xn = layerNorm(x, T2, D, W[p + "attn_norm.weight"], W[p + "attn_norm.bias"]);
      const a = mha(xn, T2, W[p + "attn.in_proj_weight"], W[p + "attn.in_proj_bias"], W[p + "attn.out_proj.weight"], W[p + "attn.out_proj.bias"]);
      addInPlace(x, a);
      // depthwise-separable conv module
      const dw = W[p + "conv.depthwise.weight"].data, db = W[p + "conv.depthwise.bias"].data, y = new Float32Array(T2 * D);
      for (let c = 0; c < D; c++) for (let t = 0; t < T2; t++) { let s = db[c]; for (let k = 0; k < 15; k++) { const tt = t + k - 7; if (tt >= 0 && tt < T2) s += x[tt * D + c] * dw[c * 15 + k]; } y[t * D + c] = s; }
      const pw = linear(y, T2, D, W[p + "conv.pointwise.weight"], W[p + "conv.pointwise.bias"], D);
      const cn = geluInPlace(layerNorm(pw, T2, D, W[p + "conv.norm.weight"], W[p + "conv.norm.bias"]));
      addInPlace(x, cn);
      const f2 = ff(p + "ff2.", x); for (let i = 0; i < x.length; i++) x[i] += 0.5 * f2[i];
      x = layerNorm(x, T2, D, W[p + "final_norm.weight"], W[p + "final_norm.bias"]);
    }
    return { logits: linear(x, T2, D, W["asr.proj.weight"], W["asr.proj.bias"], VOCAB), Tsub: T2 };
  }
  function ctcGreedy(logits, Tsub, validFrames) { const out = []; let prev = -1; const n = Math.min(Tsub, validFrames);
    for (let t = 0; t < n; t++) { let best = 0, bv = -Infinity; for (let v = 0; v < VOCAB; v++) { const s = logits[t * VOCAB + v]; if (s > bv) { bv = s; best = v; } } if (best !== prev && best !== 0) out.push(best); prev = best; }
    return out; }
  function asrDecode(W, mel, T) { const { logits, Tsub } = asrLogits(W, mel, T); return ctcGreedy(logits, Tsub, Math.max(1, Math.floor(T / 4))); }

  // ---------------------------------------------------------------- procedural synthesizer
  let PS = null;
  function psSynth(ids) { const n = 960, dur = 0.06, fade = 80, out = new Float32Array(n * ids.length);
    for (let j = 0; j < ids.length; j++) { const tok = ids[j], o = j * n;
      for (let i = 0; i < n; i++) { const t = i * dur / (n - 1); let w = 0; for (let h = 0; h < 3; h++) w += PS.amps[tok][h] * Math.sin(2 * Math.PI * PS.freqs[tok][h] * t + PS.phases[tok][h]);
        if (i < fade) w *= i / (fade - 1); if (i >= n - fade) w *= (n - 1 - i) / (fade - 1); out[o + i] = w; } }
    return out; }

  // ---------------------------------------------------------------- channel simulation
  function mulberry32(seed) { return function () { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  function gauss(rng) { let u = 0, v = 0; while (u === 0) u = rng(); while (v === 0) v = rng(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
  function unitRms(x) { let s = 0; for (let i = 0; i < x.length; i++) s += x[i] * x[i]; const r = Math.sqrt(s / x.length) || 1e-8; for (let i = 0; i < x.length; i++) x[i] /= r; return x; }
  function noise(kind, n, rng) {
    const white = new Float32Array(n); for (let i = 0; i < n; i++) white[i] = gauss(rng);
    if (kind === "white") return unitRms(white);
    if (kind === "pink") { const b = [0.049922035, -0.095993537, 0.050612699, -0.004709510], a = [1, -2.494956002, 2.017265875, -0.522189400], y = new Float32Array(n);
      for (let i = 0; i < n; i++) { let s = 0; for (let k = 0; k < 4; k++) if (i - k >= 0) s += b[k] * white[i - k]; for (let k = 1; k < 4; k++) if (i - k >= 0) s -= a[k] * y[i - k]; y[i] = s; } return unitRms(y); }
    if (kind === "brown") { const y = new Float32Array(n); let c = 0, m = 0; for (let i = 0; i < n; i++) { c += white[i]; y[i] = c; m += c; } m /= n; for (let i = 0; i < n; i++) y[i] -= m; return unitRms(y); }
    const w = unitRms(white), p = noise("pink", n, rng), br = noise("brown", n, rng), y = new Float32Array(n); for (let i = 0; i < n; i++) y[i] = w[i] + p[i] + br[i]; return unitRms(y);
  }
  function addNoise(wav, snrDb, kind, rng) { if (snrDb === null || snrDb === undefined) return wav; let sp = 0; for (let i = 0; i < wav.length; i++) sp += wav[i] * wav[i]; sp = Math.max(sp / wav.length, 1e-10);
    const nz = noise(kind, wav.length, rng), g = Math.sqrt(sp / Math.pow(10, snrDb / 10)), out = new Float32Array(wav.length); for (let i = 0; i < wav.length; i++) out[i] = wav[i] + g * nz[i]; return out; }
  function reverb(wav, rng, decay = 0.4, mix = 0.3, predelayS = 0.02, irLenS = 0.1) { const irLen = Math.round(SR * irLenS), pre = Math.round(SR * predelayS), ir = new Float32Array(irLen); let l1 = 0;
    for (let i = pre; i < irLen; i++) { const tn = (i - pre) / (irLen - pre - 1); ir[i] = gauss(rng) * Math.exp(-5 * decay * tn); l1 += Math.abs(ir[i]); } for (let i = pre; i < irLen; i++) ir[i] /= (l1 || 1); ir[0] = 1;
    const out = new Float32Array(wav.length); for (let n = 0; n < wav.length; n++) { let s = 0; const kmax = Math.min(irLen, n + 1); for (let k = 0; k < kmax; k++) s += wav[n - k] * ir[k]; out[n] = (1 - mix) * wav[n] + mix * s; } return out; }
  function clip(wav, thr) { let pk = 1e-8; for (let i = 0; i < wav.length; i++) pk = Math.max(pk, Math.abs(wav[i])); const out = new Float32Array(wav.length); for (let i = 0; i < wav.length; i++) out[i] = Math.max(-thr, Math.min(thr, wav[i] / pk)) * pk; return out; }
  function resampleLinear(wav, ratio) { const n = Math.max(2, Math.round(wav.length * ratio)), out = new Float32Array(n), m = wav.length; // matches F.interpolate(align_corners=False)
    for (let i = 0; i < n; i++) { const src = (i + 0.5) * m / n - 0.5, i0 = Math.floor(src); const f = src - i0; const a = wav[Math.min(Math.max(i0, 0), m - 1)], b = wav[Math.min(Math.max(i0 + 1, 0), m - 1)]; out[i] = a + (b - a) * (i0 < 0 ? 0 : (i0 + 1 > m - 1 ? 0 : f)); } return out; }
  function applyChannel(wav, cond, rng) { let w = wav;
    if (cond.reverb) w = reverb(w, rng, cond.reverbDecay ?? 0.4, cond.reverbMix ?? 0.3);
    if (cond.clip) w = clip(w, cond.clipThr ?? 0.5);
    if (cond.drift) w = resampleLinear(w, cond.driftRatio ?? 1.01);
    if (cond.snr !== null && cond.snr !== undefined) w = addNoise(w, cond.snr, cond.noise || "mixed", rng);
    return w; }

  function setConstants(c) { melFB = new Float32Array(N_FREQ * N_MELS); melPinv = new Float32Array(N_FREQ * N_MELS);
    for (let k = 0; k < N_FREQ; k++) for (let m = 0; m < N_MELS; m++) { melFB[k * N_MELS + m] = c.mel_fb[k][m]; melPinv[k * N_MELS + m] = c.mel_pinv[k][m]; }
    PS = { freqs: c.ps_freqs, amps: c.ps_amps, phases: c.ps_phases }; }

  return { SR, HOP, N_MELS, encode, decode, editDistance, loadContainer, setConstants, ttsInfer, griffinLim, melFromWav, asrDecode, asrLogits, psSynth, applyChannel, addNoise, mulberry32, CMDS };
});
