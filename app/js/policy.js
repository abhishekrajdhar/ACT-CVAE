// Pure-JS forward pass for the `language_conditioned_act_cvae` export from train.py.
//
//   image ─CNN(4× stride-2)→ 8×8×64 ─proj+pos→ 64 spatial tokens
//   [spatial(64), lang(1), z(1)] → TransformerEncoder → memory
//   (action_queries + lang) ─TransformerDecoder(memory)→ action_head → K×4 logits
//
// At inference z = 0, so the z token is just z_proj.bias.
// Mirrors PyTorch's post-norm TransformerEncoderLayer/DecoderLayer (ReLU, eps 1e-5).

const sigmoid = (x) => 1 / (1 + Math.exp(-x));

export class Policy {
  constructor(json) {
    if (json.format !== 'language_conditioned_act_cvae' && json.format !== 'language_conditioned_act') {
      throw new Error(`Unsupported model format "${json.format}"`);
    }
    this.meta = json;
    this.D = json.d_model;
    this.heads = json.n_heads;
    this.K = json.chunk_size;
    this.A = json.action_dim;
    this.size = json.img_size;
    this.convLayers = json.n_conv_layers ?? 3;
    this.nSpatial = json.n_spatial_tokens;
    this.useZ = !!json.use_vae && json.latent_dim > 0;
    this.intents = json.intent_labels;
    this.w = {};
    for (const [k, v] of Object.entries(json.weights)) this.w[k] = Float32Array.from(v.data);
  }

  static async fromURL(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status} loading ${url}`);
    return new Policy(await res.json());
  }

  static async fromFile(file) {
    return new Policy(JSON.parse(await file.text()));
  }

  get summary() {
    const m = this.meta;
    return `${(m.num_params / 1000).toFixed(0)}K params · ${m.img_size}px · chunk ${m.chunk_size}${this.useZ ? ' · CVAE' : ''}`;
  }

  // image: Float32Array CHW in [0,1]. Returns K arrays of 4 probabilities
  // [forward, backward, left, right].
  predict(image, intent) {
    const logits = this.forward(image, intent);
    const out = [];
    for (let k = 0; k < this.K; k++) {
      out.push(Array.from(logits.subarray(k * this.A, (k + 1) * this.A), sigmoid));
    }
    return out;
  }

  forward(image, intent) {
    const { w, D } = this;
    // CNN backbone
    let x = image, c = 3, s = this.size;
    const widths = [16, 32, 64, 64].slice(0, this.convLayers);
    widths.forEach((outC, i) => {
      x = conv3x3s2Relu(x, c, s, w[`conv${i + 1}.weight`], w[`conv${i + 1}.bias`], outC);
      c = outC;
      s /= 2;
    });

    // Spatial tokens: feature map [C, S] → [S, D]
    const S = this.nSpatial;
    const feats = new Float32Array(S * c);
    for (let ch = 0; ch < c; ch++) for (let p = 0; p < S; p++) feats[p * c + ch] = x[ch * S + p];
    const spatial = linear(feats, S, c, D, w['spatial_proj.weight'], w['spatial_proj.bias']);
    addInPlace(spatial, w['spatial_pos']);

    const lang = w['lang_embed.weight'].subarray(intent * D, (intent + 1) * D);
    const nTok = S + 1 + (this.useZ ? 1 : 0);
    let mem = new Float32Array(nTok * D);
    mem.set(spatial, 0);
    mem.set(lang, S * D);
    if (this.useZ) mem.set(w['z_proj.bias'], (S + 1) * D);

    for (let l = 0; l < this.meta.n_enc_layers; l++) mem = this._encoder(mem, nTok, `transformer_encoder.layers.${l}`);

    const K = this.K;
    let q = new Float32Array(K * D);
    const queries = w['action_queries'];
    for (let k = 0; k < K; k++) for (let d = 0; d < D; d++) q[k * D + d] = queries[k * D + d] + lang[d];
    for (let l = 0; l < this.meta.n_dec_layers; l++) q = this._decoder(q, K, mem, nTok, `transformer_decoder.layers.${l}`);

    const h = linear(q, K, D, D, w['action_head.0.weight'], w['action_head.0.bias']);
    reluInPlace(h);
    return linear(h, K, D, this.A, w['action_head.2.weight'], w['action_head.2.bias']);
  }

  _encoder(x, n, p) {
    const a = this._attn(x, n, x, n, `${p}.self_attn`);
    addInPlace(a, x);
    const h = this._norm(a, n, `${p}.norm1`);
    const f = this._ffn(h, n, p);
    addInPlace(f, h);
    return this._norm(f, n, `${p}.norm2`);
  }

  _decoder(x, n, mem, nMem, p) {
    const a = this._attn(x, n, x, n, `${p}.self_attn`);
    addInPlace(a, x);
    const h1 = this._norm(a, n, `${p}.norm1`);
    const c = this._attn(h1, n, mem, nMem, `${p}.multihead_attn`);
    addInPlace(c, h1);
    const h2 = this._norm(c, n, `${p}.norm2`);
    const f = this._ffn(h2, n, p);
    addInPlace(f, h2);
    return this._norm(f, n, `${p}.norm3`);
  }

  _ffn(x, n, p) {
    const { w, D } = this;
    const hidden = w[`${p}.linear1.bias`].length;
    const h = linear(x, n, D, hidden, w[`${p}.linear1.weight`], w[`${p}.linear1.bias`]);
    reluInPlace(h);
    return linear(h, n, hidden, D, w[`${p}.linear2.weight`], w[`${p}.linear2.bias`]);
  }

  _norm(x, n, p) {
    const { w, D } = this;
    const g = w[`${p}.weight`], b = w[`${p}.bias`];
    const out = new Float32Array(n * D);
    for (let t = 0; t < n; t++) {
      const o = t * D;
      let mean = 0;
      for (let i = 0; i < D; i++) mean += x[o + i];
      mean /= D;
      let v = 0;
      for (let i = 0; i < D; i++) v += (x[o + i] - mean) ** 2;
      const inv = 1 / Math.sqrt(v / D + 1e-5);
      for (let i = 0; i < D; i++) out[o + i] = g[i] * (x[o + i] - mean) * inv + b[i];
    }
    return out;
  }

  // nn.MultiheadAttention with packed in_proj weights (rows: Q, K, V).
  _attn(qIn, nQ, kvIn, nKV, p) {
    const { w, D, heads } = this;
    const W = w[`${p}.in_proj_weight`], B = w[`${p}.in_proj_bias`];
    const Q = linear(qIn, nQ, D, D, W.subarray(0, D * D), B.subarray(0, D));
    const Kt = linear(kvIn, nKV, D, D, W.subarray(D * D, 2 * D * D), B.subarray(D, 2 * D));
    const Vt = linear(kvIn, nKV, D, D, W.subarray(2 * D * D), B.subarray(2 * D));
    const hd = D / heads, scale = 1 / Math.sqrt(hd);
    const ctx = new Float32Array(nQ * D);
    const score = new Float32Array(nKV);
    for (let h = 0; h < heads; h++) {
      const off = h * hd;
      for (let i = 0; i < nQ; i++) {
        let max = -Infinity;
        for (let j = 0; j < nKV; j++) {
          let s = 0;
          for (let d = 0; d < hd; d++) s += Q[i * D + off + d] * Kt[j * D + off + d];
          score[j] = s * scale;
          if (score[j] > max) max = score[j];
        }
        let sum = 0;
        for (let j = 0; j < nKV; j++) sum += (score[j] = Math.exp(score[j] - max));
        for (let d = 0; d < hd; d++) {
          let acc = 0;
          for (let j = 0; j < nKV; j++) acc += score[j] * Vt[j * D + off + d];
          ctx[i * D + off + d] = acc / sum;
        }
      }
    }
    return linear(ctx, nQ, D, D, w[`${p}.out_proj.weight`], w[`${p}.out_proj.bias`]);
  }
}

// y[n, out] = x[n, in] · Wᵀ + b   (W is PyTorch layout [out, in])
function linear(x, n, inD, outD, W, b) {
  const y = new Float32Array(n * outD);
  for (let t = 0; t < n; t++) {
    const xo = t * inD;
    for (let o = 0; o < outD; o++) {
      const wo = o * inD;
      let s = b[o];
      for (let i = 0; i < inD; i++) s += x[xo + i] * W[wo + i];
      y[t * outD + o] = s;
    }
  }
  return y;
}

// 3×3 conv, stride 2, padding 1, followed by ReLU. Input CHW, square.
function conv3x3s2Relu(x, inC, size, W, b, outC) {
  const out = size / 2, plane = size * size;
  const y = new Float32Array(outC * out * out);
  for (let oc = 0; oc < outC; oc++) {
    const wBase = oc * inC * 9;
    for (let oy = 0; oy < out; oy++) {
      for (let ox = 0; ox < out; ox++) {
        let s = b[oc];
        for (let ic = 0; ic < inC; ic++) {
          const xBase = ic * plane, wk = wBase + ic * 9;
          for (let ky = 0; ky < 3; ky++) {
            const iy = oy * 2 + ky - 1;
            if (iy < 0 || iy >= size) continue;
            const row = xBase + iy * size;
            for (let kx = 0; kx < 3; kx++) {
              const ix = ox * 2 + kx - 1;
              if (ix < 0 || ix >= size) continue;
              s += x[row + ix] * W[wk + ky * 3 + kx];
            }
          }
        }
        y[(oc * out + oy) * out + ox] = s > 0 ? s : 0;
      }
    }
  }
  return y;
}

function addInPlace(a, b) { for (let i = 0; i < a.length; i++) a[i] += b[i]; }
function reluInPlace(a) { for (let i = 0; i < a.length; i++) if (a[i] < 0) a[i] = 0; }
