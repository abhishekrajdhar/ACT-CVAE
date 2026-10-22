"""
Language-Conditioned ACT + CVAE Training -- Fork Road 128px
CNN spatial tokens + language token + CVAE latent z + Transformer + action chunking.

During training: encoder(action_chunk) -> mu, logvar -> z -> injected into transformer
During inference: z = 0 (prior mean)

Encoder input: [spatial_tokens(64), lang_token(1), z_token(1)] = 66 tokens
Action queries get lang_token added (language injection).

Two-phase training: SGD to escape mean, then Adam to refine.
"""
import json, os, glob, base64, io, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from collections import Counter

IMG_SIZE = 128
CHUNK_SIZE = 5
ACTION_DIM = 4
D_MODEL = 64
N_HEADS = 2
N_ENC_LAYERS = 1
N_DEC_LAYERS = 1
FFN_MULT = 2
N_SPATIAL = 64  # 8x8 after 4 conv layers
LATENT_DIM = 16
KL_WEIGHT = 0.01  # beta for KL term
EPOCHS = 200
BATCH = 16

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
torch.manual_seed(42)
np.random.seed(42)

# Load data
DATA_DIR = os.environ.get('DATA_DIR', 'data')
files = sorted(glob.glob(os.path.join(DATA_DIR, '*.json')))
all_samples = []
seen_ts = set()
intent_labels = None
for f in files:
    with open(f) as fh:
        d = json.load(fh)
    if intent_labels is None:
        intent_labels = d.get('metadata', {}).get('intent_labels', ['always_take_left', 'always_take_right'])
    for s in d.get('samples', []):
        ts = s.get('timestamp', id(s))
        if ts not in seen_ts:
            all_samples.append(s)
            seen_ts.add(ts)

n = len(all_samples)
num_intents = max(s['language_id'] for s in all_samples) + 1
print(f"Samples: {n}, Intents: {num_intents} {intent_labels}")
print(f"Distribution: {Counter(s['language_id'] for s in all_samples)}")

# Sort by intent then timestamp for chunking
samples_by_intent = {}
for s in all_samples:
    lid = s['language_id']
    if lid not in samples_by_intent:
        samples_by_intent[lid] = []
    samples_by_intent[lid].append(s)
for lid in samples_by_intent:
    samples_by_intent[lid].sort(key=lambda x: x['timestamp'])

sorted_samples = []
for lid in sorted(samples_by_intent.keys()):
    sorted_samples.extend(samples_by_intent[lid])

# Decode
print("Decoding images...")
images = torch.zeros(n, 3, IMG_SIZE, IMG_SIZE)
langs = torch.zeros(n, dtype=torch.long)
actions = torch.zeros(n, ACTION_DIM)
for i, s in enumerate(sorted_samples):
    b64 = s['image'].split(',')[1] if ',' in s['image'] else s['image']
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    images[i] = torch.from_numpy(np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0)
    langs[i] = s['language_id']
    actions[i] = torch.tensor([s['actions']['forward'], s['actions']['backward'],
                                s['actions']['left'], s['actions']['right']], dtype=torch.float32)

fwd = actions[:, 0].mean(); lft = actions[:, 2].mean(); rgt = actions[:, 3].mean()
print(f"fwd={fwd:.2f} left={lft:.2f} right={rgt:.2f}")

# Build chunks per intent
chunk_indices = []
for lid in sorted(samples_by_intent.keys()):
    intent_idx = [i for i in range(n) if langs[i].item() == lid]
    for start in range(len(intent_idx) - CHUNK_SIZE + 1):
        chunk_indices.append((intent_idx[start], intent_idx[start:start + CHUNK_SIZE]))
print(f"Chunks: {len(chunk_indices)} (K={CHUNK_SIZE})")

# Dataset
class DS(Dataset):
    def __init__(self, chunks, imgs, lngs, acts, aug=False):
        self.chunks, self.imgs, self.lngs, self.acts, self.aug = chunks, imgs, lngs, acts, aug
    def __len__(self): return len(self.chunks)
    def __getitem__(self, i):
        img_idx, chunk_idx = self.chunks[i]
        img = self.imgs[img_idx].clone()
        lng = self.lngs[img_idx]
        act_chunk = self.acts[chunk_idx]
        if self.aug:
            img = img * (0.7 + torch.rand(1).item() * 0.6)
            mean = img.mean()
            img = (img - mean) * (0.7 + torch.rand(1).item() * 0.6) + mean
            img = img + torch.randn_like(img) * 0.03
            for c in range(3): img[c] = img[c] * (0.9 + torch.rand(1).item() * 0.2)
            img = img.clamp(0, 1)
        return img, lng, act_chunk

np.random.shuffle(chunk_indices)
sp = int(len(chunk_indices) * 0.85)
train_dl = DataLoader(DS(chunk_indices[:sp], images, langs, actions, aug=True),
                      batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True)
val_dl = DataLoader(DS(chunk_indices[sp:], images, langs, actions),
                    batch_size=64, num_workers=2, pin_memory=True)
print(f"Train: {sp}, Val: {len(chunk_indices)-sp}")

# Model
class LangConditionedACT_CVAE(nn.Module):
    def __init__(self):
        super().__init__()
        # Vision CNN
        self.conv1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.relu = nn.ReLU()

        # Spatial tokens
        self.spatial_proj = nn.Linear(64, D_MODEL)
        self.spatial_pos = nn.Parameter(torch.randn(N_SPATIAL, D_MODEL) * 0.02)

        # Language embedding
        self.lang_embed = nn.Embedding(num_intents, D_MODEL)

        # CVAE encoder: action chunk -> mu, logvar
        self.cvae_encoder = nn.Sequential(
            nn.Linear(ACTION_DIM * CHUNK_SIZE, D_MODEL),
            nn.ReLU(),
            nn.Linear(D_MODEL, D_MODEL),
            nn.ReLU(),
        )
        self.cvae_mu = nn.Linear(D_MODEL, LATENT_DIM)
        self.cvae_logvar = nn.Linear(D_MODEL, LATENT_DIM)

        # z projection to d_model (becomes a token in the encoder)
        self.z_proj = nn.Linear(LATENT_DIM, D_MODEL)

        # Transformer
        enc_layer = nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=N_HEADS,
                                               dim_feedforward=D_MODEL * FFN_MULT, dropout=0.0, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=N_ENC_LAYERS)
        dec_layer = nn.TransformerDecoderLayer(d_model=D_MODEL, nhead=N_HEADS,
                                               dim_feedforward=D_MODEL * FFN_MULT, dropout=0.0, batch_first=False)
        self.transformer_decoder = nn.TransformerDecoder(dec_layer, num_layers=N_DEC_LAYERS)

        # Action queries and head
        self.action_queries = nn.Parameter(torch.randn(CHUNK_SIZE, D_MODEL) * 0.02)
        self.action_head = nn.Sequential(nn.Linear(D_MODEL, D_MODEL), nn.ReLU(), nn.Linear(D_MODEL, ACTION_DIM))

    def encode_z(self, action_chunk):
        """CVAE encoder: action chunk [B, K, 4] -> mu, logvar [B, LATENT_DIM]"""
        flat = action_chunk.flatten(1)  # [B, K*4]
        h = self.cvae_encoder(flat)
        return self.cvae_mu(h), self.cvae_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, image, language_id, action_chunk=None):
        """
        Training: action_chunk provided -> encode z from actions
        Inference: action_chunk=None -> z=0
        Returns: (logits, mu, logvar) during training, logits during inference
        """
        B = image.size(0)

        # Vision CNN
        x = self.relu(self.conv1(image))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))  # [B, 64, 8, 8]

        # Spatial tokens
        x = x.flatten(2).permute(0, 2, 1)  # [B, 64, 64]
        x = self.spatial_proj(x) + self.spatial_pos.unsqueeze(0)  # [B, 64, D]

        # Language token
        lang_token = self.lang_embed(language_id)  # [B, D]

        # CVAE: encode z from action chunk during training
        mu, logvar = None, None
        if action_chunk is not None:
            mu, logvar = self.encode_z(action_chunk)
            z = self.reparameterize(mu, logvar)
        else:
            z = torch.zeros(B, LATENT_DIM, device=image.device)

        # Project z to a token
        z_token = self.z_proj(z).unsqueeze(1)  # [B, 1, D]

        # Encoder: [spatial_tokens(64), lang_token(1), z_token(1)] = 66 tokens
        enc_input = torch.cat([x, lang_token.unsqueeze(1), z_token], dim=1)  # [B, 66, D]
        memory = self.transformer_encoder(enc_input.permute(1, 0, 2))  # [66, B, D]

        # Decoder: action queries + lang injection
        queries = self.action_queries.unsqueeze(1).expand(-1, B, -1) + lang_token.unsqueeze(0)
        decoded = self.transformer_decoder(queries, memory).permute(1, 0, 2)  # [B, K, D]

        logits = self.action_head(decoded)  # [B, K, 4]

        if action_chunk is not None:
            return logits, mu, logvar
        return logits

model = LangConditionedACT_CVAE().to(device)
num_params = sum(p.numel() for p in model.parameters())
print(f"Model: LangConditionedACT_CVAE ({num_params:,} params)")

def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)

bce = nn.BCEWithLogitsLoss()
best_val = float('inf')
best_acc = 0.0
t0 = time.time()

# Phase 1: SGD
print(f"\n=== Phase 1: SGD, LR=0.01, 100 epochs, KL_WEIGHT={KL_WEIGHT} ===")
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

for epoch in range(1, 101):
    if epoch <= 10:
        lr = 0.01 * epoch / 10
    elif epoch > 60:
        lr = 0.01 * max(1 - (epoch - 60) / 40, 0.01)
    else:
        lr = 0.01
    for pg in optimizer.param_groups: pg['lr'] = lr

    model.train()
    tloss, tkl, nt = 0, 0, 0
    for img, lng, act in train_dl:
        img, lng, act = img.to(device), lng.to(device), act.to(device)
        logits, mu, logvar = model(img, lng, act)
        recon_loss = bce(logits, act)
        kl_loss = kl_divergence(mu, logvar)
        loss = recon_loss + KL_WEIGHT * kl_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tloss += recon_loss.item() * img.size(0)
        tkl += kl_loss.item() * img.size(0)
        nt += img.size(0)

    model.eval()
    vloss, cor, tot, nv = 0, 0, 0, 0
    with torch.no_grad():
        for img, lng, act in val_dl:
            img, lng, act = img.to(device), lng.to(device), act.to(device)
            # Evaluate with z=0 (inference mode)
            logits = model(img, lng)
            vloss += bce(logits, act).item() * img.size(0)
            cor += ((torch.sigmoid(logits[:, 0, :]) > 0.5).float() == act[:, 0, :]).sum().item()
            tot += act[:, 0, :].numel(); nv += img.size(0)
    vloss /= nv
    acc = cor / tot
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}  bce={tloss/nt:.4f}  kl={tkl/nt:.4f}  val={vloss:.4f}  acc={acc:.3f}  [{time.time()-t0:.0f}s]")
    if vloss < best_val:
        best_val = vloss
        best_acc = acc
        torch.save(model.state_dict(), 'best_model.pt')

print(f"Phase 1 best: val={best_val:.4f} acc={best_acc:.3f}")

# Phase 2: Adam
print(f"\n=== Phase 2: Adam, LR=5e-4, 100 epochs ===")
optimizer2 = optim.Adam(model.parameters(), lr=5e-4)
scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=100)

for epoch in range(101, 201):
    model.train()
    tloss, tkl, nt = 0, 0, 0
    for img, lng, act in train_dl:
        img, lng, act = img.to(device), lng.to(device), act.to(device)
        logits, mu, logvar = model(img, lng, act)
        recon_loss = bce(logits, act)
        kl_loss = kl_divergence(mu, logvar)
        loss = recon_loss + KL_WEIGHT * kl_loss
        optimizer2.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer2.step()
        tloss += recon_loss.item() * img.size(0)
        tkl += kl_loss.item() * img.size(0)
        nt += img.size(0)
    scheduler2.step()

    model.eval()
    vloss, cor, tot, nv = 0, 0, 0, 0
    with torch.no_grad():
        for img, lng, act in val_dl:
            img, lng, act = img.to(device), lng.to(device), act.to(device)
            logits = model(img, lng)
            vloss += bce(logits, act).item() * img.size(0)
            cor += ((torch.sigmoid(logits[:, 0, :]) > 0.5).float() == act[:, 0, :]).sum().item()
            tot += act[:, 0, :].numel(); nv += img.size(0)
    vloss /= nv
    acc = cor / tot
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d}  bce={tloss/nt:.4f}  kl={tkl/nt:.4f}  val={vloss:.4f}  acc={acc:.3f}  [{time.time()-t0:.0f}s]")
    if vloss < best_val:
        best_val = vloss
        best_acc = acc
        torch.save(model.state_dict(), 'best_model.pt')

print(f"\nBest overall: val={best_val:.4f} acc={best_acc:.3f}")

# Eval
model.load_state_dict(torch.load('best_model.pt', weights_only=True))
model.eval()

print("\n--- Per-Intent Probabilities (z=0, t=0) ---")
for lid in range(num_intents):
    mask = langs == lid
    test_imgs = images[mask][:200].to(device)
    test_lngs = langs[mask][:200].to(device)
    with torch.no_grad():
        logits = model(test_imgs, test_lngs)
        probs = torch.sigmoid(logits[:, 0, :]).cpu()
    m = probs.mean(dim=0); s = probs.std(dim=0)
    print(f"  [{lid}] {intent_labels[lid]} ({mask.sum()} samples):")
    print(f"    fwd={m[0]:.3f}+/-{s[0]:.3f}  left={m[2]:.3f}+/-{s[2]:.3f}  right={m[3]:.3f}+/-{s[3]:.3f}")

# Cross-intent test
print("\n--- Cross-Intent Test (same 50 images, z=0) ---")
test_imgs = images[:50].to(device)
for lid in range(num_intents):
    test_lngs = torch.full((50,), lid, dtype=torch.long, device=device)
    with torch.no_grad():
        probs = torch.sigmoid(model(test_imgs, test_lngs)[:, 0, :]).cpu()
    m = probs.mean(dim=0)
    print(f"  Intent '{intent_labels[lid]}': fwd={m[0]:.3f}  left={m[2]:.3f}  right={m[3]:.3f}")

# z sampling test
print("\n--- z sampling test (50 images, intent 0, 5 z samples) ---")
test_imgs = images[:50].to(device)
test_lngs = torch.zeros(50, dtype=torch.long, device=device)
for z_idx in range(5):
    z_sample = torch.randn(50, LATENT_DIM, device=device) * 0.5
    with torch.no_grad():
        # Manual forward with custom z
        x = model.relu(model.conv1(test_imgs))
        x = model.relu(model.conv2(x))
        x = model.relu(model.conv3(x))
        x = model.relu(model.conv4(x))
        x = x.flatten(2).permute(0, 2, 1)
        x = model.spatial_proj(x) + model.spatial_pos.unsqueeze(0)
        lang_token = model.lang_embed(test_lngs)
        z_token = model.z_proj(z_sample).unsqueeze(1)
        enc_input = torch.cat([x, lang_token.unsqueeze(1), z_token], dim=1)  # [B, 66, D]
        memory = model.transformer_encoder(enc_input.permute(1, 0, 2))
        queries = model.action_queries.unsqueeze(1).expand(-1, 50, -1) + lang_token.unsqueeze(0)
        decoded = model.transformer_decoder(queries, memory).permute(1, 0, 2)
        logits = model.action_head(decoded)
        probs = torch.sigmoid(logits[:, 0, :]).cpu()
    print(f"  z_sample {z_idx}: fwd={probs[:,0].mean():.3f}  left={probs[:,2].mean():.3f}  right={probs[:,3].mean():.3f}")

# Export
print("\n--- Exporting model.json ---")
model_cpu = model.cpu()
state = model_cpu.state_dict()
weights = {}
for key, tensor in state.items():
    weights[key] = {'shape': list(tensor.shape), 'data': tensor.flatten().tolist()}

out = 'model.json'
with open(out, 'w') as f:
    json.dump({
        'format': 'language_conditioned_act_cvae',
        'num_params': num_params,
        'num_intents': num_intents,
        'intent_labels': intent_labels[:num_intents],
        'img_size': IMG_SIZE,
        'n_conv_layers': 4,
        'chunk_size': CHUNK_SIZE,
        'action_dim': ACTION_DIM,
        'd_model': D_MODEL,
        'n_heads': N_HEADS,
        'n_enc_layers': N_ENC_LAYERS,
        'n_dec_layers': N_DEC_LAYERS,
        'n_spatial_tokens': N_SPATIAL,
        'latent_dim': LATENT_DIM,
        'use_vae': True,
        'weights': weights,
    }, f)
print(f"Exported {out} ({os.path.getsize(out)/1024:.0f} KB)")
print("Done!")
