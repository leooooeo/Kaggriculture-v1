"""Factored policy network (PyTorch).

Shared CNN+global encoder -> spatial feature map S and a global context ctx.
Unit heads consume S gathered at each unit's (x,y) plus ctx.
Market heads consume ctx only.

Import is torch-guarded so the rest of the package works without torch.
"""
from . import spec
from . import features as F
from . import action_codec as A

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fn
    _HAS_TORCH = True
except Exception:  # noqa: BLE001
    _HAS_TORCH = False


N_BUCKETS = len(spec.QTY_BUCKETS)


if _HAS_TORCH:

    class ConvBlock(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.c = nn.Conv2d(cin, cout, 3, padding=1)
            self.n = nn.GroupNorm(8, cout)

        def forward(self, x):
            return Fn.relu(self.n(self.c(x)))

    class PolicyNet(nn.Module):
        def __init__(self, c_board=None, g_global=None, width=96, ctx_dim=192):
            super().__init__()
            c_board = c_board or F.C_BOARD
            g_global = g_global or F.G_GLOBAL
            self.c_board = c_board
            self.g_global = g_global

            self.glob_mlp = nn.Sequential(
                nn.Linear(g_global, ctx_dim), nn.ReLU(),
                nn.Linear(ctx_dim, ctx_dim), nn.ReLU(),
            )
            # board stem then fuse global (broadcast) then more conv
            self.stem = nn.Sequential(ConvBlock(c_board, width), ConvBlock(width, width))
            self.fuse = nn.Sequential(
                ConvBlock(width + ctx_dim, width), ConvBlock(width, width))
            self.spatial_dim = width
            # ctx = pooled spatial (mean+max) + global embedding
            self.ctx_head = nn.Sequential(
                nn.Linear(2 * width + ctx_dim, ctx_dim), nn.ReLU())

            # ---- unit heads ----
            uin = width + ctx_dim
            self.unit_trunk = nn.Sequential(nn.Linear(uin, ctx_dim), nn.ReLU())
            self.h_type = nn.Linear(ctx_dim, spec.N_UNIT_ACTIONS)
            self.h_crop = nn.Linear(ctx_dim, len(spec.CROPS))
            self.h_item = nn.Linear(ctx_dim, len(spec.UNIT_ITEMS))
            self.h_count = nn.Linear(ctx_dim, N_BUCKETS)

            # ---- market heads ----
            self.m_sell_p = nn.Linear(ctx_dim, len(spec.SELL_SLOTS))
            self.m_sell_q = nn.Linear(ctx_dim, len(spec.SELL_SLOTS) * N_BUCKETS)
            self.m_bprod_p = nn.Linear(ctx_dim, len(spec.BUY_PRODUCT_SLOTS))
            self.m_bprod_q = nn.Linear(ctx_dim, len(spec.BUY_PRODUCT_SLOTS) * N_BUCKETS)
            self.m_bseed_p = nn.Linear(ctx_dim, len(spec.BUY_SEED_SLOTS))
            self.m_bseed_q = nn.Linear(ctx_dim, len(spec.BUY_SEED_SLOTS) * N_BUCKETS)
            self.m_banim_p = nn.Linear(ctx_dim, len(spec.BUY_ANIMAL_SLOTS))
            self.m_banim_q = nn.Linear(ctx_dim, len(spec.BUY_ANIMAL_SLOTS) * N_BUCKETS)
            self.m_hire = nn.Linear(ctx_dim, spec.MAX_HIRE + 1)
            self.m_land = nn.Linear(ctx_dim, 1)

        # -- encoder --
        def encode(self, board, glob):
            B, _, H, W = board.shape
            gemb = self.glob_mlp(glob)                       # (B, ctx)
            x = self.stem(board)                             # (B, width, H, W)
            gmap = gemb[:, :, None, None].expand(-1, -1, H, W)
            x = self.fuse(torch.cat([x, gmap], 1))           # (B, width, H, W)
            pooled = torch.cat([x.mean((2, 3)), x.amax((2, 3)), gemb], 1)
            ctx = self.ctx_head(pooled)                      # (B, ctx)
            return x, ctx

        # -- units: gather S at positions, run heads. upos:(B,U,3) umask:(B,U) --
        def unit_logits(self, S, ctx, upos, umask):
            B, C, H, W = S.shape
            U = upos.shape[1]
            xs = upos[..., 0].long().clamp(0, W - 1)
            ys = upos[..., 1].long().clamp(0, H - 1)
            flat = S.view(B, C, H * W)
            idx = (ys * W + xs)                              # (B,U)
            gathered = torch.gather(flat, 2, idx[:, None, :].expand(-1, C, -1))
            gathered = gathered.permute(0, 2, 1)             # (B,U,C)
            ctxU = ctx[:, None, :].expand(-1, U, -1)
            feat = self.unit_trunk(torch.cat([gathered, ctxU], -1))  # (B,U,ctx)
            return {
                "type": self.h_type(feat),
                "crop": self.h_crop(feat),
                "item": self.h_item(feat),
                "count": self.h_count(feat),
            }

        def market_logits(self, ctx):
            def q(t, n):
                return t.view(t.shape[0], n, N_BUCKETS)
            return {
                "sell_present": self.m_sell_p(ctx),
                "sell_qty": q(self.m_sell_q(ctx), len(spec.SELL_SLOTS)),
                "bprod_present": self.m_bprod_p(ctx),
                "bprod_qty": q(self.m_bprod_q(ctx), len(spec.BUY_PRODUCT_SLOTS)),
                "bseed_present": self.m_bseed_p(ctx),
                "bseed_qty": q(self.m_bseed_q(ctx), len(spec.BUY_SEED_SLOTS)),
                "banim_present": self.m_banim_p(ctx),
                "banim_qty": q(self.m_banim_q(ctx), len(spec.BUY_ANIMAL_SLOTS)),
                "hire": self.m_hire(ctx),
                "buy_land": self.m_land(ctx).squeeze(-1),
            }

        def forward(self, board, glob, upos, umask):
            S, ctx = self.encode(board, glob)
            return self.unit_logits(S, ctx, upos, umask), self.market_logits(ctx)
