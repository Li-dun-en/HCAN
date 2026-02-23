"""诊断当前 HCL 的塌陷风险和 StopGradient 状态"""
import torch
import torch.nn.functional as F
from models.dft_layer import DualGeoHCAN

model = DualGeoHCAN(512, 64, n_fine=4, n_coarse=2, c_out=7)

print("=" * 60)
print("HCL Risk Diagnosis")
print("=" * 60)

# 1. 当前 tau 锐度分析 (sim * 5.0 = tau=0.2)
with torch.no_grad():
    fine_n = F.normalize(model.geo_fine.centroids, dim=-1)
    coarse_n = F.normalize(model.geo_coarse.centroids, dim=-1)
    sim = torch.matmul(fine_n, coarse_n.T)  # [4, 2]
    
    print("\n[1] Cosine similarity matrix (Fine x Coarse):")
    print(f"    {sim.numpy()}")
    
    print("\n[2] Mapping M at different tau values:")
    for tau in [0.2, 0.1, 0.05]:
        M = F.softmax(sim / tau, dim=-1)
        max_vals = M.max(dim=-1)[0]
        entropy = -(M * torch.log(M + 1e-8)).sum(dim=-1)
        print(f"  tau={tau}: M=")
        for i in range(4):
            arrow = "C0" if M[i, 0] > M[i, 1] else "C1"
            print(f"    Fine[{i}]->{arrow}: [{M[i,0]:.4f}, {M[i,1]:.4f}]  "
                  f"max={max_vals[i]:.4f} entropy={entropy[i]:.4f}")
        print(f"    Uniform threshold: max<0.6 count = "
              f"{(max_vals < 0.6).sum().item()}/4")

# 2. StopGradient 验证
print("\n[3] StopGradient on M check:")
model.train()
x = torch.randn(2, 24, 512)
out, u, T, a, ortho, geo, lat, hcl = model(x)

# Check: centroids should NOT have gradient from HCL
loss = hcl * 10.0  # amplify HCL
loss.backward()

fine_centroid_grad = model.geo_fine.centroids.grad
coarse_centroid_grad = model.geo_coarse.centroids.grad
print(f"  Fine centroids grad from HCL-only:  "
      f"{'None (detached OK)' if fine_centroid_grad is None else f'norm={fine_centroid_grad.norm().item():.6f}'}")
print(f"  Coarse centroids grad from HCL-only: "
      f"{'None (detached OK)' if coarse_centroid_grad is None else f'norm={coarse_centroid_grad.norm().item():.6f}'}")

# 3. Collapse simulation: what if all fine centroids are identical?
print("\n[4] Collapse simulation:")
model2 = DualGeoHCAN(512, 64, n_fine=4, n_coarse=2, c_out=7)
with torch.no_grad():
    # Force all fine centroids to be the same
    model2.geo_fine.centroids.data = model2.geo_fine.centroids[0:1].expand(4, -1).clone()
    fine_n = F.normalize(model2.geo_fine.centroids, dim=-1)
    coarse_n = F.normalize(model2.geo_coarse.centroids, dim=-1)
    sim = torch.matmul(fine_n, coarse_n.T)
    M_collapsed = F.softmax(sim * 5.0, dim=-1)  # current tau
    print(f"  All-same fine centroids, tau=0.2:")
    print(f"    M = {M_collapsed.numpy()}")
    print(f"    All rows identical: {torch.allclose(M_collapsed[0], M_collapsed[1])}")
    M_sharp = F.softmax(sim / 0.05, dim=-1)
    print(f"  All-same fine centroids, tau=0.05:")
    print(f"    M = {M_sharp.numpy()}")
    print(f"    Still can't prevent collapse of identical centroids (expected)")
    print(f"    --> ortho_loss is the real defense against centroid collapse")

print("\n" + "=" * 60)
print("Diagnosis Complete")
print("=" * 60)
