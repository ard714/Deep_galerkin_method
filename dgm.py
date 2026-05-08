
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os, time

torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGDIR, exist_ok=True)


# ======================================================================
# SECTION 1: DGM NETWORK ARCHITECTURE (Eq. 4.2 in the paper)
# ======================================================================

class DGMLayer(nn.Module):
    """Single DGM layer with LSTM-style gating (Eq. 4.2).

    Update rule:
        Z^l = sigma(U_z x + W_z S^l + b_z)     [update gate]
        G^l = sigma(U_g x + W_g S^l + b_g)     [gate]
        R^l = sigma(U_r x + W_r S^l + b_r)     [reset gate]
        H^l = sigma(U_h x + W_h (S^l * R^l) + b_h) [candidate]
        S^{l+1} = (1 - G^l) * H^l + Z^l * S^l
    """
    def __init__(self, n_hidden, n_input):
        super().__init__()
        # Update gate Z
        self.Uz = nn.Linear(n_input, n_hidden)
        self.Wz = nn.Linear(n_hidden, n_hidden, bias=False)
        # Gate G
        self.Ug = nn.Linear(n_input, n_hidden)
        self.Wg = nn.Linear(n_hidden, n_hidden, bias=False)
        # Reset gate R
        self.Ur = nn.Linear(n_input, n_hidden)
        self.Wr = nn.Linear(n_hidden, n_hidden, bias=False)
        # Candidate H
        self.Uh = nn.Linear(n_input, n_hidden)
        self.Wh = nn.Linear(n_hidden, n_hidden, bias=False)

    def forward(self, x, S):
        Z = torch.sigmoid(self.Uz(x) + self.Wz(S))
        G = torch.sigmoid(self.Ug(x) + self.Wg(S))
        R = torch.sigmoid(self.Ur(x) + self.Wr(S))
        H = torch.tanh(self.Uh(x) + self.Wh(S * R))
        S_new = (1 - G) * H + Z * S
        return S_new


class DGMNet(nn.Module):
    """Deep Galerkin Method Network.

    Architecture (from Section 4.2):
        S^1 = sigma(W^1 x + b^1)
        S^{l+1} = DGMLayer(x, S^l)   for l = 1, ..., L
        f(t,x;theta) = W * S^{L+1} + b

    Args:
        n_input:  dimension of input (t,x) vector
        n_hidden: number of units M per sub-layer
        n_layers: number of DGM layers L
        n_output: output dimension (1 for scalar PDE solutions)
    """
    def __init__(self, n_input=2, n_hidden=50, n_layers=3, n_output=1):
        super().__init__()
        self.init_layer = nn.Linear(n_input, n_hidden)
        self.dgm_layers = nn.ModuleList(
            [DGMLayer(n_hidden, n_input) for _ in range(n_layers)]
        )
        self.output_layer = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        S = torch.tanh(self.init_layer(x))
        for layer in self.dgm_layers:
            S = layer(x, S)
        return self.output_layer(S)


class StandardPINN(nn.Module):
    """Standard MLP-based PINN for comparison."""
    def __init__(self, n_input=2, n_hidden=50, n_layers=4, n_output=1):
        super().__init__()
        layers = [nn.Linear(n_input, n_hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_hidden, n_hidden), nn.Tanh()]
        layers.append(nn.Linear(n_hidden, n_output))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ======================================================================
# SECTION 2: PROBLEM 1 -- BLACK-SCHOLES PDE (European option baseline)
# ======================================================================
# This validates the DGM on the Black-Scholes PDE before tackling
# the free boundary (American option) case.
#
# PDE: du/dt + 0.5*sigma^2*x^2 * d2u/dx2 + (r-c)*x*du/dx - r*u = 0
# Terminal: u(T,x) = max(x - K, 0)
# Boundary: u(t,0) = 0
#
# Exact (Black-Scholes formula) available for validation.

def black_scholes_call(S, K, T, t, r, sigma, c=0.0):
    """Exact Black-Scholes European call price."""
    from scipy.stats import norm
    tau = T - t
    if tau < 1e-12:
        return np.maximum(S - K, 0.0)
    d1 = (np.log(S / K) + (r - c + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return S * np.exp(-c * tau) * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)


def solve_black_scholes_dgm():
    """Solve Black-Scholes PDE using DGM (Section 4 of the paper)."""
    print("\n" + "=" * 70)
    print("  PROBLEM 1: Black-Scholes PDE via DGM")
    print("  du/dt + 0.5*sig^2*x^2*d2u/dx2 + (r-c)*x*du/dx - r*u = 0")
    print("=" * 70)

    # Parameters (similar to Table 1 in the paper)
    sigma = 0.25
    r = 0.05
    c = 0.02
    K = 1.0
    T = 1.0
    x_max = 3.0  # truncated domain

    model = DGMNet(n_input=2, n_hidden=50, n_layers=3).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

    n_interior = 500
    n_boundary = 100
    n_terminal = 200

    losses_hist = []
    t_start = time.time()

    for epoch in range(5000):
        optimizer.zero_grad()

        # --- Interior points: random (t, x) in [0,T) x (0, x_max) ---
        t_int = T * torch.rand(n_interior, 1, device=DEVICE, requires_grad=True)
        x_int = x_max * torch.rand(n_interior, 1, device=DEVICE, requires_grad=True)
        tx = torch.cat([t_int, x_int], dim=1)

        u = model(tx)

        # Compute derivatives via autograd
        du_dt = torch.autograd.grad(u, t_int, torch.ones_like(u),
                                    create_graph=True)[0]
        du_dx = torch.autograd.grad(u, x_int, torch.ones_like(u),
                                    create_graph=True)[0]
        d2u_dx2 = torch.autograd.grad(du_dx, x_int, torch.ones_like(du_dx),
                                       create_graph=True)[0]

        # PDE residual (backward in time formulation)
        pde_res = (du_dt
                   + 0.5 * sigma**2 * x_int**2 * d2u_dx2
                   + (r - c) * x_int * du_dx
                   - r * u)
        loss_pde = torch.mean(pde_res**2)

        # --- Boundary at x=0: u(t,0) = 0 ---
        t_bc = T * torch.rand(n_boundary, 1, device=DEVICE)
        x_bc = torch.zeros(n_boundary, 1, device=DEVICE)
        u_bc = model(torch.cat([t_bc, x_bc], dim=1))
        loss_bc = torch.mean(u_bc**2)

        # --- Terminal condition: u(T,x) = max(x-K, 0) ---
        x_tc = x_max * torch.rand(n_terminal, 1, device=DEVICE)
        t_tc = T * torch.ones(n_terminal, 1, device=DEVICE)
        u_tc = model(torch.cat([t_tc, x_tc], dim=1))
        payoff = torch.relu(x_tc - K)
        loss_tc = torch.mean((u_tc - payoff)**2)

        loss = loss_pde + 10 * loss_bc + 10 * loss_tc
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses_hist.append(loss.item())

        if (epoch + 1) % 1000 == 0:
            elapsed = time.time() - t_start
            print(f"  Epoch {epoch+1:5d} | Loss: {loss.item():.4e} "
                  f"| PDE: {loss_pde.item():.2e} "
                  f"| BC: {loss_bc.item():.2e} "
                  f"| TC: {loss_tc.item():.2e} "
                  f"| Time: {elapsed:.1f}s")

    # --- Evaluation ---
    nx, nt = 100, 50
    x_eval = np.linspace(0.01, x_max, nx)
    t_eval_pts = [0.0, 0.25, 0.5, 0.75]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, t_val in enumerate(t_eval_pts):
        ax = axes[idx // 2][idx % 2]
        t_arr = t_val * np.ones(nx)
        tx_test = torch.FloatTensor(np.column_stack([t_arr, x_eval])).to(DEVICE)
        with torch.no_grad():
            u_pred = model(tx_test).cpu().numpy().flatten()
        u_exact = black_scholes_call(x_eval, K, T, t_val, r, sigma, c)

        ax.plot(x_eval, u_exact, "b-", lw=2, label="Exact (Black-Scholes)")
        ax.plot(x_eval, u_pred, "r--", lw=2, label="DGM")
        ax.set_xlabel("Stock price x")
        ax.set_ylabel("Option price u(t,x)")
        ax.set_title(f"t = {t_val:.2f}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("DGM Solution vs Exact Black-Scholes", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "fig1_black_scholes.png"), dpi=150)
    plt.close()

    # Error analysis
    t0_arr = np.zeros(nx)
    tx0 = torch.FloatTensor(np.column_stack([t0_arr, x_eval])).to(DEVICE)
    with torch.no_grad():
        u_pred_0 = model(tx0).cpu().numpy().flatten()
    u_exact_0 = black_scholes_call(x_eval, K, T, 0, r, sigma, c)
    mask = u_exact_0 > 0.01
    rel_err = np.abs(u_pred_0[mask] - u_exact_0[mask]) / u_exact_0[mask]
    print(f"\n  Mean relative error at t=0: {np.mean(rel_err)*100:.2f}%")
    print(f"  Max  relative error at t=0: {np.max(rel_err)*100:.2f}%")

    # Loss curve
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.semilogy(losses_hist, "b-", alpha=0.7)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Total Loss")
    ax2.set_title("Training Loss -- Black-Scholes PDE")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "fig2_bs_loss.png"), dpi=150)
    plt.close()

    return model


# ======================================================================
# SECTION 3: PROBLEM 2 -- HEAT EQUATION (as HJB baseline)
# ======================================================================
# Solves: du/dt = nu * d2u/dx2   on [0,1] x [0, 0.5]
# IC: u(x,0) = sin(pi*x)
# BC: u(0,t) = u(1,t) = 0
# Exact: u(x,t) = sin(pi*x) * exp(-nu*pi^2*t)
#
# This demonstrates the DGM architecture on a standard parabolic PDE
# before the high-dimensional HJB (Section 5 of the paper).

def solve_heat_equation_dgm():
    """Solve 1D heat equation with DGM."""
    print("\n" + "=" * 70)
    print("  PROBLEM 2: Heat Equation via DGM")
    print("  du/dt = nu * d2u/dx2, u(x,0)=sin(pi*x), u(0,t)=u(1,t)=0")
    print("=" * 70)

    nu = 0.1

    model = DGMNet(n_input=2, n_hidden=50, n_layers=3).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

    losses = {"total": [], "pde": [], "bc": [], "ic": []}
    t_start = time.time()

    for epoch in range(5000):
        optimizer.zero_grad()

        # Interior
        x_i = torch.rand(800, 1, device=DEVICE, requires_grad=True)
        t_i = 0.5 * torch.rand(800, 1, device=DEVICE, requires_grad=True)
        xt = torch.cat([x_i, t_i], dim=1)

        u = model(xt)
        du_dt = torch.autograd.grad(u, t_i, torch.ones_like(u),
                                    create_graph=True)[0]
        du_dx = torch.autograd.grad(u, x_i, torch.ones_like(u),
                                    create_graph=True)[0]
        d2u_dx2 = torch.autograd.grad(du_dx, x_i, torch.ones_like(du_dx),
                                       create_graph=True)[0]

        loss_pde = torch.mean((du_dt - nu * d2u_dx2)**2)

        # Boundary
        t_b = 0.5 * torch.rand(100, 1, device=DEVICE)
        u_b0 = model(torch.cat([torch.zeros(100, 1, device=DEVICE), t_b], dim=1))
        u_b1 = model(torch.cat([torch.ones(100, 1, device=DEVICE), t_b], dim=1))
        loss_bc = torch.mean(u_b0**2) + torch.mean(u_b1**2)

        # Initial condition
        x_ic = torch.rand(200, 1, device=DEVICE)
        t_ic = torch.zeros(200, 1, device=DEVICE)
        u_ic = model(torch.cat([x_ic, t_ic], dim=1))
        loss_ic = torch.mean((u_ic - torch.sin(np.pi * x_ic))**2)

        loss = loss_pde + 10 * loss_bc + 10 * loss_ic
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses["total"].append(loss.item())
        losses["pde"].append(loss_pde.item())
        losses["bc"].append(loss_bc.item())
        losses["ic"].append(loss_ic.item())

        if (epoch + 1) % 1000 == 0:
            print(f"  Epoch {epoch+1:5d} | Loss: {loss.item():.4e} "
                  f"| PDE: {loss_pde.item():.2e}")

    # Evaluation
    nx, nt = 80, 80
    xg = np.linspace(0, 1, nx)
    tg = np.linspace(0, 0.5, nt)
    X, T = np.meshgrid(xg, tg)

    xt_test = torch.FloatTensor(
        np.column_stack([X.ravel(), T.ravel()])
    ).to(DEVICE)
    with torch.no_grad():
        u_pred = model(xt_test).cpu().numpy().reshape(nt, nx)
    u_exact = np.sin(np.pi * X) * np.exp(-nu * np.pi**2 * T)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))

    im0 = axes[0].contourf(X, T, u_exact, 30, cmap="RdYlBu_r")
    axes[0].set_title("Exact Solution")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].contourf(X, T, u_pred, 30, cmap="RdYlBu_r")
    axes[1].set_title("DGM Solution")
    plt.colorbar(im1, ax=axes[1])

    err = np.abs(u_exact - u_pred)
    im2 = axes[2].contourf(X, T, err, 30, cmap="hot_r")
    axes[2].set_title(f"|Error| (max={np.max(err):.2e})")
    plt.colorbar(im2, ax=axes[2])

    for key in ["total", "pde", "ic"]:
        axes[3].semilogy(losses[key], alpha=0.7, label=key)
    axes[3].set_title("Training Losses")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    for ax in axes[:3]:
        ax.set_xlabel("x")
        ax.set_ylabel("t")

    plt.suptitle("DGM for 1D Heat Equation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "fig3_heat_equation.png"), dpi=150)
    plt.close()

    print(f"  Max absolute error: {np.max(err):.4e}")
    return model


# ======================================================================
# SECTION 4: PROBLEM 3 -- BURGERS' EQUATION (Section 6 of the paper)
# ======================================================================
# du/dt + alpha*u*du/dx = nu * d2u/dx2
# on [0,1] x [0,1] with Dirichlet BCs
#
# The paper trains a SINGLE network to solve over a range of
# (nu, alpha, a, b) parameters. We demonstrate the fixed-parameter
# case and the parameterized case.

def solve_burgers_dgm():
    """Solve Burgers' equation using DGM (Section 6 of the paper)."""
    print("\n" + "=" * 70)
    print("  PROBLEM 3: Burgers' Equation via DGM")
    print("  du/dt + alpha*u*du/dx = nu*d2u/dx2")
    print("=" * 70)

    # Fixed parameters first
    nu_val = 0.01 / np.pi
    alpha_val = 1.0

    model = DGMNet(n_input=2, n_hidden=50, n_layers=3).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

    losses = []
    t_start = time.time()

    for epoch in range(8000):
        optimizer.zero_grad()

        # Interior: (x,t) in [-1,1] x [0,1]
        x_i = 2 * torch.rand(1000, 1, device=DEVICE, requires_grad=True) - 1
        t_i = torch.rand(1000, 1, device=DEVICE, requires_grad=True)
        xt = torch.cat([x_i, t_i], dim=1)

        u = model(xt)
        du_dt = torch.autograd.grad(u, t_i, torch.ones_like(u),
                                    create_graph=True)[0]
        du_dx = torch.autograd.grad(u, x_i, torch.ones_like(u),
                                    create_graph=True)[0]
        d2u_dx2 = torch.autograd.grad(du_dx, x_i, torch.ones_like(du_dx),
                                       create_graph=True)[0]

        pde_res = du_dt + alpha_val * u * du_dx - nu_val * d2u_dx2
        loss_pde = torch.mean(pde_res**2)

        # Boundary: u(-1,t) = u(1,t) = 0
        t_bc = torch.rand(200, 1, device=DEVICE)
        u_left = model(torch.cat([-torch.ones(200, 1, device=DEVICE), t_bc], dim=1))
        u_right = model(torch.cat([torch.ones(200, 1, device=DEVICE), t_bc], dim=1))
        loss_bc = torch.mean(u_left**2) + torch.mean(u_right**2)

        # Initial condition: u(x,0) = -sin(pi*x)
        x_ic = 2 * torch.rand(300, 1, device=DEVICE) - 1
        t_ic = torch.zeros(300, 1, device=DEVICE)
        u_ic = model(torch.cat([x_ic, t_ic], dim=1))
        loss_ic = torch.mean((u_ic - (-torch.sin(np.pi * x_ic)))**2)

        loss = loss_pde + 20 * loss_bc + 20 * loss_ic
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

        if (epoch + 1) % 2000 == 0:
            elapsed = time.time() - t_start
            print(f"  Epoch {epoch+1:5d} | Loss: {loss.item():.4e} | Time: {elapsed:.1f}s")

    # Evaluation
    nx, nt = 100, 100
    xg = np.linspace(-1, 1, nx)
    tg = np.linspace(0, 1, nt)
    X, T = np.meshgrid(xg, tg)

    xt_test = torch.FloatTensor(
        np.column_stack([X.ravel(), T.ravel()])
    ).to(DEVICE)
    with torch.no_grad():
        u_pred = model(xt_test).cpu().numpy().reshape(nt, nx)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    im = axes[0].contourf(X, T, u_pred, 40, cmap="RdYlBu_r")
    axes[0].set_title("DGM Solution: Burgers' Equation")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("t")
    plt.colorbar(im, ax=axes[0])

    # Time slices
    colors = plt.cm.viridis(np.linspace(0, 1, 5))
    for i, t_val in enumerate([0, 0.25, 0.5, 0.75, 1.0]):
        idx = int(t_val * (nt - 1))
        axes[1].plot(xg, u_pred[idx], color=colors[i], lw=2,
                     label=f"t={t_val:.2f}")
    axes[1].set_title("Solution at Different Times")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("u(x,t)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].semilogy(losses, "b-", alpha=0.7)
    axes[2].set_title("Training Loss")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Loss")
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("DGM for Burgers' Equation (Section 6 of paper)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "fig4_burgers.png"), dpi=150)
    plt.close()

    return model


# ======================================================================
# SECTION 5: PROBLEM 4 -- 2D POISSON EQUATION (Elliptic PDE)
# ======================================================================
# Laplacian(u) = f(x,y) on [0,1]^2
# u = 0 on boundary
# f(x,y) = -2*pi^2 * sin(pi*x)*sin(pi*y)
# Exact: u(x,y) = sin(pi*x)*sin(pi*y)

def solve_poisson_dgm():
    """Solve 2D Poisson equation comparing DGM vs standard PINN."""
    print("\n" + "=" * 70)
    print("  PROBLEM 4: 2D Poisson Equation -- DGM vs Standard PINN")
    print("  Laplacian(u) = -2*pi^2*sin(pi*x)*sin(pi*y)")
    print("=" * 70)

    results = {}
    for name, ModelClass in [("DGM", DGMNet), ("PINN", StandardPINN)]:
        print(f"\n  Training {name}...")
        model = ModelClass(n_input=2, n_hidden=40, n_layers=3).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1500, gamma=0.5)

        losses = []
        for epoch in range(3000):
            optimizer.zero_grad()

            x = torch.rand(600, 1, device=DEVICE, requires_grad=True)
            y = torch.rand(600, 1, device=DEVICE, requires_grad=True)
            xy = torch.cat([x, y], dim=1)

            u = model(xy)
            du_dx = torch.autograd.grad(u, x, torch.ones_like(u),
                                        create_graph=True)[0]
            d2u_dx2 = torch.autograd.grad(du_dx, x, torch.ones_like(du_dx),
                                           create_graph=True)[0]
            du_dy = torch.autograd.grad(u, y, torch.ones_like(u),
                                        create_graph=True)[0]
            d2u_dy2 = torch.autograd.grad(du_dy, y, torch.ones_like(du_dy),
                                           create_graph=True)[0]

            f_rhs = -2 * np.pi**2 * torch.sin(np.pi * x) * torch.sin(np.pi * y)
            loss_pde = torch.mean((d2u_dx2 + d2u_dy2 - f_rhs)**2)

            nb = 60
            yb = torch.rand(nb, 1, device=DEVICE)
            xb = torch.rand(nb, 1, device=DEVICE)
            loss_bc = (
                torch.mean(model(torch.cat([torch.zeros(nb, 1, device=DEVICE), yb], 1))**2)
                + torch.mean(model(torch.cat([torch.ones(nb, 1, device=DEVICE), yb], 1))**2)
                + torch.mean(model(torch.cat([xb, torch.zeros(nb, 1, device=DEVICE)], 1))**2)
                + torch.mean(model(torch.cat([xb, torch.ones(nb, 1, device=DEVICE)], 1))**2)
            )

            loss = loss_pde + 20 * loss_bc
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())

            if (epoch + 1) % 1000 == 0:
                print(f"    Epoch {epoch+1:4d} | Loss: {loss.item():.4e}")

        # Evaluate
        n_eval = 60
        xg = np.linspace(0, 1, n_eval)
        yg = np.linspace(0, 1, n_eval)
        X, Y = np.meshgrid(xg, yg)
        xy_test = torch.FloatTensor(
            np.column_stack([X.ravel(), Y.ravel()])
        ).to(DEVICE)
        with torch.no_grad():
            u_pred = model(xy_test).cpu().numpy().reshape(n_eval, n_eval)
        u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
        err = np.abs(u_exact - u_pred)
        results[name] = {"pred": u_pred, "err": err, "losses": losses,
                         "max_err": np.max(err)}
        print(f"    {name} max error: {np.max(err):.4e}")

    # Plotting
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    n_eval = 60
    xg = np.linspace(0, 1, n_eval)
    yg = np.linspace(0, 1, n_eval)
    X, Y = np.meshgrid(xg, yg)
    u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)

    im0 = axes[0][0].contourf(X, Y, u_exact, 25, cmap="viridis")
    axes[0][0].set_title("Exact Solution")
    plt.colorbar(im0, ax=axes[0][0])

    for row, name in enumerate(["DGM", "PINN"]):
        im1 = axes[row][1].contourf(X, Y, results[name]["pred"], 25, cmap="viridis")
        axes[row][1].set_title(f"{name} Solution")
        plt.colorbar(im1, ax=axes[row][1])

        im2 = axes[row][2].contourf(X, Y, results[name]["err"], 25, cmap="hot_r")
        axes[row][2].set_title(
            f"{name} |Error| (max={results[name]['max_err']:.2e})")
        plt.colorbar(im2, ax=axes[row][2])

    for ax_row in axes:
        for ax in ax_row:
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_aspect("equal")

    axes[1][0].semilogy(results["DGM"]["losses"], "b-", alpha=0.7, label="DGM")
    axes[1][0].semilogy(results["PINN"]["losses"], "r-", alpha=0.7, label="PINN")
    axes[1][0].set_title("Loss Comparison")
    axes[1][0].legend()
    axes[1][0].grid(True, alpha=0.3)
    axes[1][0].set_aspect("auto")

    plt.suptitle("DGM vs PINN: 2D Poisson Equation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "fig5_poisson_comparison.png"), dpi=150)
    plt.close()

    return results

# ======================================================================
# MAIN
# ======================================================================

if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("#  DEEP GALERKIN METHOD (DGM) -- COMPLETE IMPLEMENTATION")
    print("#  Based on: Sirignano & Spiliopoulos (2018)")
    print("#" * 70)

    solve_black_scholes_dgm()
    solve_heat_equation_dgm()
    solve_burgers_dgm()
    solve_poisson_dgm()
    generate_summary()

    print("\n" + "=" * 70)
    print("  ALL DONE. Figures saved to:", FIGDIR)
    print("=" * 70)