
import argparse
import os
import time
import zlib

import numpy as np
from scipy import stats

LN2 = np.log(2.0)

# ----------------------------------------------------------------------------
# The metric
# ----------------------------------------------------------------------------

def eli(k, t, M):
    k = np.asarray(k, dtype=float)
    return np.log10(M * LN2 / ((k + 1.0) * np.asarray(t, dtype=float)))


def eli_from_values(best, t, ref_sorted):
    k = np.searchsorted(ref_sorted, best, side="left")  # strictly better
    return eli(k, t, ref_sorted.size), k


def eli_pvalue(e):
    return 1.0 - np.power(2.0, -np.power(10.0, -np.asarray(e, dtype=float)))


def fisher_combine(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-300, 1.0)
    x2 = -2.0 * np.sum(np.log(p))
    return stats.chi2.sf(x2, 2 * p.size)


def eli_null_quantile(q):
    # ELI <= e  <=>  E >= ln2*10^-e  -> P = exp(-ln2 10^-e) = 2^(-10^-e)
    return -np.log10(-np.log2(q))


class TailELI:


    def __init__(self, ref_sorted, L=0.0, k0=1000, tail=True):
        self.ref, self.M, self.L, self.k0, self.tail = ref_sorted, ref_sorted.size, L, k0, tail
        self.u = ref_sorted[k0]
        low = np.maximum(ref_sorted[:k0] - L, 1e-300)
        self.alpha = k0 / np.sum(np.log((self.u - L) / low))

    def F(self, y):
        y = np.asarray(y, dtype=float)
        k = np.searchsorted(self.ref, y, side="left")
        z = np.maximum(y - self.L, 1e-300) / (self.u - self.L)
        tail = (self.k0 / self.M) * np.power(z, self.alpha)
        if not self.tail:  # plain empirical ELI (lower bound when k = 0)
            return (k + 1.0) / self.M
        return np.where(k >= self.k0, (k + 1.0) / self.M, tail)

    def eli(self, y, t):
        with np.errstate(divide="ignore"):  # F = 0 (underflow) -> ELI = +inf
            return np.log10(LN2 / (self.F(y) * np.asarray(t, dtype=float)))


# ----------------------------------------------------------------------------
# Benchmark functions (vectorised over rows)
# ----------------------------------------------------------------------------

def sphere(z):
    return np.sum(z * z, axis=1)


def rosenbrock(z):
    z = z + 1.0  # optimum moved to z = 0
    return np.sum(100.0 * (z[:, 1:] - z[:, :-1] ** 2) ** 2 + (z[:, :-1] - 1.0) ** 2, axis=1)


def rastrigin(z):
    return 10.0 * z.shape[1] + np.sum(z * z - 10.0 * np.cos(2 * np.pi * z), axis=1)


def ackley(z):
    d = z.shape[1]
    return (-20.0 * np.exp(-0.2 * np.sqrt(np.sum(z * z, axis=1) / d))
            - np.exp(np.sum(np.cos(2 * np.pi * z), axis=1) / d) + 20.0 + np.e)


def griewank(z):
    i = np.sqrt(np.arange(1, z.shape[1] + 1))
    return 1.0 + np.sum(z * z, axis=1) / 4000.0 - np.prod(np.cos(z / i), axis=1)


def schwefel(z):
    return 418.9828872724338 * z.shape[1] - np.sum(z * np.sin(np.sqrt(np.abs(z))), axis=1)


FUNCS = {
    #  name        f           bound   shiftable
    "Sphere":     (sphere,     100.0,  True),
    "Rosenbrock": (rosenbrock,  30.0,  True),
    "Rastrigin":  (rastrigin,    5.12, True),
    "Ackley":     (ackley,      32.0,  True),
    "Griewank":   (griewank,   600.0,  True),
    "Schwefel":   (schwefel,   500.0,  False),
}


class Problem:
    def __init__(self, name, dim, shifted, seed=2026):
        f, b, shiftable = FUNCS[name]
        self.name, self.f, self.dim = name, f, dim
        self.lb, self.ub = -b, b
        rng = np.random.default_rng(seed + zlib.crc32(name.encode()) % 1000)
        if shifted and shiftable:
            self.o = rng.uniform(0.8 * self.lb, 0.8 * self.ub, dim)
        else:
            self.o = np.zeros(dim)
        self.shifted = shifted and shiftable

    def __call__(self, X):
        return self.f(X - self.o)


def reference_sample(prob, M, seed, chunk=1_000_000):
    rng = np.random.default_rng(seed)
    out = np.empty(M)
    for s in range(0, M, chunk):
        n = min(chunk, M - s)
        out[s:s + n] = prob(rng.uniform(prob.lb, prob.ub, (n, prob.dim)))
    out.sort()
    return out


# ----------------------------------------------------------------------------
# Optimisers. Each returns (evals, best_so_far) arrays, one entry per batch.
# ----------------------------------------------------------------------------

class Tracker:
    def __init__(self, prob, T):
        self.prob, self.T, self.n, self.best = prob, T, 0, np.inf
        self.ev, self.bs = [], []

    def __call__(self, X):
        room = self.T - self.n
        if X.shape[0] > room:
            X = X[:room]
        f = self.prob(X)
        self.n += f.size
        self.best = min(self.best, f.min())
        self.ev.append(self.n)
        self.bs.append(self.best)
        return f

    @property
    def done(self):
        return self.n >= self.T

    def history(self):
        return np.array(self.ev), np.array(self.bs)


def random_search(prob, T, rng, batch=100):
    tr = Tracker(prob, T)
    while not tr.done:
        tr(rng.uniform(prob.lb, prob.ub, (batch, prob.dim)))
    return tr.history()


def ga(prob, T, rng, pop=50, pc=0.9, alpha=0.5, sig=0.1):
    """Real-coded generational GA: binary tournament, BLX-alpha, Gaussian
    mutation (p_m = 1/D), one elite."""
    D, lb, ub = prob.dim, prob.lb, prob.ub
    tr = Tracker(prob, T)
    X = rng.uniform(lb, ub, (pop, D)); f = tr(X)
    while not tr.done:
        i, j = rng.integers(pop, size=(2, pop))
        P = X[np.where(f[i] < f[j], i, j)]
        P1, P2 = P[0::2], P[1::2]
        lo, hi = np.minimum(P1, P2), np.maximum(P1, P2)
        I = hi - lo
        C1 = rng.uniform(lo - alpha * I, hi + alpha * I)
        C2 = rng.uniform(lo - alpha * I, hi + alpha * I)
        cx = rng.random(P1.shape[0]) < pc
        C1[~cx], C2[~cx] = P1[~cx], P2[~cx]
        C = np.vstack([C1, C2])
        m = rng.random(C.shape) < 1.0 / D
        C[m] += rng.normal(0, sig * (ub - lb), m.sum())
        C = np.clip(C, lb, ub)
        e = np.argmin(f); C[0] = X[e]
        fc = np.empty(pop); fc[0] = f[e]
        fn = tr(C[1:]); n = fn.size
        fc[1:1 + n] = fn
        if n < pop - 1:
            break
        X, f = C, fc
    return tr.history()


def pso(prob, T, rng, pop=40, w=0.7298, c1=1.49618, c2=1.49618, vmax=0.2):
    """Global-best PSO with constriction-equivalent coefficients."""
    D, lb, ub = prob.dim, prob.lb, prob.ub
    tr = Tracker(prob, T)
    X = rng.uniform(lb, ub, (pop, D)); V = np.zeros((pop, D))
    f = tr(X); P, fp = X.copy(), f.copy(); g = P[np.argmin(fp)].copy()
    vm = vmax * (ub - lb)
    while not tr.done:
        r1, r2 = rng.random((2, pop, D))
        V = np.clip(w * V + c1 * r1 * (P - X) + c2 * r2 * (g - X), -vm, vm)
        X = np.clip(X + V, lb, ub)
        f = tr(X); n = f.size
        imp = np.zeros(pop, bool); imp[:n] = f < fp[:n]
        P[imp], fp[imp] = X[imp], f[imp[:n]]
        g = P[np.argmin(fp)].copy()
    return tr.history()


def de(prob, T, rng, pop=50, F=0.5, CR=0.9):
    """DE/rand/1/bin."""
    D, lb, ub = prob.dim, prob.lb, prob.ub
    tr = Tracker(prob, T)
    X = rng.uniform(lb, ub, (pop, D)); f = tr(X)
    idx = np.arange(pop)
    while not tr.done:
        r = np.array([rng.choice(np.delete(idx, i), 3, replace=False) for i in idx])
        Vm = X[r[:, 0]] + F * (X[r[:, 1]] - X[r[:, 2]])
        cr = rng.random((pop, D)) < CR
        cr[idx, rng.integers(D, size=pop)] = True
        U = np.clip(np.where(cr, Vm, X), lb, ub)
        fu = tr(U); n = fu.size
        s = np.zeros(pop, bool); s[:n] = fu <= f[:n]
        X[s], f[s] = U[s], fu[s[:n]]
    return tr.history()


def gwo(prob, T, rng, pop=30):
    """Grey Wolf Optimizer (Mirjalili et al., 2014)."""
    D, lb, ub = prob.dim, prob.lb, prob.ub
    tr = Tracker(prob, T)
    X = rng.uniform(lb, ub, (pop, D)); f = tr(X)
    L = np.full((3, D), 0.0); fl = np.full(3, np.inf)

    def update_leaders(X, f):
        nonlocal L, fl
        allX = np.vstack([L, X]); allf = np.concatenate([fl, f])
        o = np.argsort(allf)[:3]
        L, fl = allX[o].copy(), allf[o].copy()

    update_leaders(X, f)
    iters = (T - pop) // pop
    it = 0
    while not tr.done:
        a = 2.0 - 2.0 * it / max(iters, 1)
        Xn = np.zeros_like(X)
        for l in range(3):
            A = 2 * a * rng.random((pop, D)) - a
            C = 2 * rng.random((pop, D))
            Xn += L[l] - A * np.abs(C * L[l] - X)
        X = np.clip(Xn / 3.0, lb, ub)
        f = tr(X)
        update_leaders(X[:f.size], f)
        it += 1
    return tr.history()


ALGOS = {"RS": random_search, "GA": ga, "PSO": pso, "DE": de, "GWO": gwo}


def best_at(ev, bs, checkpoints):
    """Best-so-far value at each checkpoint budget (step function)."""
    i = np.searchsorted(ev, checkpoints, side="right") - 1
    return bs[np.clip(i, 0, None)]


# ----------------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------------

def exp_null(M, runs, dim=10):
    """Experiment 1: calibration of ELI under pure random search."""
    rows, samples = [], {}
    for name in ["Sphere", "Rastrigin"]:
        prob = Problem(name, dim, shifted=True)
        ref = reference_sample(prob, M, seed=11)
        rng = np.random.default_rng(12)
        for t in [10, 100, 1000]:
            best = np.array([prob(rng.uniform(prob.lb, prob.ub, (t, dim))).min()
                             for _ in range(runs)])
            e, _ = eli_from_values(best, t, ref)
            samples[(name, t)] = e
            p = eli_pvalue(e)
            rows.append(dict(func=name, t=t, median=np.median(e), q05=np.quantile(e, 0.05),
                             q95=np.quantile(e, 0.95), mean=np.mean(e),
                             typeI=np.mean(p < 0.05)))
    th = dict(median=eli_null_quantile(0.5), q05=eli_null_quantile(0.05),
              q95=eli_null_quantile(0.95), mean=(np.euler_gamma + np.log(LN2)) / np.log(10))
    return rows, th, samples


def exp_tail_validation(dim, M_small, M_big, k0):
    """Experiment 2: validate the tail extension.
    (a) unshifted Sphere, where F(y) = V_D y^(D/2) / (2b)^D exactly for y <= b^2;
    (b) every function: fit the tail on M_small samples and predict the ELI of
        quantiles that only the M_big sample resolves empirically."""
    from math import gamma, pi
    out = {}
    # (a) analytic check
    prob = Problem("Sphere", dim, shifted=False)
    ref = reference_sample(prob, M_small, seed=21)
    te = TailELI(ref, L=0.0, k0=k0)
    VD = pi ** (dim / 2) / gamma(dim / 2 + 1)
    ys = np.logspace(-12, np.log10(min(te.u, prob.ub ** 2) * 0.99), 60)
    F_exact = VD * ys ** (dim / 2) / (2 * prob.ub) ** dim
    e_exact = np.log10(LN2 / (F_exact * 1000))
    e_tail = te.eli(ys, 1000)
    out["sphere"] = dict(alpha=te.alpha, alpha_true=dim / 2, maxerr=np.max(np.abs(e_tail - e_exact)),
                         ys=ys, e_exact=e_exact, e_tail=e_tail, u=te.u)
    # (b) cross-validation against a 100x larger empirical sample
    rows = []
    for name in FUNCS:
        p = Problem(name, dim, shifted=True)
        big = reference_sample(p, M_big, seed=7)
        small = reference_sample(p, M_small, seed=31)
        te = TailELI(small, L=0.0, k0=k0)
        # test points: the 1st..100th smallest values of the big sample,
        # i.e. quantiles 1e-7..1e-5, all far below the small sample's resolution
        y = big[[0, 1, 3, 9, 29, 99]]
        e_big, _ = eli_from_values(y, 1000, big)
        e_small_emp, k_small = eli_from_values(y, 1000, small)
        e_tail = te.eli(y, 1000)
        rows.append(dict(func=name, alpha=te.alpha, mae_tail=np.mean(np.abs(e_tail - e_big)),
                         max_tail=np.max(np.abs(e_tail - e_big)),
                         mae_cens=np.mean(np.abs(e_small_emp - e_big))))
    out["cv"] = rows
    return out


def run_suite(M, T, R, dim, shifted, funcs, algos, checkpoints, log=print):
    """Returns (res, refs); res[(func, algo)] = (R, C) best-so-far values at the
    checkpoints, refs[func] = sorted reference sample."""
    res, refs = {}, {}
    for name in funcs:
        prob = Problem(name, dim, shifted)
        t0 = time.time()
        refs[name] = reference_sample(prob, M, seed=7)
        for a in algos:
            B = np.empty((R, checkpoints.size))
            for r in range(R):
                rng = np.random.default_rng([zlib.crc32(name.encode()) % 10**6,
                                             list(ALGOS).index(a), r, int(shifted)])
                ev, bs = ALGOS[a](prob, T, rng)
                B[r] = best_at(ev, bs, checkpoints)
            res[(name, a)] = B
        log(f"  {name:<11} shifted={shifted}  done in {time.time() - t0:.1f}s")
    return res, refs


# ----------------------------------------------------------------------------
# Output helpers
# ----------------------------------------------------------------------------

def fval(x):
    return f"{x:.3f}" if x >= 1e-2 else f"{x:.1e}".replace("e-0", "e-")


def fmt(x, nd=2):
    return f"{x:.{nd}f}"


def fp(p):
    return r"$<10^{-16}$" if p < 1e-16 else "$" + f"{p:.1e}".replace("e-0", r"\times10^{-").replace("e-", r"\times10^{-") + "}$" if p < 0.01 else f"{p:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    args = ap.parse_args()

    dim, K0 = 10, 1000
    CV_TOL = 1.0  # accept the tail model if its cross-validated ELI error <= one decade
    if args.quick:
        M, T, R, NULLRUNS, M_null, M_small = 10**5, 2000, 5, 200, 10**5, 10**4
        K0 = 100
    else:
        M, T, R, NULLRUNS, M_null, M_small = 10**7, 10**4, 30, 2000, 10**6, 10**5

    resdir = os.path.join(args.out, "results"); figdir = os.path.join(args.out, "figures")
    os.makedirs(resdir, exist_ok=True); os.makedirs(figdir, exist_ok=True)
    checkpoints = np.unique(np.round(np.logspace(np.log10(50), np.log10(T), 25)).astype(int))
    ceiling = np.log10(M * LN2 / checkpoints)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.grid": True,
                         "grid.alpha": 0.3, "legend.fontsize": 7.5})
    COL = {"RS": "#7f7f7f", "GA": "#1f77b4", "PSO": "#2ca02c", "DE": "#d62728", "GWO": "#9467bd"}
    MK = {"RS": "o", "GA": "s", "PSO": "^", "DE": "D", "GWO": "v"}

    # ---- Worked example ----------------------------------------------------
    ref10 = np.sort(np.array([7.2, -3.1, 5.5, -8.4, 1.9, -0.8, 9.3, -4.6, 2.7, -6.0]) ** 2)
    with open(os.path.join(resdir, "example.csv"), "w") as fh:
        fh.write("alg,best,k,ELI,p\n")
        for alg, y in [("A", 0.30), ("B", 5.00), ("C", 25.00)]:
            e, k = eli_from_values(np.array([y]), 2, ref10)
            fh.write(f"{alg},{y},{int(k[0])},{e[0]:.4f},{eli_pvalue(e)[0]:.4f}\n")
            print(f"Example {alg}: k={int(k[0])} ELI={e[0]:.3f} p={eli_pvalue(e)[0]:.3f}")

    # ---- Experiment 1 ------------------------------------------------------
    print("Experiment 1: calibration under random search")
    rows, th, samples = exp_null(M_null, NULLRUNS, dim)
    with open(os.path.join(resdir, "table_null.tex"), "w") as fh:
        for r in rows:
            fh.write(f"{r['func']} & {r['t']} & {fmt(r['median'])} & {fmt(r['q05'])} & "
                     f"{fmt(r['q95'])} & {fmt(r['mean'])} & {fmt(r['typeI'], 3)} \\\\\n")
        fh.write(f"\\hline\nTheory & $\\infty$ & {fmt(abs(th['median']))} & {fmt(th['q05'])} & "
                 f"{fmt(th['q95'])} & {fmt(th['mean'])} & 0.050 \\\\\n")
    for r in rows:
        print("   ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    xs = np.linspace(-1.5, 2.0, 400)
    dens = np.log(10) * LN2 * 10 ** (-xs) * np.power(2.0, -10.0 ** (-xs))
    ax.hist(samples[("Rastrigin", 1000)], bins=40, density=True, alpha=0.45, color="#1f77b4",
            label="RS, Rastrigin, $t=1000$")
    ax.hist(samples[("Sphere", 10)], bins=40, density=True, histtype="step", color="#d62728",
            label="RS, Sphere, $t=10$")
    ax.plot(xs, dens, "k-", lw=1.4, label=r"Theory, $\log_{10}(\ln 2/E)$")
    ax.axvline(eli_null_quantile(0.95), color="k", ls=":", lw=1)
    ax.text(eli_null_quantile(0.95) + 0.04, 0.55, "95%", fontsize=7)
    ax.set_xlabel("ELI"); ax.set_ylabel("Density"); ax.legend(loc="upper left", fontsize=6.5)
    ax.set_ylim(0, 1.05)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "fig_null.pdf")); plt.close(fig)

    # ---- Experiment 2 ------------------------------------------------------
    print("Experiment 2: tail-extension validation")
    tv = exp_tail_validation(dim, M_small, M, K0)
    s = tv["sphere"]
    print(f"    Sphere analytic: alpha_hat={s['alpha']:.3f} (true {s['alpha_true']}), "
          f"max|dELI|={s['maxerr']:.3f}, u={s['u']:.1f}")
    with open(os.path.join(resdir, "table_tail.tex"), "w") as fh:
        for r in tv["cv"]:
            fh.write(f"{r['func']} & {fmt(r['alpha'])} & {fmt(r['mae_cens'])} & "
                     f"{fmt(r['mae_tail'])} & {fmt(r['max_tail'])} \\\\\n")
            print("   ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    with open(os.path.join(resdir, "tail_sphere.txt"), "w") as fh:
        fh.write(f"alpha_hat={s['alpha']:.4f} alpha_true={s['alpha_true']} "
                 f"max_abs_err={s['maxerr']:.4f} u={s['u']:.3f}\n")
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    ax.plot(s["ys"], s["e_exact"], "k-", lw=1.4, label="Exact (analytic $F$)")
    ax.plot(s["ys"][::3], s["e_tail"][::3], "o", ms=3, mfc="none", color="#d62728",
            label=rf"Tail-extended, $\hat\alpha={s['alpha']:.2f}$")
    ax.axvline(s["u"], color="#7f7f7f", ls=":", lw=1)
    ax.text(s["u"] * 0.6, ax.get_ylim()[0] + 2, "$u$", fontsize=8, ha="right")
    ax.set_xscale("log"); ax.set_xlabel("Best objective value $y$"); ax.set_ylabel("ELI ($t=1000$)")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "fig_tail.pdf")); plt.close(fig)

    # ---- Experiment 3: shifted suite -------------------------------------
    print("Experiment 3: shifted suite")
    funcs, algos = list(FUNCS), list(ALGOS)
    res, refs = run_suite(M, T, R, dim, True, funcs, algos, checkpoints)
    use_tail = {r["func"]: bool(r["mae_tail"] <= CV_TOL) for r in tv["cv"]}
    print("    tail extension accepted:", use_tail)
    tails = {n: TailELI(refs[n], L=0.0, k0=K0, tail=use_tail[n]) for n in funcs}
    ELI = {key: tails[key[0]].eli(B, checkpoints) for key, B in res.items()}

    fig, axs = plt.subplots(2, 3, figsize=(7.0, 4.3), sharex=True)
    for ax, name in zip(axs.ravel(), funcs):
        for a in algos:
            E = ELI[(name, a)]
            q1, med, q3 = np.quantile(E, [0.25, 0.5, 0.75], axis=0)
            ax.plot(checkpoints, med, color=COL[a], marker=MK[a], ms=2.5, lw=1, label=a)
            ax.fill_between(checkpoints, q1, q3, color=COL[a], alpha=0.12, lw=0)
        ax.plot(checkpoints, ceiling, "k--", lw=0.9, label="Empirical ceiling")
        ax.axhline(eli_null_quantile(0.95), color="k", ls=":", lw=0.8)
        ax.set_xscale("log"); ax.set_title(name)
    for ax in axs[1]:
        ax.set_xlabel("Evaluations $t$")
    for ax in axs[:, 0]:
        ax.set_ylabel("ELI$(t)$")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, ncol=6, loc="lower center", fontsize=7.5, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(figdir, "fig_anytime.pdf")); plt.close(fig)

    summary = {}
    with open(os.path.join(resdir, "table_suite.tex"), "w") as fh, \
         open(os.path.join(resdir, "suite.csv"), "w") as fc:
        fc.write("func,algo,median_fT,eliT_med,eliT_q1,eliT_q3,emp_cens_pct,fisher_p,eli1000_med\n")
        for name in funcs:
            meds = {a: np.median(ELI[(name, a)][:, -1]) for a in algos}
            best_a = max(algos, key=meds.get)
            if sum(np.isclose(meds[a], meds[best_a]) for a in algos) > 1:
                best_a = None  # tie (e.g. all censored at the ceiling): no bold
            for j, a in enumerate(algos):
                eT = ELI[(name, a)][:, -1]
                k_emp = np.searchsorted(refs[name], res[(name, a)][:, -1])
                cens = 100 * np.mean(k_emp == 0)  # runs beyond the empirical ceiling
                pf = fisher_combine(eli_pvalue(eT))
                e1k = np.median(ELI[(name, a)][:, np.argmin(np.abs(checkpoints - 1000))])
                fT = np.median(res[(name, a)][:, -1])
                summary[(name, a)] = dict(eT=eT, fT=res[(name, a)][:, -1], pf=pf)
                fc.write(f"{name},{a},{fT},{np.median(eT)},{np.quantile(eT,.25)},{np.quantile(eT,.75)},"
                         f"{cens},{pf},{e1k}\n")
                prefix = f"\\multirow{{5}}{{*}}{{{name}}}" if j == 0 else ""
                ge = r"$\geq$" if (not use_tail[name] and np.median(k_emp) == 0) else ""
                es = f"{ge}{fmt(np.median(eT))} [{fmt(np.quantile(eT,.25))}, {fmt(np.quantile(eT,.75))}]"
                if a == best_a:
                    es = r"\textbf{" + es + "}"
                fh.write(f"{prefix} & {a} & {fval(fT)} & {es} & {int(round(cens))} & {fp(pf)} \\\\\n")
            fh.write("\\hline\n")

    with open(os.path.join(resdir, "wilcoxon.csv"), "w") as fh:
        fh.write("func,best,second,p\n")
        for name in funcs:
            order = sorted(algos, key=lambda a: -np.median(summary[(name, a)]["eT"]))
            b, s2 = order[0], order[1]
            p = stats.wilcoxon(summary[(name, b)]["eT"], summary[(name, s2)]["eT"]).pvalue
            fh.write(f"{name},{b},{s2},{p}\n")
            print(f"    {name}: best {b} vs {s2}: Wilcoxon p={p:.2e}")

    # ---- Experiment 4: aggregation & invariance ---------------------------
    print("Experiment 4: aggregation and transformation invariance")
    transforms = {
        "Identity": lambda v: v,
        r"$\log_{10}(1+f)$": lambda v: np.log1p(v) / np.log(10.0),
        r"$\sqrt{f}$": lambda v: np.sqrt(np.maximum(v, 0)),
    }
    lines = []
    for tn, g in transforms.items():
        mean_err = [np.mean([np.mean(g(summary[(n, a)]["fT"])) for n in funcs]) for a in algos]
        eli_g = {}
        for a in algos:
            vals = []
            for n in funcs:
                te = TailELI(g(refs[n]), L=float(g(np.array([0.0]))[0]), k0=K0, tail=use_tail[n])
                vals.append(np.median(te.eli(g(summary[(n, a)]["fT"]), T)))
            eli_g[a] = np.mean(vals)
        lines.append((tn, stats.rankdata(mean_err), stats.rankdata([-eli_g[a] for a in algos]),
                      [eli_g[a] for a in algos]))
        print(f"    {tn}: err-rank {stats.rankdata(mean_err)}  ELI {[round(eli_g[a],2) for a in algos]}")
    with open(os.path.join(resdir, "table_agg.tex"), "w") as fh:
        for tn, rk_e, rk_l, vals in lines:
            fh.write(f"{tn} & " + " & ".join(str(int(x)) for x in rk_e) + " & " +
                     " & ".join(str(int(r)) for r in rk_l) + " \\\\\n")

    # ---- Experiment 5: centre bias -----------------------------------------
    print("Experiment 5: centre bias")
    cb_funcs, cb_algos = ["Sphere", "Rastrigin", "Ackley", "Griewank"], ["PSO", "DE", "GWO"]
    res_u, refs_u = run_suite(M, T, R, dim, False, cb_funcs, cb_algos, checkpoints)
    tails_u = {n: TailELI(refs_u[n], L=0.0, k0=K0, tail=use_tail[n]) for n in cb_funcs}
    ELI_u = {key: tails_u[key[0]].eli(B, checkpoints) for key, B in res_u.items()}
    with open(os.path.join(resdir, "table_centre.tex"), "w") as fh, \
         open(os.path.join(resdir, "centre.csv"), "w") as fc:
        fc.write("func,algo,eliT_unshifted,eliT_shifted,delta,fT_unshifted,fT_shifted\n")
        for name in cb_funcs:
            cells = []
            for a in cb_algos:
                eu = np.median(ELI_u[(name, a)][:, -1]); es_ = np.median(ELI[(name, a)][:, -1])
                cells.append(f"{fmt(eu)} & {fmt(es_)} & {fmt(es_ - eu)}")
                fc.write(f"{name},{a},{eu},{es_},{es_-eu},{np.median(res_u[(name,a)][:,-1])},"
                         f"{np.median(res[(name,a)][:,-1])}\n")
            fh.write(f"{name} & " + " & ".join(cells) + " \\\\\n")
            print("   ", name, cells)

    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4))
    for ax, name in zip(axs, ["Rastrigin", "Griewank"]):
        for a in cb_algos:
            ax.plot(checkpoints, np.median(ELI_u[(name, a)], 0), color=COL[a], ls="-",
                    marker=MK[a], ms=2.5, lw=1, label=f"{a}, unshifted")
            ax.plot(checkpoints, np.median(ELI[(name, a)], 0), color=COL[a], ls="--",
                    lw=1, label=f"{a}, shifted")
        ax.set_xscale("log"); ax.set_title(name); ax.set_xlabel("Evaluations $t$")
    axs[0].set_ylabel("Median ELI$(t)$"); axs[0].legend(fontsize=6.3, ncol=2, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "fig_centre.pdf")); plt.close(fig)

    with open(os.path.join(resdir, "settings.txt"), "w") as fh:
        fh.write(f"dim={dim} M={M} T={T} R={R} k0={K0} M_null={M_null} null_runs={NULLRUNS} "
                 f"M_small={M_small}\n")
        fh.write(f"empirical_ceiling_at_T={ceiling[-1]:.3f} at_t0={ceiling[0]:.3f}\n")
        fh.write("alphas=" + ", ".join(f"{n}:{tails[n].alpha:.3f}" for n in funcs) + "\n")
        fh.write("checkpoints=" + ",".join(map(str, checkpoints)) + "\n")
    print("Done.")


if __name__ == "__main__":
    main()
