"""
Surrogate-assisted multi-objective optimisation of a tube heat exchanger
+ a sequential-learning (Bayesian optimisation) comparison.

v2 changes vs. v1:
  1. Physically consistent model (epsilon-NTU heat transfer, Blasius-type pressure drop):
     pressure drop now INCREASES with tube length, as it should.
  2. Reproducible (fixed random seeds).
  3. Surrogates validated on a held-out test set (R2 on unseen data, not on training data).
  4. Pump limit handled as a real constraint inside NSGA-II (epsilon-constraint style),
     not by filtering the Pareto front afterwards.
  5. The final design is re-checked with the true model.
  6. NEW: sequential learning - a Gaussian-process + Expected-Improvement loop finds a
     good constrained design with ~30 model evaluations instead of 1,000.
"""
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.stats import norm
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

SEED = 42
warnings.filterwarnings("ignore")   # silence GP hyper-parameter convergence warnings
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------- 1. "expensive" model
CP = 4180.0          # J/(kg K), water
T_COLD = 20.0        # degC, wall / cold side
D = 0.02             # m, tube diameter
LOWER = np.array([0.1, 60.0, 1.0])    # flow rate [kg/s], inlet T [degC], length [m]
UPPER = np.array([0.5, 100.0, 5.0])
DP_MAX = 40.0        # kPa, pump limit


def true_model(X):
    """Return heat duty Q [kW] and pressure drop dP [kPa] (stand-in for a CFD run)."""
    m, t_in, L = X[:, 0], X[:, 1], X[:, 2]
    h = 2500.0 * (m / 0.3) ** 0.8                 # W/(m2 K), Dittus-Boelter-like scaling
    ua = h * np.pi * D * L                         # W/K
    ntu = ua / (m * CP)
    eff = 1.0 - np.exp(-ntu)                       # effectiveness, constant-wall-T case
    q = m * CP * (t_in - T_COLD) * eff / 1000.0    # kW
    dp = 25.0 * L * (m / 0.3) ** 1.75              # kPa, Blasius-type
    return q, dp


def sample(n):
    return LOWER + rng.random((n, 3)) * (UPPER - LOWER)


# ---------------------------------------------------------------- 2. DoE + surrogates
X = sample(1000)
Q, P = true_model(X)
X_tr, X_te, Q_tr, Q_te, P_tr, P_te = train_test_split(X, Q, P, test_size=0.2, random_state=SEED)

rf_q = RandomForestRegressor(n_estimators=300, random_state=SEED).fit(X_tr, Q_tr)
rf_p = RandomForestRegressor(n_estimators=300, random_state=SEED).fit(X_tr, P_tr)
print(f"Held-out R2  heat duty: {r2_score(Q_te, rf_q.predict(X_te)):.4f}")
print(f"Held-out R2  pressure drop: {r2_score(P_te, rf_p.predict(X_te)):.4f}")


# ---------------------------------------------------------------- 3. NSGA-II with constraint
class HeatExchangerProblem(Problem):
    def __init__(self):
        super().__init__(n_var=3, n_obj=2, n_ieq_constr=1, xl=LOWER, xu=UPPER)

    def _evaluate(self, x, out, *args, **kwargs):
        q, dp = rf_q.predict(x), rf_p.predict(x)
        out["F"] = np.column_stack([-q, dp])        # maximise Q, minimise dP
        out["G"] = dp - DP_MAX                      # dP <= DP_MAX


res = minimize(HeatExchangerProblem(), NSGA2(pop_size=100), ("n_gen", 100), seed=SEED, verbose=False)
pareto_q, pareto_p = -res.F[:, 0], res.F[:, 1]
best = np.argmax(pareto_q)
q_true, p_true = true_model(res.X[best:best + 1])
print("\nSelected design (max Q with dP <= limit):")
print(f"  x = flow {res.X[best, 0]:.3f} kg/s, T_in {res.X[best, 1]:.1f} C, L {res.X[best, 2]:.2f} m")
print(f"  surrogate: Q = {pareto_q[best]:.1f} kW, dP = {pareto_p[best]:.1f} kPa")
print(f"  true model: Q = {q_true[0]:.1f} kW, dP = {p_true[0]:.1f} kPa"
      + ("  -> violates the pump limit!" if p_true[0] > DP_MAX else "  -> feasible"))

# reference: true constrained optimum by brute force (only possible because the model is cheap)
Xg = sample(200_000)
qg, pg = true_model(Xg)
q_ref = qg[pg <= DP_MAX].max()
print(f"  true constrained optimum (brute force, 200k evaluations): Q = {q_ref:.1f} kW")


# ---------------------------------------------------------------- 4. sequential learning (BO)
def fit_gp(Xs, y):
    kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(3), nu=2.5) + WhiteKernel(1e-6)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3, random_state=SEED)
    return gp.fit((Xs - LOWER) / (UPPER - LOWER), y)


def constrained_ei(Xc, gp_q, gp_p, best_feasible):
    Z = (Xc - LOWER) / (UPPER - LOWER)
    mu_q, sd_q = gp_q.predict(Z, return_std=True)
    mu_p, sd_p = gp_p.predict(Z, return_std=True)
    sd_q = np.maximum(sd_q, 1e-9)
    z = (mu_q - best_feasible) / sd_q
    ei = (mu_q - best_feasible) * norm.cdf(z) + sd_q * norm.pdf(z)      # expected improvement
    p_feas = norm.cdf((DP_MAX - mu_p) / np.maximum(sd_p, 1e-9))         # probability dP <= limit
    return ei * p_feas


def bayesian_optimisation(n_init=8, budget=30):
    Xs = sample(n_init)
    q, p = true_model(Xs)
    history = []
    for _ in range(budget - n_init):
        feas = p <= DP_MAX
        best_f = q[feas].max() if feas.any() else q.min()
        history.append(best_f)
        gp_q, gp_p = fit_gp(Xs, q), fit_gp(Xs, p)
        cand = sample(5000)
        x_next = cand[np.argmax(constrained_ei(cand, gp_q, gp_p, best_f))][None, :]
        q_n, p_n = true_model(x_next)
        Xs, q, p = np.vstack([Xs, x_next]), np.append(q, q_n), np.append(p, p_n)
    feas = p <= DP_MAX
    history.append(q[feas].max())
    return np.array(history)


def random_search(n_init=8, budget=30):
    Xs = sample(budget)
    q, p = true_model(Xs)
    q_feas = np.where(p <= DP_MAX, q, -np.inf)
    return np.maximum.accumulate(q_feas)[n_init - 1:]


bo = bayesian_optimisation()
rs = random_search()
print(f"\nBest feasible Q after 30 true-model evaluations:")
print(f"  Bayesian optimisation: {bo[-1]:.1f} kW")
print(f"  Random sampling:       {rs[-1]:.1f} kW")
print(f"  True optimum (reference): {q_ref:.1f} kW")

# ---------------------------------------------------------------- 5. plots
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
ax[0].scatter(pareto_q, pareto_p, s=14, color="tab:red", label="Pareto front (surrogate)")
ax[0].scatter(q_true, p_true, s=80, marker="*", color="k", label="Selected design (true model)")
ax[0].axhline(DP_MAX, ls="--", color="gray", label="Pump limit")
ax[0].set_xlabel("Heat duty Q [kW] (maximise)")
ax[0].set_ylabel("Pressure drop [kPa] (minimise)")
ax[0].set_title("NSGA-II on Random-Forest surrogates")
ax[0].legend(); ax[0].grid(alpha=0.3)

n = np.arange(8, 8 + len(bo))
ax[1].plot(n, bo, "o-", label="Bayesian optimisation (GP + constrained EI)")
ax[1].plot(n, rs, "s--", label="Random sampling")
ax[1].axhline(q_ref, color="k", ls=":", label="True constrained optimum")
ax[1].set_xlabel("Number of true-model evaluations")
ax[1].set_ylabel("Best feasible Q [kW]")
ax[1].set_title("Sequential learning vs. random sampling")
ax[1].legend(); ax[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results_v2.png", dpi=150)
print("\nSaved figure: results_v2.png")
