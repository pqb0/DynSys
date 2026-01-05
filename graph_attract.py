
import sys, math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from abc import ABC, abstractmethod



# 1. ABSTRACT BASE CLASS FOR MAPS

class DiscreteMap(ABC):
    """
    Base class for 2‑D discrete‑time maps.  Sub‑classes must implement
    ``step(self, x)`` and may override ``escape_radius``.
    """
    dim = 2
    escape_radius = 100.0                     # generic cutoff

    @abstractmethod
    def step(self, x):
        """One iteration."""
        pass

    def random_seed(self):
        """Default random initial condition."""
        return np.random.uniform(-1, 1, self.dim)



# 2. MAP CLASSES (original + all new ones)

# -------------------------------------------------
# 2.1  Original maps
# -------------------------------------------------
class HenonMap(DiscreteMap):
    def __init__(self, a=1.4, b=0.3):
        self.a = a
        self.b = b
        self.escape_radius = 4.0

    def step(self, x):
        return np.array([1.0 - self.a * x[0] ** 2 + x[1],
                         self.b * x[0]])


class CoupledLogisticMap(DiscreteMap):
    def __init__(self, r=3.137, s=0.85):
        self.r = r
        self.s = s
        self.escape_radius = 4.0

    def step(self, x):
        return np.array([self.r * x[0] * (1.0 - x[0]) + self.s * x[1],
                         self.r * x[1] * (1.0 - x[1]) + self.s * x[0]])


class IkedaMap(DiscreteMap):
    def __init__(self, u=1.339):
        self.u = u
        self.escape_radius =10.0

    def step(self, x):
        t = 0.4 - 6.0 / (1.0 + x[0] ** 2 + x[1] ** 2)
        c, s = np.cos(t), np.sin(t)
        return np.array([1.0 + self.u * (x[0] * c - x[1] * s),
                         self.u * (x[0] * s + x[1] * c)])


# -------------------------------------------------
# 2.2  New maps
# -------------------------------------------------
class DeJongDampedMap(DiscreteMap):
    """Damped De Jong attractor."""
    def __init__(self, a=2.01, b=-2.53, c=1.61, d=-0.33, rho=0.98):
        self.a, self.b, self.c, self.d, self.rho = a, b, c, d, rho
        self.escape_radius = 5.0

    def step(self, x):
        x_new = self.rho * (np.sin(self.a * x[1]) - np.cos(self.b * x[0]))
        y_new = self.rho * (np.sin(self.c * x[0]) - np.cos(self.d * x[1]))
        return np.array([x_new, y_new])


class SkewTentMap(DiscreteMap):
    """2‑D skew‑tent (component‑wise)."""
    def __init__(self, s=0.3):
        if not (0.0 < s < 1.0):
            raise ValueError("skew parameter s must be in (0,1)")
        self.s = s
        self.escape_radius = 2.0

    def _tent(self, u):
        return np.where(u < self.s,
                        u / self.s,
                        (1.0 - u) / (1.0 - self.s))

    def step(self, x):
        # map from [0,1] → [0,1] then centre at 0
        return self._tent(x) - 0.5


class LoziMap(DiscreteMap):
    """Piecewise‑linear Henon‑type."""
    def __init__(self, a=1.5, b=0.5):
        self.a, self.b = a, b
        self.escape_radius = 3.0

    def step(self, x):
        return np.array([1.0 - self.a * abs(x[0]) + self.b * x[1],
                         x[0]])


class ArnoldCatMap(DiscreteMap):
    """Linear, area‑preserving cat map on the torus."""
    def __init__(self):
        self.escape_radius = 1e6

    def step(self, x):
        return np.array([(x[0] + x[1]) % 1.0,
                         (x[0] + 2.0 * x[1]) % 1.0])


class BakersMap(DiscreteMap):
    """Classic baker’s map."""
    def __init__(self):
        self.escape_radius = 1e6

    def step(self, x):
        if x[0] < 0.5:
            return np.array([2.0 * x[0], 0.5 * x[1]])
        else:
            return np.array([2.0 * x[0] - 1.0,
                             0.5 * (x[1] + 1.0)])


class StandardMap(DiscreteMap):
    """Chirikov standard (kicked‑rotor) map."""
    def __init__(self, K=1.0):
        self.K = K
        self.escape_radius = 1e6

    def step(self, x):
        theta, p = x
        p_new = (p + self.K * np.sin(theta)) % (2.0 * np.pi)
        theta_new = (theta + p_new) % (2.0 * np.pi)
        return np.array([theta_new, p_new])


class CoupledSineMap(DiscreteMap):
    """Smooth nonlinear coupling via sine."""
    def __init__(self, a=1.9, b=0.9, c=0.9, d=1.9):
        self.a, self.b, self.c, self.d = a, b, c, d
        self.escape_radius = 4.0

    def step(self, x):
        return np.array([self.a * np.sin(x[0]) + self.b * np.sin(x[1]),
                         self.c * np.sin(x[1]) + self.d * np.sin(x[0])])


class DuffingMap(DiscreteMap):
    """Discrete‑time Duffing map (sampled once per forcing period)."""
    def __init__(self,
                 delta=0.2,
                 alpha=-1.0,
                 beta=1.0,
                 gamma=0.3,
                 omega=1.0):
        self.delta = delta
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.omega = omega
        self.phase = 0.0
        self.escape_radius = 5.0

    def step(self, x):
        xn, yn = x
        x_next = yn
        y_next = (-self.delta * yn
                  - self.alpha * xn
                  - self.beta * (xn ** 3)
                  + self.gamma * np.cos(self.phase))
        self.phase = (self.phase + self.omega) % (2.0 * np.pi)
        return np.array([x_next, y_next])


class RabinovichFabrikantMap(DiscreteMap):
    """2‑D projection of the 3‑D Rabinovich‑Fabrikant system."""
    def __init__(self, a=0.87, beta=0.87, gamma=0.87):
        self.a = a
        self.beta = beta
        self.gamma = gamma
        self.z = 0.0
        self.escape_radius = 5.0

    def step(self, x):
        xn, yn = x
        z_next = 1.0 - self.a * self.z - (xn ** 2 + yn ** 2)
        x_next = xn * (self.z + self.gamma) - yn * self.beta
        y_next = xn * self.beta + yn * (self.z + self.gamma)
        self.z = z_next
        return np.array([x_next, y_next])



# 3. ORBIT GENERATOR

def generate_orbit(system: DiscreteMap,
                   n_points: int = 1_000_000,
                   discard: int = 5_000):
    """
    Generate a long orbit for *system*.
    *discard* points are thrown away as transient.
    Returns a (N,2) float32 numpy array.
    """
    xs = []
    x = system.random_seed()
    discard_left = discard

    while len(xs) < n_points:
        x = system.step(x)

        # Reject diverging or non‑finite points
        if (not np.all(np.isfinite(x))) or np.linalg.norm(x) > system.escape_radius:
            x = system.random_seed()
            continue

        if discard_left > 0:
            discard_left -= 1
            continue

        xs.append(x.copy())

    return np.array(xs, dtype=np.float32)



# 4. SRB DENSITY ESTIMATION (2‑D histogram)

def density_coloring(points, bins: int = 1000):
    """
    Compute a 2‑D histogram density for *points*.
    Returns an array of the same length as *points* containing the count
    of hits in the bin each point belongs to.
    """
    x, y = points[:, 0], points[:, 1]
    xlim = (x.min(), x.max())
    ylim = (y.min(), y.max())

    H, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[xlim, ylim])
    H = H.T  # transpose to match (y, x) indexing

    ix = np.clip(np.searchsorted(xedges, x, side='right') - 1, 0, bins - 1)
    iy = np.clip(np.searchsorted(yedges, y, side='right') - 1, 0, bins - 1)

    density = H[iy, ix]
    density[density <= 0] = 1      # safe for LogNorm
    return density



# 5. PLOTTER (Matplotlib)

def plot_attractor(points, density, title="Strange Attractor", cmap="inferno"):
    """Scatter plot coloured by SRB density (logarithmic colormap)."""
    plt.figure(figsize=(8, 8))
    sc = plt.scatter(points[:, 0], points[:, 1],
                     c=density, s=0.15, cmap=cmap,
                     norm=LogNorm(), linewidths=0)
    plt.axis('equal')
    plt.axis('off')
    plt.title(title, fontsize=14)

    cbar = plt.colorbar(sc, fraction=0.046, pad=0.04)
    cbar.set_label('Invariant density (log scale)')

    plt.tight_layout()
    plt.show()



# 6. MAP REGISTRY (name → constructor & default parameters)

# Each entry is a tuple:
#   (callable that returns a new map object,
#    ordered list of (parameter_name, default_value))
MAPS = {
    # original
    "henon":   (lambda: HenonMap(),
                [("a", 1.4), ("b", 0.3)]),
    "logistic":(lambda: CoupledLogisticMap(),
                [("r", 3.137), ("s", 0.85)]),
    "ikeda":   (lambda: IkedaMap(),
                [("u", 1.339)]),

    # new maps
    "dejong":     (lambda: DeJongDampedMap(),
                   [("a", 2.01), ("b", -2.53), ("c", 1.61), ("d", -0.33), ("rho", 0.98)]),
    "skew_tent": (lambda: SkewTentMap(),
                   [("s", 0.3)]),
    "lozi":       (lambda: LoziMap(),
                   [("a", 1.5), ("b", 0.5)]),
    "cat":        (lambda: ArnoldCatMap(),
                   []),
    "baker":      (lambda: BakersMap(),
                   []),
    "standard":   (lambda: StandardMap(),
                   [("K", 1.0)]),
    "sine":       (lambda: CoupledSineMap(),
                   [("a", 1.9), ("b", 0.9), ("c", 0.9), ("d", 1.9)]),
    "duffing":    (lambda: DuffingMap(),
                   [("delta", 0.2), ("alpha", -1.0), ("beta", 1.0),
                    ("gamma", 0.3), ("omega", 1.0)]),
    "rab_fab":    (lambda: RabinovichFabrikantMap(),
                   [("a", 0.87), ("beta", 0.87), ("gamma", 0.87)]),
}



# 7. USER INTERACTION – ask for N, map, parameters

def ask_int(prompt, default):
    """Prompt for an integer, using default if empty."""
    while True:
        s = input(f"{prompt} [{default}]: ").strip()
        if s == "":
            return default
        try:
            val = int(s)
            if val <= 0:
                raise ValueError
            return val
        except ValueError:
            print("Please enter a positive integer.")


def ask_float(prompt, default):
    """Prompt for a float, using default if empty."""
    while True:
        s = input(f"{prompt} [{default}]: ").strip()
        if s == "":
            return default
        try:
            return float(s)
        except ValueError:
            print("Please enter a numeric value.")


def main():
    print("\n=== Strange‑Attractor Explorer ===\n")

    # -------------------------------------------------
    # 1) Number of points
    # -------------------------------------------------
    N = ask_int("Number of points to generate (N)", 500_000)

    # -------------------------------------------------
    # 2) Choose map
    # -------------------------------------------------
    print("\nAvailable maps:")
    map_keys = list(MAPS.keys())
    for i, name in enumerate(map_keys, 1):
        print(f"  {i}. {name}")

    while True:
        choice = input(f"\nSelect a map [1‑{len(map_keys)}] (default 1): ").strip()
        if choice == "":
            idx = 0
            break
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(map_keys):
                break
            else:
                raise ValueError
        except ValueError:
            print("Invalid selection – try again.")

    map_name = map_keys[idx]
    map_factory, param_spec = MAPS[map_name]

    # -------------------------------------------------
    # 3) Ask for parameters (show defaults)
    # -------------------------------------------------
    print(f"\nEnter parameters for {map_name} (press Enter to keep default):")
    user_params = {}
    for pname, default in param_spec:
        val = ask_float(f"  {pname}", default)
        user_params[pname] = val

    # -------------------------------------------------
    # 4) Build the map instance using the user parameters
    # -------------------------------------------------
    # The constructors accept keyword arguments that match the
    # parameter names defined above.
    system = map_factory()
    for pname, val in user_params.items():
        setattr(system, pname, val)   # set the attribute directly

    print("\n[INFO] Generating orbit …")
    points = generate_orbit(system, n_points=N, discard=5_000)
    print(f"[INFO] Points generated: {len(points):,}")

    print("[INFO] Computing density …")
    density = density_coloring(points, bins=1000)

    print("[INFO] Plotting …")
    plot_attractor(points, density, title=system.__class__.__name__)

    print("\nDone. Close the plot window to exit.\n")


if __name__ == "__main__":
    main()

# a = -1.246 b = 0.9 c =1.269  d = 2.354


# r 0.779     b 0.962

