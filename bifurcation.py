
import sys, math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from abc import ABC, abstractmethod



# 1. ABSTRACT BASE CLASS

class DiscreteMap(ABC):
    """Base class for all 2‑D discrete maps."""
    dim = 2
    escape_radius = 50.0

    @abstractmethod
    def step(self, x):
        """One iteration."""
        pass

    def random_seed(self):
        """Default random initial condition."""
        return np.random.uniform(-1, 1, self.dim)



# 2. MAP CLASSES (original + the new ones)


# 2.1  Original maps

class HenonMap(DiscreteMap):
    def __init__(self, a=1.4, b=0.3):
        self.a, self.b = a, b
        self.escape_radius = 4.0

    def step(self, x):
        return np.array([1.0 - self.a * x[0] ** 2 + x[1],
                         self.b * x[0]])


class CoupledLogisticMap(DiscreteMap):
    def __init__(self, r=3.137, s=0.85):
        self.r, self.s = r, s
        self.escape_radius = 4.0

    def step(self, x):
        return np.array([self.r * x[0] * (1 - x[0]) + self.s * x[1],
                         self.r * x[1] * (1 - x[1]) + self.s * x[0]])


class IkedaMap(DiscreteMap):
    def __init__(self, u=1.339):
        self.u = u
        self.escape_radius = 4.0

    def step(self, x):
        t = 0.4 - 6.0 / (1.0 + x[0] ** 2 + x[1] ** 2)
        c, s = np.cos(t), np.sin(t)
        return np.array([1.0 + self.u * (x[0] * c - x[1] * s),
                         self.u * (x[0] * s + x[1] * c)])



# 2.2  New maps

class DeJongDampedMap(DiscreteMap):
    """
    Damped De Jong attractor:

        x' = ρ ( sin(a·y) – cos(b·x) )
        y' = ρ ( sin(c·x) – cos(d·y) )
    """
    def __init__(self, a=2.01, b=-2.53, c=1.61, d=-0.33, rho=0.98):
        self.a, self.b, self.c, self.d, self.rho = a, b, c, d, rho
        self.escape_radius = 5.0

    def step(self, x):
        x_new = self.rho * (np.sin(self.a * x[1]) -
                            np.cos(self.b * x[0]))
        y_new = self.rho * (np.sin(self.c * x[0]) -
                            np.cos(self.d * x[1]))
        return np.array([x_new, y_new])


class SkewTentMap(DiscreteMap):
    """Component‑wise skew‑tent map (parameter s∈(0,1))."""
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
    """Piecewise‑linear Henon‑type map."""
    def __init__(self, a=1.5, b=0.5):
        self.a, self.b = a, b
        self.escape_radius = 3.0

    def step(self, x):
        return np.array([1.0 - self.a * abs(x[0]) + self.b * x[1],
                         x[0]])


class ArnoldCatMap(DiscreteMap):
    """Linear area‑preserving cat map on the unit torus."""
    def __init__(self):
        self.escape_radius = 1e6

    def step(self, x):
        return np.array([(x[0] + x[1]) % 1.0,
                         (x[0] + 2.0 * x[1]) % 1.0])


class BakersMap(DiscreteMap):
    """Classic baker’s map (piecewise linear)."""
    def __init__(self):
        self.escape_radius = 1e6

    def step(self, x):
        if x[0] < 0.5:
            return np.array([2.0 * x[0], 0.5 * x[1]])
        else:
            return np.array([2.0 * x[0] - 1.0,
                             0.5 * (x[1] + 1.0)])


class StandardMap(DiscreteMap):
    """Chirikov standard (kicked rotor) map."""
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
    """
    Discrete Duffing map (sampled once per forcing period).
    Parameters: δ (damping), α (linear), β (cubic), γ (forcing amplitude),
                ω (phase increment).
    """
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
    """
    2‑D projection of the 3‑D Rabinovich‑Fabrikant system.
    """
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



# 3. MAP REGISTRY

# Each entry: (class, list of (parameter_name, default_value))
MAPS = {
    # original
    "henon":        (HenonMap,
                     [("a", 1.4), ("b", 0.3)]),
    "logistic":     (CoupledLogisticMap,
                     [("r", 3.137), ("s", 0.85)]),
    "ikeda":        (IkedaMap,
                     [("u", 1.339)]),

    # new
    "dejong":       (DeJongDampedMap,
                     [("a", 2.01), ("b", -2.53),
                      ("c", 1.61), ("d", -0.33), ("rho", 0.98)]),
    "skew_tent":    (SkewTentMap,
                     [("s", 0.3)]),
    "lozi":         (LoziMap,
                     [("a", 1.5), ("b", 0.5)]),
    "cat":          (ArnoldCatMap,
                     []),
    "baker":        (BakersMap,
                     []),
    "standard":     (StandardMap,
                     [("K", 1.0)]),
    "sine":         (CoupledSineMap,
                     [("a", 1.9), ("b", 0.9),
                      ("c", 0.9), ("d", 1.9)]),
    "duffing":      (DuffingMap,
                     [("delta", 0.2), ("alpha", -1.0),
                      ("beta", 1.0), ("gamma", 0.3), ("omega", 1.0)]),
    "rab_fab":      (RabinovichFabrikantMap,
                     [("a", 0.87), ("beta", 0.87), ("gamma", 0.87)]),
}



# 4. ORBIT GENERATOR

def generate_orbit(system: DiscreteMap,
                   n_points: int = 500_000,
                   discard: int = 5_000):
    """Generate a long orbit for *system*."""
    xs = []
    x = system.random_seed()
    discard_left = discard

    while len(xs) < n_points:
        x = system.step(x)

        if (not np.all(np.isfinite(x))) or np.linalg.norm(x) > system.escape_radius:
            x = system.random_seed()
            continue

        if discard_left > 0:
            discard_left -= 1
            continue

        xs.append(x.copy())

    return np.array(xs, dtype=np.float32)



# 5. SRB DENSITY (2‑D histogram)

def density_coloring(points, bins: int = 800):
    """Return a density estimate for *points*."""
    x, y = points[:, 0], points[:, 1]
    H, xe, ye = np.histogram2d(x, y, bins=bins)
    ix = np.clip(np.searchsorted(xe, x) - 1, 0, bins - 1)
    iy = np.clip(np.searchsorted(ye, y) - 1, 0, bins - 1)
    d = H.T[iy, ix]
    d[d <= 0] = 1
    return d



# 6. PLOTTING

def plot_attractor(points, density, title):
    """Scatter plot coloured by SRB density (log scale)."""
    plt.figure(figsize=(8, 8))
    plt.scatter(points[:, 0], points[:, 1],
                c=density, s=0.15, cmap="inferno",
                norm=LogNorm(), linewidths=0)
    plt.axis("equal")
    plt.axis("off")
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_bifurcation(P, X, xlabel, title):
    """Simple bifurcation diagram (parameter vs. x‑coordinate)."""
    plt.figure(figsize=(10, 6))
    plt.scatter(P, X, s=0.1, color="black", alpha=0.6)
    plt.xlabel(xlabel)
    plt.ylabel("x")
    plt.title(title)
    plt.tight_layout()
    plt.show()



# 7. BIFURCATION ENGINE

def bifurcation_diagram(map_cls,
                        varying_name,
                        var_range,
                        fixed_params,
                        n_param=800,
                        discard=2000,
                        sample=300,
                        x_index=0):
    """
    Compute a bifurcation diagram for *map_cls* while varying a
    single parameter.

    Parameters
    ----------
    map_cls        : class of the map (subclass of DiscreteMap)
    varying_name   : string – name of the parameter to sweep
    var_range      : (p_min, p_max) – inclusive range for the sweep
    fixed_params   : dict – all other parameters fixed to these values
    n_param        : number of points along the parameter axis
    discard        : transient iterations to discard for each p
    sample         : number of points to record for each p
    x_index        : which component of the state vector to plot
    """
    p_vals = np.linspace(*var_range, n_param)
    P, X = [], []

    for p in p_vals:
        # build a fresh map instance with the current sweeping value
        kw = fixed_params.copy()
        kw[varying_name] = p
        system = map_cls(**kw)

        # start from a random seed
        x = system.random_seed()

        # discard transients
        for _ in range(discard):
            x = system.step(x)
            if not np.all(np.isfinite(x)):
                x = system.random_seed()

        # record samples
        for _ in range(sample):
            x = system.step(x)
            if not np.all(np.isfinite(x)):
                break
            P.append(p)
            X.append(x[x_index])

    return np.array(P), np.array(X)



# 8. USER INPUT HELPERS

def ask_int(prompt, default):
    while True:
        s = input(f"{prompt} [{default}]: ").strip()
        if s == "":
            return default
        try:
            v = int(s)
            if v <= 0:
                raise ValueError
            return v
        except ValueError:
            print("Please enter a positive integer.")


def ask_float(prompt, default):
    while True:
        s = input(f"{prompt} [{default}]: ").strip()
        if s == "":
            return default
        try:
            return float(s)
        except ValueError:
            print("Please enter a numeric value.")


def ask_range(prompt, default_min, default_max):
    """
    Prompt the user for a range as “min,max”.  If the user presses
    Enter, the default (default_min, default_max) is returned.
    """
    while True:
        s = input(f"{prompt} [{default_min},{default_max}]: ").strip()
        if s == "":
            return default_min, default_max
        parts = s.replace(" ", "").split(",")
        if len(parts) != 2:
            print("Enter two numbers separated by a comma.")
            continue
        try:
            lo, hi = float(parts[0]), float(parts[1])
            if lo >= hi:
                print("Left value must be smaller than right value.")
                continue
            return lo, hi
        except ValueError:
            print("Both values must be numeric.")



# 9. MAIN PROGRAM

def main():
    print("\n=== Strange‑Attractor + Bifurcation Explorer ===\n")

    
    # 1) Number of points for the attractor
    
    N = ask_int("Number of points to generate for attractor (N)", 500_000)

    
    # 2) Choose map
    
    print("\nAvailable maps:")
    map_names = list(MAPS.keys())
    for i, name in enumerate(map_names, 1):
        print(f"  {i}. {name}")

    while True:
        sel = input(f"\nSelect a map [1‑{len(map_names)}] (default 1): ").strip()
        if sel == "":
            idx = 0
            break
        try:
            idx = int(sel) - 1
            if 0 <= idx < len(map_names):
                break
            else:
                raise ValueError
        except ValueError:
            print("Invalid selection – try again.")

    chosen_name = map_names[idx]
    map_cls, param_spec = MAPS[chosen_name]

    
    # 3) Ask for each parameter (show defaults)
    
    print(f"\nEnter parameters for {chosen_name} (press Enter to keep defaults):")
    user_params = {}
    for pname, default in param_spec:
        val = ask_float(f"  {pname}", default)
        user_params[pname] = val

    
    # 4) Choose mode
    
    print("\nChoose mode:")
    print("  1. Attractor (SRB density)")
    print("  2. Bifurcation diagram (vary one parameter)")
    mode = input("Selection [1/2] (default 1): ").strip()
    if mode != "2":
        # ---------- ATTRACTION MODE ----------
        print("\nGenerating attractor …")
        system = map_cls(**user_params)
        points = generate_orbit(system, n_points=N, discard=5_000)
        density = density_coloring(points)
        plot_attractor(points, density, f"{chosen_name} attractor")
    else:
        # ---------- BIFURCATION MODE ----------
        if not param_spec:
            print("\n[ERROR] Selected map has no tunable parameters – cannot create a bifurcation diagram.")
            sys.exit(1)

        # List parameters again for the user to pick which to vary
        print("\nParameters:")
        for i, (pname, default) in enumerate(param_spec, 1):
            print(f"  {i}. {pname} (default = {default})")

        while True:
            p_sel = input(f"Select parameter to vary [1‑{len(param_spec)}] (default 1): ").strip()
            if p_sel == "":
                p_idx = 0
                break
            try:
                p_idx = int(p_sel) - 1
                if 0 <= p_idx < len(param_spec):
                    break
                else:
                    raise ValueError
            except ValueError:
                print("Invalid selection – try again.")

        var_name, var_default = param_spec[p_idx]

        # Build a dict of *fixed* parameters (all except the one we vary)
        fixed_params = {k: v for k, v in user_params.items() if k != var_name}

        # Ask for the range to sweep
        print(f"\nEnter sweep range for {var_name}.")
        # Use a modest default: 0.5·default … 1.5·default (or ±0.5 if default≈0)
        if var_default != 0:
            dmin = var_default * 0.5
            dmax = var_default * 1.5
        else:
            dmin, dmax = -1.0, 1.0
        var_min, var_max = ask_range(f"Range for {var_name} (min,max)", dmin, dmax)

        # Optional: how many parameter steps
        n_steps = ask_int("Number of parameter steps", 800)

        # Generate bifurcation data
        print("\nComputing bifurcation diagram … (this may take a minute…)")

        P, X = bifurcation_diagram(
            map_cls,
            varying_name=var_name,
            var_range=(var_min, var_max),
            fixed_params=fixed_params,
            n_param=n_steps,
            discard=2000,
            sample=300,
            x_index=0
        )

        plot_bifurcation(P, X, xlabel=var_name,
                         title=f"Bifurcation diagram of {chosen_name} ({var_name})")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
