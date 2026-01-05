import sys, math
import numpy as np
import pygame


# CONFIGURATION

WIDTH, HEIGHT = 900, 700          # window size
FPS           = 60
N_POINTS      = 6000
fade_alpha    = 8                # lower → longer trails


# MAP PARAMETERS (default values for each map)

params = {
    "henon":    {"a": 1.4, "b": 0.3},
    "ikeda":    {"u": 0.918},
    "standard": {"K": 1.0},
    "logistic": {"r": 3.7, "s": 0.0},

    # NEW MAPS 
    "lozi":     {"a": 1.7, "b": 0.5},
    "dejong":   {"a": 1.4, "b": -2.3, "c": 2.4, "d": -2.1},
    "damped":   {"K": 1.2, "gamma": 0.95},
    "skew":     {"a": -0.4, "lam": 0.6},
    "duffing":  {"delta": 0.2, "alpha": -1.0, "beta": 1.0,
                "gamma": 0.3, "omega": 1.0},
}
current_map = "henon"


# TOGGLES (extra visual diagnostics – currently off)

SHOW_MANIFOLDS = False
SHOW_CONEFIELD = False
SHOW_DENSITY   = False
SHOW_POINCARE  = False
SHOW_TIMEAVG   = False
SHOW_SYMBOLIC  = False


# PYGAME INITIALISATION

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Dynamical Systems Lab")
clock = pygame.time.Clock()
fade_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

font = pygame.font.SysFont(None, 20)
symbol_font = pygame.font.SysFont(None, 12)


# SLIDER UI (horizontal sliders)

class Slider:
    """Horizontal slider – knob draggable with the mouse."""
    def __init__(self, x, y, w, min_val, max_val, val):
        self.rect = pygame.Rect(x, y, w, 6)      # track
        self.min = min_val
        self.max = max_val
        self.val = val
        self.dragging = False

    def handle(self, e):
        if e.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(e.pos):
            self.dragging = True
        if e.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        if e.type == pygame.MOUSEMOTION and self.dragging:
            t = (e.pos[0] - self.rect.x) / self.rect.width
            self.val = self.min + np.clip(t, 0, 1) * (self.max - self.min)
            return True
        return False

    def draw(self):
        pygame.draw.rect(screen, (180, 180, 180), self.rect)
        knob = self.rect.x + (self.val - self.min) / (self.max - self.min) * self.rect.width
        pygame.draw.circle(screen, (255, 80, 80), (int(knob), self.rect.y + 3), 7)



# SLIDERS (auto‑fit to each map)

BASE_Y = HEIGHT - 140
DY = 28

sliders = {
    "henon": [
        Slider(40, BASE_Y,     260, 0.0, 2.5, params["henon"]["a"]),
        Slider(40, BASE_Y + DY,260, 0.0, 1.0, params["henon"]["b"]),
    ],
    "ikeda": [
        Slider(40, BASE_Y,     260, 0.5, 1.5, params["ikeda"]["u"]),
    ],
    "standard": [
        Slider(40, BASE_Y,     260, 0.0, 5.0, params["standard"]["K"]),
    ],
    "logistic": [
        Slider(40, BASE_Y,     260, 0.0, 5.0, params["logistic"]["r"]),
        Slider(40, BASE_Y + DY,260,-1.0, 1.0, params["logistic"]["s"]),
    ],
    "lozi": [
        Slider(40, BASE_Y,     260, 0.0, 2.5, params["lozi"]["a"]),
        Slider(40, BASE_Y + DY,260, 0.0, 1.0, params["lozi"]["b"]),
    ],
    "dejong": [
        Slider(40, BASE_Y + i*DY, 260, -3.0, 3.0, params["dejong"][k])
        for i, k in enumerate(("a","b","c","d"))
    ],
    "damped": [
        Slider(40, BASE_Y,     260, 0.0, 5.0, params["damped"]["K"]),
        Slider(40, BASE_Y + DY,260, 0.7, 1.0, params["damped"]["gamma"]),
    ],
    "skew": [
        Slider(40, BASE_Y,     260, -1.5, 1.5, params["skew"]["a"]),
        Slider(40, BASE_Y + DY,260, 0.0, 1.2, params["skew"]["lam"]),
    ],
    # Duffing map needs five sliders (δ, α, β, γ, ω)
    "duffing": [
        Slider(40, BASE_Y + i*DY, 260, 0.0, 1.0, params["duffing"][k])
        for i, k in enumerate(("delta","alpha","beta","gamma","omega"))
    ],
}
# Mapping from map name → ordered list of its parameter keys (for label printing)
slider_names = {k: list(params[k].keys()) for k in sliders}


# UTILITIES (coordinate conversion, drawing, reseeding)

def to_screen(p):
    """Map a point in logical window [-2,2]² → pixel coordinates."""
    if not np.all(np.isfinite(p)):
        return None
    return (
        int((p[0] + 2.0) / 4.0 * WIDTH),
        int((p[1] + 2.0) / 4.0 * HEIGHT)
    )

def draw_point(pos, col, r=1):
    s = pygame.Surface((2 * r, 2 * r), pygame.SRCALPHA)
    pygame.draw.circle(s, (*col, 150), (r, r), r)
    screen.blit(s, (pos[0] - r, pos[1] - r))

def reseed_point(_):
    """Return a fresh random point inside the view rectangle."""
    return np.random.uniform(-2, 2, 2)


# MAP FUNCTIONS (read parameters from the global `params` dict)

def henon(p):
    a, b = params["henon"].values()
    return np.array([1.0 - a * p[0] ** 2 + p[1], b * p[0]])

def ikeda(p):
    x, y = p
    u = params["ikeda"]["u"]
    t = 0.4 - 6.0 / (1.0 + x * x + y * y)
    c, s = math.cos(t), math.sin(t)
    return np.array([1.0 + u * (x * c - y * s),
                    u * (x * s + y * c)])

def standard(p):
    x, m = p
    K = params["standard"]["K"]
    m = (m + K * math.sin(x)) % (2.0 * math.pi)
    x = (x + m) % (2.0 * math.pi)
    # centre the torus in the [-2,2] window
    return np.array([x - math.pi, m - math.pi])

def logistic(p):
    r, s = params["logistic"].values()
    return np.array([r * p[0] * (1.0 - p[0]) + s * p[1],
                    r * p[1] * (1.0 - p[1]) + s * p[0]])

def lozi(p):
    a, b = params["lozi"].values()
    return np.array([1.0 - a * abs(p[0]) + p[1],
                    b * p[0]])

def dejong(p):
    a, b, c, d = params["dejong"].values()
    x, y = p
    return np.array([math.sin(a * y) - math.cos(b * x),
                    math.sin(c * x) - math.cos(d * y)])

def damped(p):
    x, m = p
    K, gamma = params["damped"].values()
    m = gamma * m + K * math.sin(x)
    x = x + m
    return np.array([ (x % (2.0 * math.pi)) - math.pi,
                      (m % (2.0 * math.pi)) - math.pi ])

def skew(p):
    a, lam = params["skew"].values()
    x, y = p
    return np.array([x * x - y * y + a,
                    lam * y + 2.0 * x * y])

def duffing(p):
    """
    Discrete‑time Duffing map (sampled once per forcing period).
    Parameters are taken from ``params["duffing"]``:
        delta – damping,
        alpha – linear stiffness,
        beta  – cubic stiffness,
        gamma – forcing amplitude,
        omega – forcing frequency (phase increment).
    """
    δ   = params["duffing"]["delta"]
    α   = params["duffing"]["alpha"]
    β   = params["duffing"]["beta"]
    γ   = params["duffing"]["gamma"]
    ω   = params["duffing"]["omega"]

    x, y = p                     # x = position, y = velocity (or momentum)
    # one step of the Duffing oscillator (Euler‑type integration)
    x_next = y
    y_next = (-δ * y
              - α * x
              - β * (x ** 3)
              + γ * math.cos(ω * duffing.phase))   # use a global phase
    duffing.phase = (duffing.phase + ω) % (2.0 * math.pi)
    return np.array([x_next, y_next])

# initialise the phase for the cosine forcing (global variable on the function)
duffing.phase = 0.0

# Dictionary that maps a map name → its function
maps = {
    "henon":    henon,
    "ikeda":    ikeda,
    "standard": standard,
    "logistic": logistic,
    "lozi":     lozi,
    "dejong":   dejong,
    "damped":   damped,
    "skew":     skew,
    "duffing":  duffing
}


# POINT CLOUD (initial random seeds)

points = np.random.uniform(-1, 1, (N_POINTS, 2))
time_avg = np.zeros(N_POINTS)
n_steps = np.zeros(N_POINTS, int)
symbolic = np.full(N_POINTS, "", object)


# MAIN LOOP

running = True
paused  = False

while running:
    clock.tick(FPS)

   
    # EVENT PROCESSING
   
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False

        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_ESCAPE:
                running = False
            if e.key == pygame.K_SPACE:
                paused = not paused

           
            # Map selection – keys 1‑9 (now includes Duffing on 9)
           
            if e.key in [pygame.K_1, pygame.K_2, pygame.K_3,
                         pygame.K_4, pygame.K_5, pygame.K_6,
                         pygame.K_7, pygame.K_8, pygame.K_9]:
                current_map = list(maps)[e.key - pygame.K_1]

           
            # Reset (clears the point cloud and restores defaults)
           
            if e.key == pygame.K_r:
                points = np.random.uniform(-1, 1, (N_POINTS, 2))
                time_avg.fill(0)
                n_steps.fill(0)
                symbolic[:] = ""
                # reset sliders to the default values stored in `params`
                for name, slist in sliders.items():
                    for i, s in enumerate(slist):
                        keys = slider_names[name]
                        if keys:                       # map has parameters
                            s.val = params[name][keys[i]]

       
        # Forward mouse events to the sliders of the active map
       
        for s in sliders[current_map]:
            s.handle(e)

   
    # UPDATE PARAMETERS from the sliders (only for the active map)
   
    for i, key in enumerate(slider_names[current_map]):
        params[current_map][key] = sliders[current_map][i].val

   
    # FADE (creates the trailing effect)
   
    fade_surface.fill((0, 0, 0, fade_alpha))
    screen.blit(fade_surface, (0, 0))

   
    # ITERATION & DRAWING (skip when paused)
   
    if not paused:
        F = maps[current_map]          # grab the function for the chosen map
        new_points = []

        for i, p in enumerate(points):
            p2 = F(p)

            # reseed points that escape or become non‑finite
            if (not np.all(np.isfinite(p2))) or np.linalg.norm(p2) > 10.0:
                p2 = reseed_point(current_map)

            scr = to_screen(p2)
            if scr:
                draw_point(scr, (220, 220, 220))

            new_points.append(p2)

        points = np.array(new_points)

   
    # UI – draw sliders with labels for the active map
   
    for s in sliders[current_map]:
        s.draw()
        i = sliders[current_map].index(s)           # position in the list
        if slider_names[current_map]:               # map has parameters
            param_key = slider_names[current_map][i]
            label = f"{param_key} = {s.val:.3f}"
        else:
            label = ""                              # no sliders for this map
        screen.blit(font.render(label, True, (200, 200, 200)),
                    (s.rect.x, s.rect.y - 18))

   
    # UI – map name in the corner
   
    screen.blit(font.render(f"Map: {current_map}", True, (255, 255, 255)),
                (10, 10))

    pygame.display.flip()



pygame.quit()
sys.exit()
