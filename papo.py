
import sys
import math
import colorsys
import numpy as np
import pygame
import numba as nb                     # JIT compilation

# =================================================
# USER CONFIGURATION
# =================================================
WIDTH, HEIGHT   = 900, 700            # window size (pixel)
FPS             = 60
MAX_ITER        = 1200               # iterations for smooth colour
FADE_ALPHA      = 48                  # fade speed (larger → faster fade)

# -------------------------------------------------
# MAP / FAMILY DEFINITIONS (used only for UI)
# -------------------------------------------------
FAMILY_NAMES = ["quadratic", "cubic", "exponential", "rational"]
FAMILY_ID    = {name: i for i, name in enumerate(FAMILY_NAMES)}   # string → int id

# escape radius for each family (used in the smooth formula)
ESCAPE_RADIUS = {
    "quadratic":   2.0,
    "cubic":       2.0,
    "exponential":10.0,
    "rational":    2.0,
}

# -------------------------------------------------
# PALETTE – smooth HSV rainbow (256 colours)
# -------------------------------------------------
def make_palette(num=256):
    """Return a (num,3) uint8 array with a smooth HSV rainbow."""
    pal = np.empty((num, 3), dtype=np.uint8)
    for i in range(num):
        hue = i / num
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        pal[i] = (int(r * 255), int(g * 255), int(b * 255))
    return pal

PALETTE = make_palette(256)            # global – read‑only, Numba can use it

# =================================================
# NUMBA‑JIT core – fast pixel‑wise iteration
# =================================================
@nb.njit(parallel=True, fastmath=True)
def _iterate_fractal(width, height,
                    xs, ys,                     # 1‑D coordinate vectors (float32)
                    c_real, c_imag,             # constant parameter (Julia) or ignored (Mandelbrot)
                    max_iter, escape_r,
                    mandelbrot, family_id):
    """
    Compute the fractal image.
    Returns a uint8 array of shape (height, width, 3) with the RGB picture.
    """
    img = np.empty((height, width, 3), dtype=np.uint8)

    # pre‑compute log(escape_r) and log(2) because they are constant
    log_escape = np.log(escape_r)
    log2       = np.log(2.0)

    for py in nb.prange(height):
        y = ys[py]
        for px in range(width):
            x = xs[px]

            # ----- initialise z and constant c -----
            if mandelbrot:
                c = x + 1j * y          # c comes from the pixel
                z = 0.0 + 0.0j
            else:
                c = c_real + 1j * c_imag
                z = x + 1j * y

            # ----- iteration -----
            escaped = False
            it = max_iter
            for i in range(max_iter):
                # -----------------------------------------------------------------
                # family switch (minimal branches – Numba can optimise them away)
                # -----------------------------------------------------------------
                if family_id == 0:                      # quadratic  z = z*z + c
                    z = z * z + c
                elif family_id == 1:                    # cubic      z = z*z*z + c
                    z = z * z * z + c
                elif family_id == 2:                    # exponential z = exp(z) + c
                    z = np.exp(z) + c
                else:                                   # rational   z = 1/z + c
                    # guard against division by zero
                    if (z.real == 0.0) and (z.imag == 0.0):
                        zz = 1e-12 + 0j
                    else:
                        zz = z
                    z = 1.0 / zz + c

                # escape test (compare |z|² with escape_r² -> cheaper)
                if (z.real * z.real + z.imag * z.imag) > escape_r * escape_r:
                    it = i
                    escaped = True
                    break

            # ----- colour -----
            if not escaped:                     # interior → black
                img[py, px, 0] = 0
                img[py, px, 1] = 0
                img[py, px, 2] = 0
                continue

            # smooth (continuous) iteration count:
            #   ν = n + 1 - log₂(log|z|/log(escape_r))
            modulus = np.sqrt(z.real * z.real + z.imag * z.imag)
            if modulus == 0.0:
                modulus = 1e-12

            nu = it + 1.0 - (np.log(np.log(modulus) - log_escape)) / log2
            hue = (nu / max_iter) % 1.0
            idx = int(hue * 255.0) & 0xFF    # fast wrap to [0,255]

            img[py, px, 0] = PALETTE[idx, 0]
            img[py, px, 1] = PALETTE[idx, 1]
            img[py, px, 2] = PALETTE[idx, 2]

    return img

# =================================================
# PYGAME INITIALISATION
# =================================================
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Holomorphic‑Dynamics Explorer (Numba‑accelerated)")
clock = pygame.time.Clock()
fade_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

font = pygame.font.SysFont(None, 20)          # UI text
small_font = pygame.font.SysFont(None, 12)    # not used yet

# =================================================
# UI – two horizontal sliders (real & imag part of c)
# =================================================
class Slider:
    """Horizontal slider – knob draggable with the mouse."""
    def __init__(self, x, y, w, min_val, max_val, val):
        self.rect = pygame.Rect(x, y, w, 6)      # the track
        self.min = min_val
        self.max = max_val
        self.val = val
        self.dragging = False

    def handle(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            self.dragging = True
        if event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        if event.type == pygame.MOUSEMOTION and self.dragging:
            t = (event.pos[0] - self.rect.x) / self.rect.width
            self.val = self.min + np.clip(t, 0, 1) * (self.max - self.min)
            return True
        return False

    def draw(self):
        pygame.draw.rect(screen, (180, 180, 180), self.rect)
        knob_x = self.rect.x + (self.val - self.min) / (self.max - self.min) * self.rect.width
        pygame.draw.circle(screen, (255, 80, 80), (int(knob_x), self.rect.y + 3), 7)


# -------------------------------------------------
# Create the two sliders (Re(c), Im(c))
# -------------------------------------------------
BASE_SLIDER_Y = HEIGHT - 80
SLIDER_GAP   = 40
sliders = [
    Slider(50, BASE_SLIDER_Y,               300, -2.0, 2.0, 0.0),   # Re(c)
    Slider(50, BASE_SLIDER_Y + SLIDER_GAP,   300, -2.0, 2.0, 0.0)    # Im(c)
]
slider_names = ["Re(c)", "Im(c)"]

# =================================================
# GLOBAL STATE
# =================================================
family = "quadratic"                 # default family
family_id = FAMILY_ID[family]
escape_r = ESCAPE_RADIUS[family]

c_param = 0.0 + 0.0j                # complex constant (Julia mode)
mandelbrot_mode = False

fractal_surface = None               # pygame surface with the current picture
needs_recompute = True               # flag set when parameters change

# -------------------------------------------------
# High‑quality fractal computation wrapper
# -------------------------------------------------
def compute_fractal():
    """Re‑compute the whole picture and return a pygame.Surface."""
    global fractal_surface, width_array, height_array

    # --- 1‑D coordinate vectors (float32) ---------------------------------------
    xs = np.linspace(-2.0, 2.0, WIDTH,  dtype=np.float32)
    ys = np.linspace(-2.0, 2.0, HEIGHT, dtype=np.float32)

    # --- call the JIT‑compiled kernel -------------------------------------------
    img = _iterate_fractal(WIDTH, HEIGHT,
                           xs, ys,
                           float(c_param.real), float(c_param.imag),
                           MAX_ITER, float(escape_r),
                           mandelbrot_mode, family_id)

    # --- turn the NumPy RGB array into a pygame Surface -------------------------
    surf = pygame.surfarray.make_surface(img.swapaxes(0, 1))
    return surf

# =================================================
# MAIN LOOP
# =================================================
running = True
paused  = False

while running:
    clock.tick(FPS)

    # -------------------------------------------------
    # EVENT HANDLING
    # -------------------------------------------------
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            running = False

        if ev.type == pygame.KEYDOWN:
            # ----- quit / pause -------------------------------------------------
            if ev.key == pygame.K_ESCAPE:
                running = False
            if ev.key == pygame.K_SPACE:
                paused = not paused

            # ----- family selection --------------------------------------------
            if ev.key == pygame.K_1:
                family = "quadratic"
            if ev.key == pygame.K_2:
                family = "cubic"
            if ev.key == pygame.K_3:
                family = "exponential"
            if ev.key == pygame.K_4:
                family = "rational"

            if ev.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
                family_id = FAMILY_ID[family]
                escape_r  = ESCAPE_RADIUS[family]
                needs_recompute = True

            # ----- toggle Mandelbrot / Julia -------------------------------------
            if ev.key == pygame.K_5:
                mandelbrot_mode = not mandelbrot_mode
                needs_recompute = True

            # ----- save screenshot ------------------------------------------------
            if ev.key == pygame.K_s:
                fn = f"{family}_{'M' if mandelbrot_mode else 'J'}_" \
                     f"{c_param.real:.3f}_{c_param.imag:.3f}.png"
                pygame.image.save(screen, fn)
                print(f"[INFO] screenshot saved → {fn}")

            # ----- reset (clear everything) ---------------------------------------
            if ev.key == pygame.K_r:
                # reset sliders to zero, centre the view again
                sliders[0].val = 0.0
                sliders[1].val = 0.0
                c_param = 0.0 + 0.0j
                needs_recompute = True

        # ----- sliders (always active – they simply have no effect in Mandelbrot) -----
        for s in sliders:
            if s.handle(ev):
                needs_recompute = True

    # -------------------------------------------------
    # Update the complex constant `c` from the sliders (only in Julia mode)
    # -------------------------------------------------
    if not mandelbrot_mode:
        c_param = complex(sliders[0].val, sliders[1].val)

    # -------------------------------------------------
    # Re‑compute the fractal when needed (and not paused)
    # -------------------------------------------------
    if needs_recompute and not paused:
        print("[INFO] computing fractal …")
        fractal_surface = compute_fractal()
        needs_recompute = False
        print("[INFO] done.")

    # -------------------------------------------------
    # DRAW
    # -------------------------------------------------
    # fading surface – gives a tiny trailing effect when you move sliders
    fade_surface.fill((0, 0, 0, FADE_ALPHA))
    screen.blit(fade_surface, (0, 0))

    if fractal_surface:
        screen.blit(fractal_surface, (0, 0))

    # UI – sliders
    for s, name in zip(sliders, slider_names):
        s.draw()
        txt = f"{name} = {s.val:.3f}"
        screen.blit(font.render(txt, True, (200, 200, 200)),
                    (s.rect.x, s.rect.y - 25))

    # UI – mode / family info
    mode_txt = "Mandelbrot" if mandelbrot_mode else f"Julia (c = {c_param.real:.3f}+{c_param.imag:.3f}i)"
    screen.blit(font.render(f"Family: {family} | Mode: {mode_txt}",
                            True, (255, 255, 255)), (10, 10))

    # UI – help line
    help_msg = "1‑4: family | 5: toggle Mandelbrot/Julia | SPACE: pause | R: reset | S: screenshot"
    screen.blit(font.render(help_msg, True, (180, 180, 180)),
                (10, HEIGHT - 30))

    pygame.display.flip()

# -------------------------------------------------
# CLEAN‑UP
# -------------------------------------------------
pygame.quit()
sys.exit()
