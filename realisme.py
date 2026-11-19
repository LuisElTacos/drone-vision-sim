"""Rendu « réaliste » optionnel : ombres portées, occlusion ambiante (SSAO) et effets de caméra.

Par défaut toutes les options sont actives sauf la distorsion (DEFAUT), qui casserait l'hypothèse de paires
rectifiées de la vision. REALISME=aucun (ou --realisme aucun) redonne exactement l'image habituelle.

Options (Realisme.options, cochables dans le panneau, ou --realisme / variable REALISME) :
  ombres      ombres portées du soleil (shadow mapping, carte de profondeur vue du soleil)
  ssao        occlusion ambiante en espace écran
  bruit       bruit capteur : bruit de lecture + bruit photonique (∝ √signal), amplifié par le gain
  flou        flou de mouvement : chaque pixel est moyenné le long de son déplacement à l'écran pendant
              le temps de pose (reprojection de la profondeur avec la matrice vue-projection précédente)
  expo        exposition automatique : gain adapté à la luminance moyenne (constante de temps ~0,4 s) ;
              en scène sombre le temps de pose s'allonge (plus de flou) puis le gain monte (plus de bruit)
  distorsion  distorsion radiale de Brown-Conrady (k1, k2), sur les coordonnées normalisées x/f, y/f
              de la caméra idéale : K = camera.intrinsics() reste valable avec dist = (k1, k2, 0, 0)
  jpeg        compression JPEG (cv2, qualité réglable)

Les ombres et le SSAO s'appliquent à toutes les vues ; les effets de caméra seulement aux caméras du drone
(instances de scene.Camera). La profondeur écrite dans le framebuffer de destination suit la distorsion,
pour que la vérité terrain (StereoCapture.depths) reste alignée avec l'image.

Le flou de mouvement a besoin de l'intervalle entre deux images : Realisme.time (temps de simulation, en s)
s'il est renseigné, sinon l'horloge murale.
"""

import math
import os
import time

import moderngl
import numpy as np

OPTIONS = ("ombres", "ssao", "bruit", "flou", "expo", "distorsion", "jpeg")
DEFAUT = ("ombres", "ssao", "bruit", "flou", "expo", "jpeg")   # options actives sans REALISME
CAMERA_FX = ("bruit", "flou", "expo", "distorsion", "jpeg")
SKY = (0.55, 0.7, 0.9)
SHADOW_UNIT = 7            # unité de texture réservée à la carte d'ombre

# Morceau GLSL inséré dans les fragment shaders des environnements (après « out vec4 f_color; »)
SHADOW_GLSL = """
uniform sampler2DShadow u_shadow_map;
uniform mat4 u_light_vp;
uniform float u_shadow_on;      // 0 : pas d'ombre ; sinon intensité de l'ombre (0..1)
uniform float u_shadow_texel;   // taille d'un texel de la carte d'ombre (m)
uniform float u_shadow_bias;    // biais de profondeur (unités de profondeur normalisée)
float shadow_factor(vec3 world, vec3 n, vec3 l) {
    if (u_shadow_on <= 0.0) return 1.0;
    float ndl = dot(n, l);
    if (ndl <= 0.0) return 1.0;   // face non éclairée : la diffusion est déjà nulle
    vec3 p = world + n * u_shadow_texel * (1.0 + 2.0 * (1.0 - ndl));   // décalage le long de la normale
    vec4 q = u_light_vp * vec4(p, 1.0);
    vec3 s = q.xyz / q.w * 0.5 + 0.5;
    if (s.x < 0.0 || s.x > 1.0 || s.y < 0.0 || s.y > 1.0 || s.z > 1.0) return 1.0;
    vec2 ts = 1.0 / vec2(textureSize(u_shadow_map, 0));
    float lit = 0.0;
    for (int i = -1; i <= 1; i++)
        for (int j = -1; j <= 1; j++)
            lit += texture(u_shadow_map, vec3(s.xy + vec2(i, j) * ts, s.z - u_shadow_bias));
    lit /= 9.0;
    return mix(1.0, lit, u_shadow_on);
}
"""


def inject_shadow(fragment_shader):
    """Ajoute shadow_factor() à un fragment shader (juste après la déclaration de f_color)."""
    marker = "out vec4 f_color;"
    assert marker in fragment_shader
    return fragment_shader.replace(marker, marker + "\n" + SHADOW_GLSL, 1)


# ---------------------------------------------------------------------------
# Shaders de post-traitement (triangle plein écran)
# ---------------------------------------------------------------------------

FS_VERTEX = """
#version 330
out vec2 v_uv;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    v_uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

SSAO_FRAGMENT = """
#version 330
uniform sampler2D u_depth;
uniform mat4 u_proj;
uniform mat4 u_inv_proj;
uniform vec3 u_kernel[16];
uniform float u_radius;
in vec2 v_uv;
out float f_ao;

vec3 view_pos(vec2 uv) {
    float d = texture(u_depth, uv).r;
    vec4 p = u_inv_proj * vec4(uv * 2.0 - 1.0, d * 2.0 - 1.0, 1.0);
    return p.xyz / p.w;
}

void main() {
    float d = texture(u_depth, v_uv).r;
    if (d >= 1.0) { f_ao = 1.0; return; }
    vec2 px = 1.0 / vec2(textureSize(u_depth, 0));
    vec3 p = view_pos(v_uv);
    // normale reconstruite à partir des voisins, du côté le plus proche (évite les bords d'objets)
    vec3 pr = view_pos(v_uv + vec2(px.x, 0.0)), pl = view_pos(v_uv - vec2(px.x, 0.0));
    vec3 pu = view_pos(v_uv + vec2(0.0, px.y)), pd = view_pos(v_uv - vec2(0.0, px.y));
    vec3 dx = abs(pr.z - p.z) < abs(p.z - pl.z) ? pr - p : p - pl;
    vec3 dy = abs(pu.z - p.z) < abs(p.z - pd.z) ? pu - p : p - pd;
    vec3 n = normalize(cross(dx, dy));
    // rotation aléatoire du noyau, motif 4 x 4 (effacé ensuite par un flou 4 x 4)
    ivec2 ip = ivec2(gl_FragCoord.xy) & 3;
    float a = 6.2831853 * float(ip.x * 4 + ip.y) / 16.0 + 0.37;
    vec3 rv = vec3(cos(a), sin(a), 0.0);
    vec3 t = normalize(rv - n * dot(rv, n));
    if (any(isnan(t))) t = normalize(cross(n, vec3(0.0, 1.0, 0.0)));
    mat3 tbn = mat3(t, cross(n, t), n);
    float radius = u_radius;
    float occ = 0.0;
    for (int i = 0; i < 16; i++) {
        vec3 s = p + tbn * u_kernel[i] * radius;
        vec4 q = u_proj * vec4(s, 1.0);
        vec2 uv = q.xy / q.w * 0.5 + 0.5;
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) continue;
        float sz = view_pos(uv).z;
        float range = smoothstep(0.0, 1.0, radius / max(abs(p.z - sz), 1e-4));
        occ += (sz >= s.z + 0.02 * radius ? 1.0 : 0.0) * range;
    }
    f_ao = 1.0 - occ / 16.0;
}
"""

LIT_FRAGMENT = """
#version 330
uniform sampler2D u_color;
uniform sampler2D u_ao;
uniform bool u_use_ao;
uniform float u_ao_strength;
in vec2 v_uv;
out vec4 f_color;
void main() {
    vec3 c = texture(u_color, v_uv).rgb;
    if (u_use_ao) {
        vec2 px = 1.0 / vec2(textureSize(u_ao, 0));
        float ao = 0.0;
        for (int i = -2; i < 2; i++)
            for (int j = -2; j < 2; j++)
                ao += texture(u_ao, v_uv + (vec2(i, j) + 0.5) * px).r;
        ao /= 16.0;
        c *= pow(ao, 2.0 * u_ao_strength);
    }
    f_color = vec4(c, 1.0);
}
"""

COMPOSITE_FRAGMENT = """
#version 330
uniform sampler2D u_lit;
uniform sampler2D u_depth;
uniform bool u_distort;
uniform vec2 u_k;            // k1, k2
uniform vec2 u_tan_half;     // tan(fov/2) horizontal et vertical de la caméra idéale
uniform float u_overscan;    // rapport champ rendu / champ idéal (en tangente)
uniform bool u_blur;
uniform mat4 u_inv_vp;       // vue-projection courante (rendu interne), inversée
uniform mat4 u_prev_vp;      // vue-projection de l'image précédente
uniform float u_blur_scale;  // temps de pose / intervalle entre images
uniform float u_gain;        // exposition
uniform bool u_noise;
uniform float u_noise_level;
uniform float u_iso;         // gain analogique (amplifie le bruit)
uniform float u_frame;
uniform int u_ss;            // sous-échantillons par axe et par pixel de sortie
in vec2 v_uv;
out vec4 f_color;

float hash13(vec3 p3) {
    p3 = fract(p3 * 0.1031);
    p3 += dot(p3, p3.zyx + 31.32);
    return fract((p3.x + p3.y) * p3.z);
}
float gauss(vec3 seed) {
    float u1 = max(hash13(seed), 1e-6), u2 = hash13(seed + 17.13);
    return sqrt(-2.0 * log(u1)) * cos(6.2831853 * u2);
}

vec2 undistort(vec2 uv_out) {
    // pixel de sortie (distordu) -> point de l'image idéale (non distordue)
    if (!u_distort) return uv_out;
    vec2 xd = (uv_out * 2.0 - 1.0) * u_tan_half;   // coordonnées normalisées x/f, y/f
    float rd = length(xd);
    float ru = rd;
    for (int i = 0; i < 8; i++) {
        float r2 = ru * ru;
        ru = rd / (1.0 + u_k.x * r2 + u_k.y * r2 * r2);
    }
    vec2 xu = rd > 1e-6 ? xd * (ru / rd) : xd;
    return xu / u_tan_half * 0.5 + 0.5;
}

vec3 sample_blur(vec2 uvi, float d, float jitter) {
    // flou de mouvement le long du déplacement à l'écran pendant la pose
    if (!u_blur) return texture(u_lit, uvi).rgb;
    vec4 w = u_inv_vp * vec4(uvi * 2.0 - 1.0, d * 2.0 - 1.0, 1.0);
    w /= w.w;
    vec4 q = u_prev_vp * w;
    vec2 prev = q.xy / q.w * 0.5 + 0.5;
    vec2 vel = (uvi - prev) * u_blur_scale;
    if (q.w <= 0.0) vel = vec2(0.0);
    float len = length(vel);
    if (len > 0.08) vel *= 0.08 / len;               // borne (téléportation, saut d'image)
    const int N = 12;
    vec3 c = vec3(0.0);
    for (int i = 0; i < N; i++) {
        float t = (float(i) + 0.5 + jitter) / float(N) - 0.5;
        c += texture(u_lit, clamp(uvi - vel * t, 0.0, 1.0)).rgb;
    }
    return c / float(N);
}

void main() {
    vec2 px = 1.0 / vec2(textureSize(u_lit, 0)) * float(u_ss) / u_overscan;   // pixel de sortie (uv)
    float jitter = hash13(vec3(gl_FragCoord.xy, u_frame)) - 0.5;
    vec3 c = vec3(0.0);
    float depth = 1.0;
    int n_in = 0;
    for (int a = 0; a < u_ss; a++)
        for (int b = 0; b < u_ss; b++) {
            vec2 o = (vec2(a, b) + 0.5) / float(u_ss) - 0.5;
            vec2 uv = undistort(v_uv + o * px);
            vec2 uvi = 0.5 + (uv - 0.5) / u_overscan;   // dans le rendu interne (champ élargi)
            if (uvi.x < 0.0 || uvi.x > 1.0 || uvi.y < 0.0 || uvi.y > 1.0) continue;
            float d = texture(u_depth, uvi).r;
            if (a == u_ss / 2 && b == u_ss / 2) depth = d;
            c += sample_blur(uvi, d, jitter);
            n_in++;
        }
    c = n_in > 0 ? c / float(n_in) : vec3(0.0);
    gl_FragDepth = depth;

    // 3. exposition
    c *= u_gain;

    // 4. bruit capteur : lecture (constant) + photonique (∝ √signal), amplifiés par le gain analogique
    if (u_noise) {
        vec3 s = clamp(c, 0.0, 1.0);
        vec3 sigma = u_noise_level * sqrt(vec3(0.004 * 0.004) * u_iso * u_iso + 0.0006 * s * u_iso);
        vec3 seed = vec3(gl_FragCoord.xy, u_frame * 7.31);
        c += sigma * vec3(gauss(seed), gauss(seed + 3.7), gauss(seed + 8.1));
    }
    f_color = vec4(clamp(c, 0.0, 1.0), 1.0);
}
"""

COPY_FRAGMENT = """
#version 330
uniform sampler2D u_tex;
uniform sampler2D u_depth;
in vec2 v_uv;
out vec4 f_color;
void main() {
    f_color = vec4(texture(u_tex, v_uv).rgb, 1.0);
    gl_FragDepth = texture(u_depth, v_uv).r;
}
"""


# ---------------------------------------------------------------------------
# Matrices
# ---------------------------------------------------------------------------

def _look_at(eye, target, up):
    f = target - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.identity(4)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def _ortho(half, near, far):
    m = np.identity(4)
    m[0, 0] = m[1, 1] = 1.0 / half
    m[2, 2] = -2.0 / (far - near)
    m[2, 3] = -(far + near) / (far - near)
    return m


def _perspective(fovy_deg, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fovy_deg) / 2.0)
    m = np.zeros((4, 4))
    m[0, 0], m[1, 1] = f / aspect, f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2.0 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def _gl(m):
    return np.asarray(m, dtype="f4").T.copy().tobytes()


class _LightCamera:
    """Caméra orthographique placée dans la direction du soleil (même interface que scene.Camera)."""

    def __init__(self, view, proj, position):
        self._view, self._proj = view.astype("f4"), proj.astype("f4")
        self.position = np.asarray(position, dtype="f4")

    def view(self):
        return self._view

    def projection(self, aspect):
        return self._proj


class _WideCamera:
    """Caméra du drone vue avec un champ élargi (marge pour la distorsion en barillet)."""

    def __init__(self, cam, overscan):
        self._cam = cam
        self.fovy = math.degrees(2.0 * math.atan(math.tan(math.radians(cam.fovy) / 2.0) * overscan))

    def __getattr__(self, name):
        return getattr(self._cam, name)

    def projection(self, aspect):
        return _perspective(self.fovy, aspect, self._cam.near, 500.0).astype("f4")


class _Buffers:
    """Cibles intermédiaires pour une taille de vue donnée."""

    def __init__(self, ctx, w, h):
        self.size = (w, h)
        self.depth = ctx.depth_texture((w, h))
        self.depth.compare_func = ""
        self.depth.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.color = ctx.texture((w, h), 4, dtype="f2")
        self.color.repeat_x = self.color.repeat_y = False
        self.scene_fbo = ctx.framebuffer([self.color], self.depth)
        self.ao = ctx.texture((w, h), 1, dtype="f2")
        self.ao.repeat_x = self.ao.repeat_y = False
        self.ao_fbo = ctx.framebuffer([self.ao])
        self.lit = ctx.texture((w, h), 4, dtype="f2")
        self.lit.repeat_x = self.lit.repeat_y = False
        self.lit_fbo = ctx.framebuffer([self.lit])
        self.levels = int(math.floor(math.log2(max(w, h))))

    def release(self):
        for o in (self.scene_fbo, self.ao_fbo, self.lit_fbo, self.depth, self.color, self.ao, self.lit):
            o.release()


class _OutBuffers:
    """Sortie 8 bits (avant compression JPEG) à la taille de la vue de destination."""

    def __init__(self, ctx, w, h):
        self.size = (w, h)
        self.color = ctx.texture((w, h), 3)
        self.depth = ctx.depth_texture((w, h))
        self.depth.compare_func = ""
        self.depth.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.fbo = ctx.framebuffer([self.color], self.depth)
        self.jpeg = ctx.texture((w, h), 3)
        self.jpeg.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.color.filter = (moderngl.NEAREST, moderngl.NEAREST)

    def release(self):
        for o in (self.fbo, self.color, self.depth, self.jpeg):
            o.release()


class _CamState:
    __slots__ = ("vp", "t", "prev_vp", "blur_scale", "gain", "frame")

    def __init__(self):
        self.vp = self.t = self.prev_vp = None
        self.blur_scale = 0.0
        self.gain = None
        self.frame = 0


def parse_options(text):
    """« ombres,ssao » -> {"ombres", "ssao"} ; « tout » -> toutes les options."""
    text = (text or "").strip().lower()
    if not text or text in ("aucun", "non", "0"):
        return set()
    if text in ("tout", "all", "1"):
        return set(OPTIONS)
    if text in ("defaut", "défaut"):
        return set(DEFAUT)
    out = {t.strip() for t in text.split(",") if t.strip()}
    bad = out - set(OPTIONS)
    if bad:
        raise ValueError(f"options inconnues : {sorted(bad)} (possibles : {', '.join(OPTIONS)}, tout)")
    return out


class Realisme:
    def __init__(self, ctx):
        self.ctx = ctx
        env = os.environ.get("REALISME")
        on = set(DEFAUT) if env is None else parse_options(env)
        self.options = {k: k in on for k in OPTIONS}
        self.params = {
            "ombre_intensite": 0.85,  # part de la lumière directe retirée à l'ombre
            "ombre_portee": 40.0,     # demi-côté de la zone couverte par la carte d'ombre (m)
            "ombre_resolution": 2048,
            "ssao_rayon": 0.6,        # m
            "ssao_intensite": 1.0,
            "pose_ms": 8.0,           # temps de pose de base (ms)
            "pose_max_ms": 25.0,      # temps de pose maximal en exposition auto (ms)
            "bruit_niveau": 1.0,
            "expo_cible": 0.42,       # luminance moyenne visée
            "expo_tau": 0.4,          # constante de temps d'adaptation (s)
            "k1": -0.20,              # distorsion radiale (barillet si < 0)
            "k2": 0.04,
            "jpeg_qualite": 60,
        }
        self.time = None              # temps de simulation (s) ; None : horloge murale
        self._progs = {}
        self._shadow_cache = {}       # clé -> (texture, fbo, light_vp, texel)
        self._dummy_shadow = ctx.depth_texture((1, 1))
        self._buffers = {}
        self._out = {}
        self._cams = {}
        fs = FS_VERTEX
        self.ssao_prog = ctx.program(vertex_shader=fs, fragment_shader=SSAO_FRAGMENT)
        self.lit_prog = ctx.program(vertex_shader=fs, fragment_shader=LIT_FRAGMENT)
        self.comp_prog = ctx.program(vertex_shader=fs, fragment_shader=COMPOSITE_FRAGMENT)
        self.copy_prog = ctx.program(vertex_shader=fs, fragment_shader=COPY_FRAGMENT)
        self._vaos = {p: ctx.vertex_array(p, []) for p in
                      (self.ssao_prog, self.lit_prog, self.comp_prog, self.copy_prog)}
        rng = np.random.default_rng(3)
        k = []
        while len(k) < 16:   # noyau hémisphérique, plus dense près du point
            v = rng.uniform((-1, -1, 0), (1, 1, 1))
            if 0.05 < np.linalg.norm(v) <= 1.0:
                v = v / np.linalg.norm(v) * rng.uniform(0, 1)
                s = len(k) / 16.0
                k.append(v * (0.1 + 0.9 * s * s))
        self.ssao_prog["u_kernel"].write(np.array(k, dtype="f4").tobytes())

    # --- programmes des environnements --------------------------------------

    def register(self, prog):
        """Programme dont le fragment shader contient SHADOW_GLSL (à appeler une fois par programme)."""
        if prog is None or id(prog) in self._progs:
            return
        self._progs[id(prog)] = prog
        u = prog.get("u_shadow_map", None)
        if u is not None:
            u.value = SHADOW_UNIT
        self._set(prog, "u_shadow_on", 0.0)

    @staticmethod
    def _set(prog, name, value):
        u = prog.get(name, None)
        if u is None:
            return
        if isinstance(value, bytes):
            u.write(value)
        else:
            u.value = value

    def wants_post(self, camera, is_drone_cam):
        if self.options["ssao"]:
            return True
        return is_drone_cam and any(self.options[k] for k in CAMERA_FX)

    # --- ombres ---------------------------------------------------------------

    def _shadow_off(self):
        self._dummy_shadow.use(location=SHADOW_UNIT)
        for prog in self._progs.values():
            self._set(prog, "u_shadow_on", 0.0)

    def prepare_shadows(self, key_env, camera, light_dir, draw_casters):
        """Prépare (ou reprend du cache) la carte d'ombre autour de la caméra, puis règle les uniformes."""
        ctx = self.ctx
        self._shadow_off()
        if not self.options["ombres"]:
            return
        half = float(self.params["ombre_portee"])
        res = int(self.params["ombre_resolution"])
        step = half / 8.0
        pos = np.asarray(camera.position, dtype=float)
        center = np.round(pos / step) * step
        center[1] = 0.0
        key = (key_env, tuple(center), half, res)
        entry = self._shadow_cache.get(key)
        if entry is None:
            if len(self._shadow_cache) >= 4:   # 4 cartes au plus (vues drone, monde, suivi)
                old = next(iter(self._shadow_cache))
                tex, fbo = self._shadow_cache.pop(old)[:2]
                fbo.release()
                tex.release()
            L = np.asarray(light_dir, dtype=float)
            L = L / np.linalg.norm(L)
            dist, near, far = 150.0, 1.0, 300.0
            eye = center - L * dist
            up = np.array([0.0, 0.0, 1.0]) if abs(L[1]) > 0.99 else np.array([0.0, 1.0, 0.0])
            view, proj = _look_at(eye, center, up), _ortho(half, near, far)
            tex = ctx.depth_texture((res, res))
            tex.compare_func = "<="
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            tex.repeat_x = tex.repeat_y = False
            fbo = ctx.framebuffer(depth_attachment=tex)
            prev_fbo, prev_vp = ctx.fbo, ctx.viewport
            fbo.use()
            ctx.viewport = (0, 0, res, res)
            fbo.clear(depth=1.0)
            ctx.enable(moderngl.DEPTH_TEST)
            draw_casters(_LightCamera(view, proj, eye))
            prev_fbo.use()
            ctx.viewport = prev_vp
            entry = (tex, fbo, proj @ view, 2.0 * half / res, 0.03 / (far - near))
            self._shadow_cache[key] = entry
        tex, _, light_vp, texel, bias = entry
        tex.use(location=SHADOW_UNIT)
        for prog in self._progs.values():
            self._set(prog, "u_light_vp", _gl(light_vp))
            self._set(prog, "u_shadow_texel", float(texel))
            self._set(prog, "u_shadow_bias", float(bias))
            self._set(prog, "u_shadow_on", float(self.params["ombre_intensite"]))

    def clear_shadow_cache(self):
        for tex, fbo, *_ in self._shadow_cache.values():
            fbo.release()
            tex.release()
        self._shadow_cache.clear()

    # --- post-traitement --------------------------------------------------------

    def _get(self, cache, cls, w, h):
        b = cache.get((w, h))
        if b is None:
            if len(cache) >= 6:
                cache.pop(next(iter(cache))).release()
            b = cache[(w, h)] = cls(self.ctx, w, h)
        return b

    def _now(self):
        return self.time if self.time is not None else time.perf_counter()

    def render_post(self, camera, is_drone_cam, aspect, draw):
        """Rend la scène dans des tampons internes, applique SSAO et effets caméra, écrit dans la vue courante."""
        ctx = self.ctx
        opt = self.options
        target, vp = ctx.fbo, tuple(ctx.viewport)
        W, H = vp[2], vp[3]
        cam_fx = is_drone_cam
        distort = cam_fx and opt["distorsion"]
        k1, k2 = float(self.params["k1"]), float(self.params["k2"])
        tan_v = math.tan(math.radians(getattr(camera, "fovy", 60.0)) / 2.0)
        tan_half = np.array([tan_v * aspect, tan_v])
        overscan = 1.0
        if distort:
            # marge nécessaire : rayon non distordu du coin de l'image distordue
            rd = float(np.linalg.norm(tan_half))
            ru = rd
            for _ in range(20):
                ru = rd / (1.0 + k1 * ru * ru + k2 * ru ** 4)
            overscan = max(1.0, ru / rd * 1.01)
        ss = 2 if distort else 1      # suréchantillonnage : la distorsion comprime la périphérie
        rw, rh = int(round(W * overscan * ss)), int(round(H * overscan * ss))
        buf = self._get(self._buffers, _Buffers, rw, rh)
        cam = _WideCamera(camera, overscan) if overscan > 1.0 else camera

        # 1. scène
        buf.scene_fbo.use()
        ctx.viewport = (0, 0, rw, rh)
        buf.scene_fbo.clear(*SKY, 1.0, depth=1.0)
        ctx.enable(moderngl.DEPTH_TEST)
        draw(cam, aspect)
        ctx.disable(moderngl.DEPTH_TEST)
        proj = np.asarray(cam.projection(aspect), dtype=float)
        viewm = np.asarray(cam.view(), dtype=float)

        # 2. SSAO puis couleur éclairée (couleur x AO floutée)
        use_ao = opt["ssao"]
        buf.depth.use(location=0)
        if use_ao:
            buf.ao_fbo.use()
            p = self.ssao_prog
            p["u_depth"].value = 0
            p["u_proj"].write(_gl(proj))
            p["u_inv_proj"].write(_gl(np.linalg.inv(proj)))
            p["u_radius"].value = float(self.params["ssao_rayon"])
            self._vaos[p].render(moderngl.TRIANGLES, vertices=3)
        buf.lit_fbo.use()
        p = self.lit_prog
        buf.color.use(location=0)
        buf.ao.use(location=1)
        p["u_color"].value, p["u_ao"].value = 0, 1
        p["u_use_ao"].value = use_ao
        self._set(p, "u_ao_strength", float(self.params["ssao_intensite"]))
        self._vaos[p].render(moderngl.TRIANGLES, vertices=3)

        # 3. état de la caméra : exposition, flou
        st = None
        gain, iso, blur_scale = 1.0, 1.0, 0.0
        vp_mat = proj @ viewm
        if cam_fx:
            st = self._cams.setdefault(id(camera), _CamState())
            now = self._now()
            if st.t is None or now > st.t:
                dt = None if st.t is None else now - st.t
                st.prev_vp, st.vp, st.t = st.vp, vp_mat, now
                st.frame += 1
                if opt["expo"]:
                    buf.lit.build_mipmaps(0, buf.levels)
                    m = np.frombuffer(buf.lit.read(level=buf.levels), dtype="f2").astype(float)
                    lum = max(0.2126 * m[0] + 0.7152 * m[1] + 0.0722 * m[2], 1e-3)
                    g_target = float(np.clip(self.params["expo_cible"] / lum, 0.25, 8.0))
                    if st.gain is None or dt is None:
                        st.gain = g_target
                    else:
                        a = 1.0 - math.exp(-dt / max(self.params["expo_tau"], 1e-3))
                        st.gain += (g_target - st.gain) * a
                else:
                    st.gain = None
                pose = self.params["pose_ms"] / 1000.0
                if st.gain is not None:   # pose allongée d'abord, puis gain analogique
                    pose_eff = min(pose * max(st.gain, 1.0), max(self.params["pose_max_ms"] / 1000.0, pose))
                    st.blur_scale = pose_eff / dt if dt else 0.0
                else:
                    st.blur_scale = pose / dt if dt else 0.0
            if st.gain is not None:
                gain = st.gain
                pose = self.params["pose_ms"] / 1000.0
                pose_eff = min(pose * max(gain, 1.0), max(self.params["pose_max_ms"] / 1000.0, pose))
                iso = max(gain * pose / pose_eff, 1.0)
            blur_scale = st.blur_scale

        # 4. composition finale (distorsion, flou, exposition, bruit)
        jpeg = cam_fx and opt["jpeg"]
        if jpeg:
            out = self._get(self._out, _OutBuffers, W, H)
            out.fbo.use()
            ctx.viewport = (0, 0, W, H)
        else:
            target.use()
            ctx.viewport = vp
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.depth_func = "1"            # écrit la profondeur de l'image (vérité terrain alignée)
        p = self.comp_prog
        buf.lit.use(location=0)
        buf.depth.use(location=1)
        p["u_lit"].value, p["u_depth"].value = 0, 1
        p["u_distort"].value = distort
        p["u_k"].value = (k1, k2)
        p["u_tan_half"].value = tuple(float(x) for x in tan_half)
        p["u_overscan"].value = float(overscan)
        p["u_ss"].value = ss
        use_blur = cam_fx and opt["flou"] and st is not None and st.prev_vp is not None and blur_scale > 0
        p["u_blur"].value = bool(use_blur)
        if use_blur:
            p["u_inv_vp"].write(_gl(np.linalg.inv(vp_mat)))
            p["u_prev_vp"].write(_gl(st.prev_vp))
            p["u_blur_scale"].value = float(min(blur_scale, 1.0))
        p["u_gain"].value = float(gain)
        p["u_noise"].value = bool(cam_fx and opt["bruit"])
        p["u_noise_level"].value = float(self.params["bruit_niveau"])
        p["u_iso"].value = float(iso)
        p["u_frame"].value = float(st.frame % 1000 if st is not None else 0)
        self._vaos[p].render(moderngl.TRIANGLES, vertices=3)

        if jpeg:
            import cv2
            img = np.frombuffer(out.color.read(), np.uint8).reshape(H, W, 3)
            q = int(self.params["jpeg_qualite"])
            ok, enc = cv2.imencode(".jpg", img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, q])
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)[:, :, ::-1]
            out.jpeg.write(np.ascontiguousarray(dec).tobytes())
            target.use()
            ctx.viewport = vp
            p = self.copy_prog
            out.jpeg.use(location=0)
            out.depth.use(location=1)
            p["u_tex"].value, p["u_depth"].value = 0, 1
            self._vaos[p].render(moderngl.TRIANGLES, vertices=3)
        ctx.depth_func = "<"
