from __future__ import annotations

import argparse
import math

import moderngl
import numpy as np
import pygame

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DURATION = 64.0
CAMERA_Z = 15.0
FOV = math.radians(50.0)
SPARK_CAPACITY = 7000

ROSE = np.array([1.00, 0.18, 0.42], dtype="f4")
LILAC = np.array([0.57, 0.30, 1.00], dtype="f4")
CYAN = np.array([0.15, 0.68, 1.00], dtype="f4")
GOLD = np.array([1.00, 0.63, 0.25], dtype="f4")

PALETTE = np.array(
    [
        [1.00, 0.20, 0.43],
        [0.66, 0.32, 1.00],
        [0.24, 0.72, 1.00],
        [1.00, 0.65, 0.30],
        [1.00, 0.48, 0.68],
    ],
    dtype="f4",
)

# Procedural stroke alphabet.
# Each two-digit token is an (x, y) point on a 4 x 6 grid.
# "|" starts a separate stroke. No font or image is loaded.
GLYPHS = {
    "A": "00 26 40|03 43",
    "B": "00 06 36 45 44 33 03|33 42 41 30 00",
    "C": "45 36 16 05 01 10 30 41",
    "D": "00 06 26 45 41 20 00",
    "E": "40 00 06 46|03 33",
    "F": "00 06 46|03 33",
    "G": "45 36 16 05 01 10 30 41 43 23",
    "H": "00 06|40 46|03 43",
    "I": "06 46|26 20|00 40",
    "J": "06 46|36 31 20 10 01",
    "K": "00 06|03 46|03 40",
    "L": "06 00 40",
    "M": "00 06 23 46 40",
    "N": "00 06 40 46",
    "O": "10 01 05 16 36 45 41 30 10",
    "P": "00 06 36 45 44 33 03",
    "Q": "10 01 05 16 36 45 41 30 10|22 40",
    "R": "00 06 36 45 44 33 03|23 40",
    "S": "45 36 16 05 04 13 33 42 41 30 10 01",
    "T": "06 46|26 20",
    "U": "06 01 10 30 41 46",
    "V": "06 20 46",
    "W": "06 10 23 30 46",
    "X": "06 40|00 46",
    "Y": "06 23 46|23 20",
    "Z": "06 46 00 40",
    "0": "10 01 05 16 36 45 41 30 10|01 45",
    "1": "15 26 20|10 30",
    "2": "05 16 36 45 44 00 40",
    "3": "06 36 45 34 23|34 42 41 30 00",
    "4": "36 03 43|36 30",
    "5": "46 06 03 33 42 41 30 00",
    "6": "45 36 16 05 01 10 30 41 42 33 03",
    "7": "06 46 10",
    "8": "13 04 05 16 36 45 44 33 13 02 01 10 30 41 42 33",
    "9": "43 13 04 05 16 36 45 41 30 10",
    "-": "03 43",
    ".": "20 21",
    "!": "26 22|20 21",
}

TRANSLITERATION = str.maketrans(
    {
        "Ç": "C",
        "Ğ": "G",
        "İ": "I",
        "Ö": "O",
        "Ş": "S",
        "Ü": "U",
        "Â": "A",
        "Î": "I",
        "Û": "U",
    }
)

def smooth01(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)

def mix(a, b, amount):
    return a * (1.0 - amount) + b * amount

def unit_vectors(rng, count):
    v = rng.normal(size=(count, 3)).astype("f4")
    v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-6)
    return v

def rotation_matrix(yaw, pitch):
    cy, sy = math.cos(yaw), math.sin(yaw)
    cx, sx = math.cos(pitch), math.sin(pitch)

ry = np.array(
        [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype="f4"
    )
    rx = np.array(
        [[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype="f4"
    )
    return ry @ rx

def projection_matrix(aspect):
    f = 1.0 / math.tan(FOV * 0.5)
    near, far = 0.1, 100.0

projection = np.zeros((4, 4), dtype="f4")
    projection[0, 0] = f / aspect
    projection[1, 1] = f
    projection[2, 2] = (far + near) / (near - far)
    projection[2, 3] = 2.0 * far * near / (near - far)
    projection[3, 2] = -1.0

view = np.eye(4, dtype="f4")
    view[2, 3] = -CAMERA_Z
    return projection @ view

def screen_to_world(x, y, width, height):
    # Intersection of the camera ray with the world-space z=0 plane.
    half_height = CAMERA_Z * math.tan(FOV * 0.5)
    aspect = width / max(height, 1)

return np.array(
        [
            (2.0 * x / width - 1.0) * half_height * aspect,
            (1.0 - 2.0 * y / height) * half_height,
            0.0,
        ],
        dtype="f4",
    )

def make_text(rng, text, count):
    segments = []
    cursor = 0.0

for char in text:
        if char == " ":
            cursor += 3.2
            continue

for stroke in GLYPHS[char].split("|"):
            points = [
                (cursor + int(token[0]), int(token[1]))
                for token in stroke.split()
            ]
            segments.extend(zip(points[:-1], points[1:]))

cursor += 5.3

starts = np.asarray([a for a, _ in segments], dtype="f4")
    ends = np.asarray([b for _, b in segments], dtype="f4")
    lengths = np.linalg.norm(ends - starts, axis=1)

selected = rng.choice(
        len(segments), size=count, p=lengths / lengths.sum()
    )
    u = rng.random((count, 1)).astype("f4")
    points = mix(starts[selected], ends[selected], u)

all_points = np.concatenate((starts, ends))
    lo, hi = all_points.min(axis=0), all_points.max(axis=0)
    scale = min(0.37, 10.8 / max(float(hi[0] - lo[0]), 1.0))

points = (points - (lo + hi) * 0.5) * scale

result = np.zeros((count, 3), dtype="f4")
    result[:, :2] = points
    result[:, :2] += rng.normal(0, 0.032, (count, 2))
    result[:, 2] = rng.normal(0, 0.085, count)

return result

def make_heart(rng, count):
    # Genuine volumetric heart:
    #
    # (x² + 9/4 d² + z² - 1)³ - x² z³ - 9/80 d² z³ <= 0
    #
    # z is up in the equation; d becomes world-space depth.
    chunks = []
    remaining = count

while remaining:
        batch_size = max(4096, remaining * 5)
        q = rng.uniform(
            [-1.35, -0.90, -1.20],
            [1.35, 0.90, 1.45],
            size=(batch_size, 3),
        ).astype("f4")

x, d, z = q.T
        a = x * x + 2.25 * d * d + z * z - 1.0
        implicit = a**3 - x * x * z**3 - 0.1125 * d * d * z**3

accepted = q[implicit <= 0.0][:remaining]
        chunks.append(accepted)
        remaining -= len(accepted)

q = np.concatenate(chunks)
    heart = q[:, [0, 2, 1]].copy()
    heart[:, 1] -= 0.12
    heart *= 2.25
    return heart.astype("f4")

# ---------------------------------------------------------------------
# GLSL
# ---------------------------------------------------------------------

FULLSCREEN_VERTEX = """
#version 330

out vec2 v_uv;

void main() {
    vec2 p = vec2(
        float((gl_VertexID << 1) & 2),
        float(gl_VertexID & 2)
    );

v_uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

BACKGROUND_FRAGMENT = """
#version 330

in vec2 v_uv;
out vec4 frag_color;

uniform vec2 u_resolution;
uniform float u_time;

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

float noise2(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);

float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));

return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float fbm(vec2 p) {
    float sum = 0.0;
    float amp = 0.5;
    mat2 turn = mat2(0.80, -0.60, 0.60, 0.80);

for (int i = 0; i < 5; ++i) {
        sum += amp * noise2(p);
        p = turn * p * 2.03 + vec2(4.1, 7.3);
        amp *= 0.5;
    }

return sum;
}

vec3 stars(vec2 p, float density, float layer) {
    vec2 grid = p * density;
    vec2 cell = floor(grid);
    vec2 local = fract(grid);

float seed = hash21(cell + layer * 17.0);

vec2 center = vec2(
        hash21(cell + 3.7 + layer),
        hash21(cell + 9.2 + layer)
    );
    center = 0.15 + center * 0.70;

float radius = length(local - center);
    float size = mix(0.018, 0.060, seed * seed);
    float star = exp(-radius * radius / (size * size));

star *= step(0.965, seed);
    star *= 0.62 + 0.38 * sin(u_time * (0.7 + seed) + seed * 90.0);

vec3 tint = mix(
        vec3(0.42, 0.60, 1.0),
        vec3(1.0, 0.74, 0.60),
        hash21(cell + 35.0)
    );

return tint * star * (0.8 + layer * 0.3);
}

void main() {
    vec2 p = v_uv - 0.5;
    p.x *= u_resolution.x / u_resolution.y;

float t = u_time * 0.012;
    vec2 warp = vec2(
        fbm(p * 2.0 + vec2(t, 0.0)),
        fbm(p * 2.0 + vec2(5.2, -t))
    );

float cloud = fbm(p * 3.0 + warp * 2.3);
    float band = exp(-pow((p.y + p.x * 0.25) * 2.3, 2.0));
    float dust = pow(max(cloud, 0.0), 2.7) * band;

vec3 color = vec3(0.002, 0.003, 0.011);

vec3 nebula = mix(
        vec3(0.020, 0.035, 0.135),
        vec3(0.170, 0.025, 0.110),
        fbm(p * 1.7 + 8.0)
    );

color += nebula * dust * 1.25;
    color += vec3(0.025, 0.070, 0.100)
        * pow(fbm(p * 4.0 - warp), 4.0) * band;

color += stars(p + vec2(t * 0.03, 0.0), 85.0, 1.0);
    color += stars(p + vec2(-t * 0.02, 0.0), 145.0, 2.0);

float vignette = 1.0 - smoothstep(0.35, 1.35, length(p));
    color *= 0.55 + 0.45 * vignette;

frag_color = vec4(color, 1.0);
}
"""

PARTICLE_VERTEX = """
#version 330

in vec3 in_position;
in vec3 in_color;
in float in_size;
in float in_alpha;

uniform mat4 u_mvp;
uniform float u_height;

out vec3 v_color;
out float v_alpha;

void main() {
    vec4 clip = u_mvp * vec4(in_position, 1.0);

v_color = in_color;
    v_alpha = in_alpha;

if (clip.w <= 0.1 || in_alpha <= 0.0001) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        gl_PointSize = 1.0;
        v_alpha = 0.0;
    } else {
        gl_Position = clip;
        gl_PointSize = clamp(in_size * u_height / clip.w, 1.0, 48.0);
    }
}
"""

PARTICLE_FRAGMENT = """
#version 330

in vec3 v_color;
in float v_alpha;

out vec4 frag_color;

void main() {
    vec2 p = gl_PointCoord * 2.0 - 1.0;
    float radius2 = dot(p, p);

if (radius2 > 1.0 || v_alpha <= 0.0001) {
        discard;
    }

float halo = exp(-radius2 * 4.2);
    float core = exp(-radius2 * 24.0);
    float edge = 1.0 - smoothstep(0.70, 1.0, radius2);

vec3 light = v_color * (0.85 + core * 3.5);
    light += vec3(1.0, 0.82, 0.90) * core * 0.28;

frag_color = vec4(light, halo * edge * v_alpha);
}
"""

EXTRACT_FRAGMENT = """
#version 330

in vec2 v_uv;
out vec4 frag_color;
uniform sampler2D u_image;

void main() {
    vec3 color = texture(u_image, v_uv).rgb;
    float brightness = max(color.r, max(color.g, color.b));

float threshold = 0.85;
    float knee = 0.45;

float soft = clamp(brightness - threshold + knee, 0.0, 2.0 * knee);
    soft = soft * soft / (4.0 * knee + 0.00001);

float contribution = max(brightness - threshold, soft);
    contribution /= max(brightness, 0.00001);

frag_color = vec4(color * contribution, 1.0);
}
"""

BLUR_FRAGMENT = """
#version 330

in vec2 v_uv;
out vec4 frag_color;

uniform sampler2D u_image;
uniform vec2 u_axis;

void main() {
    vec3 color = texture(u_image, v_uv).rgb * 0.227027;

color += texture(u_image, v_uv + u_axis * 1.384615).rgb * 0.316216;
    color += texture(u_image, v_uv - u_axis * 1.384615).rgb * 0.316216;

color += texture(u_image, v_uv + u_axis * 3.230769).rgb * 0.070270;
    color += texture(u_image, v_uv - u_axis * 3.230769).rgb * 0.070270;

frag_color = vec4(color, 1.0);
}
"""

COMPOSITE_FRAGMENT = """
#version 330

in vec2 v_uv;
out vec4 frag_color;

uniform sampler2D u_scene;
uniform sampler2D u_bloom;
uniform float u_strength;

vec3 aces(vec3 x) {
    return clamp(
        (x * (2.51 * x + 0.03)) /
        (x * (2.43 * x + 0.59) + 0.14),
        0.0,
        1.0
    );
}

void main() {
    vec3 color = texture(u_scene, v_uv).rgb;
    color += texture(u_bloom, v_uv).rgb * u_strength;

color = aces(color * 1.05);
    color = pow(color, vec3(1.0 / 2.2));

frag_color = vec4(color, 1.0);
}
"""

# ---------------------------------------------------------------------
# Renderer: HDR scene -> bloom extraction -> separable blur -> composite
# ---------------------------------------------------------------------

class Renderer:
    def __init__(self, ctx, width, height, capacity):
        self.ctx = ctx
        self.bloom_enabled = True
        self.targets = []

self.background = ctx.program(
            vertex_shader=FULLSCREEN_VERTEX,
            fragment_shader=BACKGROUND_FRAGMENT,
        )
        self.particles = ctx.program(
            vertex_shader=PARTICLE_VERTEX,
            fragment_shader=PARTICLE_FRAGMENT,
        )
        self.extract = ctx.program(
            vertex_shader=FULLSCREEN_VERTEX,
            fragment_shader=EXTRACT_FRAGMENT,
        )
        self.blur = ctx.program(
            vertex_shader=FULLSCREEN_VERTEX,
            fragment_shader=BLUR_FRAGMENT,
        )
        self.composite = ctx.program(
            vertex_shader=FULLSCREEN_VERTEX,
            fragment_shader=COMPOSITE_FRAGMENT,
        )

self.background_vao = ctx.vertex_array(self.background, [])
        self.extract_vao = ctx.vertex_array(self.extract, [])
        self.blur_vao = ctx.vertex_array(self.blur, [])
        self.composite_vao = ctx.vertex_array(self.composite, [])

self.vbo = ctx.buffer(reserve=capacity * 8 * 4, dynamic=True)
        self.particle_vao = ctx.vertex_array(
            self.particles,
            [
                (
                    self.vbo,
                    "3f 3f 1f 1f",
                    "in_position",
                    "in_color",
                    "in_size",
                    "in_alpha",
                )
            ],
        )

self.extract["u_image"].value = 0
        self.blur["u_image"].value = 0
        self.composite["u_scene"].value = 0
        self.composite["u_bloom"].value = 1

ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        ctx.disable(moderngl.DEPTH_TEST)

self.resize(width, height)

def make_target(self, width, height):
        texture = self.ctx.texture((width, height), 4, dtype="f2")
        texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        texture.repeat_x = False
        texture.repeat_y = False

framebuffer = self.ctx.framebuffer(color_attachments=[texture])
        self.targets.append((framebuffer, texture))
        return framebuffer, texture

def resize(self, width, height):
        for framebuffer, texture in self.targets:
            framebuffer.release()
            texture.release()
        self.targets.clear()

self.width = max(1, width)
        self.height = max(1, height)

self.bloom_width = max(1, self.width // 2)
        self.bloom_height = max(1, self.height // 2)

self.scene_fbo, self.scene_texture = self.make_target(
            self.width, self.height
        )
        self.bloom_a_fbo, self.bloom_a = self.make_target(
            self.bloom_width, self.bloom_height
        )
        self.bloom_b_fbo, self.bloom_b = self.make_target(
            self.bloom_width, self.bloom_height
        )

self.bloom_a_fbo.clear()
        self.bloom_b_fbo.clear()

mvp = projection_matrix(self.width / self.height)
        self.particles["u_mvp"].write(mvp.T.copy().tobytes())
        self.particles["u_height"].value = float(self.height)

def draw(self, data, time):
        ctx = self.ctx

# Scene pass.
        self.scene_fbo.use()
        ctx.viewport = (0, 0, self.width, self.height)
        ctx.disable(moderngl.BLEND)

self.background["u_resolution"].value = (
            float(self.width), float(self.height)
        )
        self.background["u_time"].value = float(time)
        self.background_vao.render(moderngl.TRIANGLES, vertices=3)

# Emissive particles: additive blending, no depth sorting required.
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE)

self.vbo.write(data.tobytes())
        self.particle_vao.render(moderngl.POINTS, vertices=len(data))

ctx.disable(moderngl.BLEND)

# Bloom pass at half resolution.
        if self.bloom_enabled:
            ctx.viewport = (0, 0, self.bloom_width, self.bloom_height)

self.bloom_a_fbo.use()
            self.scene_texture.use(location=0)
            self.extract_vao.render(moderngl.TRIANGLES, vertices=3)

for radius in (1.0, 2.0, 3.0):
                self.bloom_b_fbo.use()
                self.bloom_a.use(location=0)
                self.blur["u_axis"].value = (
                    radius / self.bloom_width, 0.0
                )
                self.blur_vao.render(moderngl.TRIANGLES, vertices=3)

self.bloom_a_fbo.use()
                self.bloom_b.use(location=0)
                self.blur["u_axis"].value = (
                    0.0, radius / self.bloom_height
                )
                self.blur_vao.render(moderngl.TRIANGLES, vertices=3)

# Final display pass.
        ctx.screen.use()
        ctx.viewport = (0, 0, self.width, self.height)

self.scene_texture.use(location=0)
        self.bloom_a.use(location=1)
        self.composite["u_strength"].value = (
            0.95 if self.bloom_enabled else 0.0
        )
        self.composite_vao.render(moderngl.TRIANGLES, vertices=3)

def release(self):
        for framebuffer, texture in self.targets:
            framebuffer.release()
            texture.release()

for resource in (
            self.particle_vao,
            self.background_vao,
            self.extract_vao,
            self.blur_vao,
            self.composite_vao,
            self.vbo,
            self.background,
            self.particles,
            self.extract,
            self.blur,
            self.composite,
        ):
            resource.release()

# ---------------------------------------------------------------------
# Vectorized simulation
# ---------------------------------------------------------------------

class Simulation:
    def __init__(self, count, text, seed=7):
        self.rng = np.random.default_rng(seed)
        self.count = count
        self.aspect = 16.0 / 9.0

rng = self.rng

self.phase = rng.uniform(0, math.tau, count).astype("f4")
        self.accent = PALETTE[rng.integers(0, len(PALETTE), count)]
        self.sizes = rng.uniform(0.060, 0.145, count).astype("f4")

self.drift = rng.uniform(
            [-6.0, -3.6, -3.5],
            [6.0, 3.6, 2.0],
            (count, 3),
        ).astype("f4")

self.text = make_text(rng, text, count)
        self.heart = make_heart(rng, count)

self.scatter = rng.normal(0, 1.7, (count, 3)).astype("f4")

self.radius = (
            5.4 * rng.random(count) ** 0.60
        ).astype("f4")

arms = rng.integers(0, 4, count)
        self.angle = (
            arms * math.tau / 4.0
            + self.radius * 1.35
            + rng.normal(0, 0.20, count)
        ).astype("f4")

self.thickness = (
            rng.normal(0, 0.16, count)
            * (1.0 - 0.11 * self.radius)
        ).astype("f4")

self.bulge_mask = rng.random(count) < 0.14
        self.bulge = rng.normal(0, 0.52, (count, 3)).astype("f4")

self.spark_position = np.zeros((SPARK_CAPACITY, 3), dtype="f4")
        self.spark_velocity = np.zeros_like(self.spark_position)
        self.spark_color = np.zeros_like(self.spark_position)
        self.spark_life = np.zeros(SPARK_CAPACITY, dtype="f4")
        self.spark_maxlife = np.ones(SPARK_CAPACITY, dtype="f4")
        self.spark_size = np.ones(SPARK_CAPACITY, dtype="f4") * 0.12
        self.spark_drag = np.ones(SPARK_CAPACITY, dtype="f4") * 0.20

# Interleaved GPU input: position.xyz, color.rgb, size, alpha.
        self.data = np.zeros((count + SPARK_CAPACITY, 8), dtype="f4")

self.reset()

def reset(self):
        self.t = 0.0
        self.position = self.drift.copy()
        self.velocity = np.zeros_like(self.position)

self.pointer = np.zeros(3, dtype="f4")
        self.flow = np.zeros(3, dtype="f4")
        self.down = False
        self.down_at = 0.0

self.yaw = 0.0
        self.pitch = 0.0
        self.shock = 0.0

self.fired = False
        self.next_firework = 50.6
        self.pending_impacts = []

self.spark_cursor = 0
        self.spark_life.fill(0.0)
        self.spark_velocity.fill(0.0)

self.color = LILAC.copy()

def charge(self):
        if not self.down:
            return 0.0
        return float(np.clip((self.t - self.down_at) / 1.7, 0.0, 1.0))

def move_pointer(self, world, wall_dt):
        delta = world - self.pointer
        self.pointer = world

motion = delta / max(wall_dt, 1.0 / 240.0)
        speed = float(np.linalg.norm(motion))

if speed > 14.0:
            motion *= 14.0 / speed

self.flow = self.flow * 0.55 + motion * 0.45

if self.down and 31.0 <= self.t < 50.0:
            self.yaw += float(delta[0]) * 0.30
            self.pitch -= float(delta[1]) * 0.30
            self.pitch = float(np.clip(self.pitch, -1.25, 1.25))

def press(self):
        if not self.down:
            self.down = True
            self.down_at = self.t

def release(self):
        if not self.down:
            return

held = self.t - self.down_at
        energy = self.charge()
        self.down = False

if held < 0.28:
            self.launch_comet(self.pointer.copy())
        else:
            self.impulse(self.pointer, 4.0 + 13.0 * energy)
            self.burst(
                self.pointer,
                int(500 + 2200 * energy),
                2.0 + 6.0 * energy,
                lifetime=1.5 + energy,
            )
            self.shock = max(self.shock, 0.5 + 0.5 * energy)

def allocate_sparks(self, count):
        count = min(int(count), SPARK_CAPACITY)
        idx = (
            np.arange(count, dtype=np.int32) + self.spark_cursor
        ) % SPARK_CAPACITY
        self.spark_cursor = (self.spark_cursor + count) % SPARK_CAPACITY
        return idx

def setup_sparks(self, idx, lifetime):
        count = len(idx)
        self.spark_life[idx] = lifetime
        self.spark_maxlife[idx] = lifetime
        self.spark_color[idx] = PALETTE[
            self.rng.integers(0, len(PALETTE), count)
        ] * 1.4
        self.spark_size[idx] = self.rng.uniform(0.08, 0.20, count)
        self.spark_drag[idx] = 0.20

def burst(self, origin, count, strength, lifetime=2.3):
        idx = self.allocate_sparks(count)
        count = len(idx)

directions = unit_vectors(self.rng, count)
        speeds = self.rng.uniform(0.25, 1.0, (count, 1)).astype("f4")

self.spark_position[idx] = (
            origin + directions * self.rng.uniform(0.01, 0.09, (count, 1))
        )
        self.spark_velocity[idx] = directions * speeds * strength
        self.spark_velocity[idx, 1] += 0.7

lives = self.rng.uniform(
            lifetime * 0.65, lifetime, count
        ).astype("f4")
        self.setup_sparks(idx, lives)

def impulse(self, origin, power):
        delta = self.position - origin
        distance2 = np.sum(delta * delta, axis=1)

direction = delta / np.sqrt(distance2[:, None] + 0.04)
        falloff = 1.0 / (1.0 + distance2 * 0.16)

self.velocity += direction * (power * falloff[:, None])

def launch_comet(self, target):
        # A short particle packet travels to the selected point.
        # Its initial vertical velocity compensates for gravity.
        idx = self.allocate_sparks(120)
        flight = 0.30

origin = np.array(
            [self.rng.uniform(-2.5, 2.5), -2.5, 0.0], dtype="f4"
        )
        self.spark_position[idx] = (
            origin + self.rng.normal(0, 0.035, (len(idx), 3))
        )

self.spark_velocity[idx] = (
            (target - self.spark_position[idx]) / flight
        )
        self.spark_velocity[idx, 1] += 0.5 * 0.65 * flight

self.setup_sparks(idx, flight)
        self.spark_drag[idx] = 0.0
        self.spark_color[idx] = GOLD * 1.8

delta = target - self.position
        distance2 = np.sum(delta * delta, axis=1)
        self.velocity += delta * (
            0.55 * np.exp(-distance2 * 0.035)
        )[:, None]

self.pending_impacts.append((self.t + flight, target))

def emit
