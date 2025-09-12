#version 430 core

const int numBalls = 4;

in vec2 uv;

out vec4 color;

uniform bool lambertian;
uniform float skyBrightness;

uniform float fov;
uniform float xStep;
uniform float yStep;
uniform float dirStartX;
uniform float dirStartY;

uniform vec3 camPos;
uniform vec3 camRight;
uniform vec3 camUp;
uniform vec3 camForward;

uniform int nBounces;
uniform int rays_per_pixel;
uniform float jitterAmount;

uniform float frame;

uniform sampler2D prevFrame;
uniform int frameNumber;

uniform int nVertices;

uniform int trisCount;

uint seed;

vec3 skyColor;

struct Ray {
    vec3 origin;
    vec3 dir;
};

struct Ball {
    vec3 pos;
    float radius;

    vec3 color;

    float emission;
    vec3 emission_color;

    float roughness;
};

struct Hit {
    bool didHit;
    vec3 hit_point;
    vec3 normal;

    vec3 color;

    float emission;
    vec3 emission_color;

    float roughness;
};

struct Vertex {
    vec3 pos;
    vec3 normal;
};

struct Triangle {
    uvec3 indices;
    vec3 color;
    vec3 emission_color;
    float emission;
    float roughness;
};

layout (std430, binding=0) buffer VertexBuffer {
    Vertex vertices[];
};

layout (std430, binding=1) buffer TriBuffer {
    Triangle tris[];
};

vec3 normalizeTriangle(vec3 v0, vec3 v1, vec3 v2) {
    return normalize(cross(v1 - v0, v2 - v0));
}

Hit raySphereIntersects(Ray ray, Ball ball) {
    float radius = ball.radius;
    vec3 origin = ray.origin;
    vec3 ballPos = ball.pos;

    vec3 sphere_to_ray = origin - ballPos;

    float b = 2 * dot(ray.dir, sphere_to_ray);
    float c = dot(sphere_to_ray, sphere_to_ray) - (radius * radius);

    float discr = b * b - 4 * c;

    if (discr < 0) {
        return Hit(false, vec3(999, 999, 999), vec3(0, 0, 0), ball.color, ball.emission, ball.emission_color, ball.roughness);
    }

    float sqrt_d = sqrt(discr);

    float t0 = (-b - sqrt_d) / 2.0;
    float t1 = (-b + sqrt_d) / 2.0;

    float eps = 0.0000005 * ball.radius;

    float t = 0;

    if (t0 > eps) {
        t = t0;
    } else if (t1 > eps) {
        t = t1;
    } else {
        return Hit(false, vec3(999, 999, 999), vec3(0, 0, 0), ball.color, ball.emission, ball.emission_color, ball.roughness);
    }

    vec3 hit_point = origin + ray.dir * t;
    vec3 normal = (hit_point - ballPos) / radius;

    if (dot(ray.dir, normal) > 0) {
        normal = -1 * normal;
    }

    return Hit(true, hit_point, normal, ball.color, ball.emission, ball.emission_color, ball.roughness);
}

float getDist(vec3 pos1, vec3 pos2) {
    float distance = sqrt(pow(pos1.x - pos2.x, 2) + pow(pos1.y - pos2.y, 2) + pow(pos1.z - pos2.z, 2));

    return distance;
}

float RandomValue(inout uint state)
{
    // advance state (LCG-ish)
    state = state * 747796405u + 2891336453u;

    // scramble
    uint t = state >> ((state >> 28u) + 4u);
    uint result = (t ^ state) * 277803737u;
    result = (result >> 22u) ^ result;

    // map to [0,1]
    return float(result) / 4294967295.0 * 2 - 1;
}

vec3 diffuse(vec3 normal) {
    vec3 dir = vec3(RandomValue(seed), RandomValue(seed), RandomValue(seed));

    if (!lambertian) {
        if (dot(dir, normal) < 0) {
            dir = -1 * dir;
        }

        return normalize(dir);
    }

    return normalize(normal + dir);
}

vec3 lerp(vec3 diffuseDir, vec3 specularDir, float n) {
    float t = n;

    vec3 d0 = length(diffuseDir) > 0.0 ? normalize(diffuseDir) : vec3(0.0);
    vec3 d1 = length(specularDir) > 0.0 ? normalize(specularDir) : vec3(0.0);
    return normalize(mix(d0, d1, t));
}

vec3 getEnvironmentLight(Ray ray) {
    return skyColor * skyBrightness;
}

Hit raycast(Ray ray, Ball balls[numBalls]) {
    float closestDist = 1e30;
    Hit closest;
    closest.didHit = false;

    // check spheres
    for (int i = 0; i < numBalls; ++i) {
        Hit h = raySphereIntersects(ray, balls[i]);

        if (h.didHit) {
            float d = getDist(ray.origin, h.hit_point);

            if (d < closestDist) {
                closestDist = d;
                closest = h;
            }
        }
    }

    return closest;
}

vec3 raytrace(Ray ray, Ball balls[numBalls], int bounces) {
    vec3 incomingLight = vec3(0, 0, 0);
    vec3 rayColor = vec3(1, 1, 1);

    for (int i=0; i < bounces; i++) {
        Hit hit = raycast(ray, balls);

        if (hit.didHit) {
            vec3 diffuseDir = diffuse(hit.normal);
            vec3 specularDir = reflect(ray.dir, hit.normal);

            vec3 dir = lerp(diffuseDir, specularDir, hit.roughness);

            ray.dir = dir;
            ray.origin = hit.hit_point + hit.normal * 0.1;

            vec3 emittedLight = hit.emission_color * hit.emission;

            if (lambertian) {
                emittedLight *= 2;
            }

            incomingLight += emittedLight * rayColor;
            rayColor *= hit.color;

        } else {
            incomingLight += getEnvironmentLight(ray) * rayColor;

            break;
        }
    }

    return incomingLight;
}

vec3 trace(Ray ray, Ball balls[numBalls], int bounces, int rays) {
    vec3 origin = ray.origin;
    vec3 dir = ray.dir;

    vec3 color = vec3(0, 0, 0);

    for (int i=0; i < rays; i++) {
        color = color + raytrace(ray, balls, bounces);

        ray.origin = origin;
        ray.dir = dir;
    }

    return color / rays;
}

vec3 getDir(float x, float y) {
    vec3 dir = vec3((dirStartX + x * xStep), dirStartY + y * yStep, 1.0);

    dir = camRight * dir.x + camUp * dir.y + camForward * dir.z;

    return normalize(dir);
}

void main() {
    skyColor = vec3(0.1, 0.6, 0.92);

    seed = uint(gl_FragCoord.x) * 1973u ^ uint(gl_FragCoord.y) * 9277u ^ uint(frameNumber) * 1664525u;

    Ball balls[numBalls];

    // pos, radius, color, emission, emission_color, roughness
    balls[0] = Ball(vec3(0, -1010, 20), 1000, vec3(1, 0.75, 1), 0, vec3(0, 0, 0), 0);   // floor
    balls[1]  = Ball(vec3(-20, 1.5, -10), 13, vec3(1, 0.3, 0.5), 0, vec3(0, 0, 0), 0);   // pink-ish
    balls[2]  = Ball(vec3(10, 5, 20), 15, vec3(1, 1, 1), 0, vec3(0, 0, 0), 1);   // blue
    balls[3]  = Ball(vec3(-1000, 100, 1000), 600, vec3(0, 0, 0), 3, vec3(1, 1, 1), 0);   // distant light

    vec3 dir = getDir(uv.x, uv.y);

    RandomValue(seed);
    RandomValue(seed);
    RandomValue(seed);

    dir += RandomValue(seed) * jitterAmount;

    Ray ray;
    ray.origin = camPos;
    ray.dir = dir;

    vec3 currColor = trace(ray, balls, nBounces, rays_per_pixel);

    vec3 prevColor = (frameNumber > 0) ? texture(prevFrame, uv).rgb : vec3(0.0);

    // Progressive average
    vec3 colorOut = (prevColor * float(frameNumber) + currColor) / float(frameNumber + 1);

    color = vec4(colorOut, 1.0);
}