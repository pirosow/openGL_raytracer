#version 430 core

in vec2 uv;

out vec4 color;

uniform int tileX;
uniform int tileY;
uniform int tileSizeX;
uniform int tileSizeY;
uniform int numTilesX;
uniform int numTilesY;

uniform int total_frames;

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

uniform sampler2D prevFrame;
uniform int frameNumber;

uniform int trisCount;
uniform int boundingBoxCount;

uint seed;

vec3 skyColor;

float seed2 = uint(gl_FragCoord.x) * 1973u ^ uint(gl_FragCoord.y) * 9277u ^ uint(total_frames) * 1664525u;

struct Ray {
    vec3 origin;
    vec3 dir;
    float bounces;
};

struct Vertex {
    vec3 pos;
    vec3 normal;
};

struct Ball {
    vec3 pos;
    float radius;

    vec3 color;

    float emission;
    vec3 emission_color;

    float roughness;
};

struct Triangle {
    Vertex v0;
    Vertex v1;
    Vertex v2;

    vec3 color;
    vec3 emission_color;

    vec2 surface;
};

struct Hit {
    bool didHit;
    vec3 hit_point;
    vec3 normal;

    vec3 color;

    float emission;
    vec3 emission_color;

    float roughness;

    float t;
};

struct BoundingBox {
    uint numTriangles;
    uint triangleOffset;

    int childA;
    int childB;

    vec3 posMin;
    vec3 posMax;
};

layout (std430, binding=0) buffer TriBuffer {
    Triangle tris[];
};

layout (std430, binding=1) buffer boundingBoxBuffer {
    BoundingBox boundingBoxes[];
};

layout (std430, binding=2) buffer indicesBuffer {
    uint triangleIndices[];
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
        return Hit(false, vec3(999, 999, 999), vec3(0, 0, 0), ball.color, ball.emission, ball.emission_color, ball.roughness, 0);
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
        return Hit(false, vec3(999, 999, 999), vec3(0, 0, 0), ball.color, ball.emission, ball.emission_color, ball.roughness, t);
    }

    vec3 hit_point = origin + ray.dir * t;
    vec3 normal = (hit_point - ballPos) / radius;

    if (dot(ray.dir, normal) > 0) {
        normal = -1 * normal;
    }

    return Hit(true, hit_point, normal, ball.color, ball.emission, ball.emission_color, ball.roughness, t);
}

Hit rayTriangleIntersects(Ray ray, Triangle triangle) {
    Hit hit;
    hit.didHit = false;

    const float EPS = 1e-6f;

    vec3 edgeAB = triangle.v1.pos - triangle.v0.pos;
    vec3 edgeAC = triangle.v2.pos - triangle.v0.pos;

    // triangleFaceVector is the (non-unit) normal * area factor
    vec3 triangleFaceVector = cross(edgeAB, edgeAC);

    // Ray-plane denominator
    float determinant = dot(ray.dir, triangleFaceVector);
    if (abs(determinant) < EPS) {
        // Ray is parallel to triangle plane (or nearly so) -> no hit
        return hit;
    }

    float invDet = 1.0f / determinant;

    // compute t (distance along ray) using plane intersection
    vec3 vertRayOffset = ray.origin - triangle.v0.pos;
    float t = -dot(vertRayOffset, triangleFaceVector) * invDet;

    // reject hits behind the ray origin or extremely close
    if (t <= EPS) {
        return hit;
    }

    // compute barycentric coordinates
    vec3 rayOffsetPerp = cross(vertRayOffset, ray.dir);
    float u = -dot(edgeAC, rayOffsetPerp) * invDet;
    float v =  dot(edgeAB, rayOffsetPerp) * invDet;

    // point is inside triangle iff u >= 0, v >= 0, u+v <= 1
    if (u < 0.0f || v < 0.0f || (u + v) > 1.0f) {
        return hit;
    }

    // success — compute hit point & normal
    vec3 hit_point = ray.origin + ray.dir * t;
    vec3 normal = normalize(triangleFaceVector);

    // make sure normal faces against the incoming ray
    if (dot(ray.dir, normal) > 0.0f) {
        normal = -normal;
    }

    hit.didHit = true;
    hit.hit_point = hit_point;
    hit.normal = normal;
    hit.color = triangle.color;
    hit.emission = triangle.surface.x;
    hit.emission_color = triangle.emission_color;
    hit.roughness = triangle.surface.y;
    hit.t = t;

    return hit;
}

// returns >=0 : distance to box (clamped to 0 if origin is inside the box)
// returns -1 : no intersection in front of the ray
float rayBoundingBoxIntersects(Ray ray, BoundingBox box) {
    // vectorized slab test (assumes IEEE inf for 1/0)
    vec3 invDir = 1.0f / ray.dir;
    vec3 tMin = (box.posMin - ray.origin) * invDir;
    vec3 tMax = (box.posMax - ray.origin) * invDir;

    vec3 t1 = min(tMin, tMax);
    vec3 t2 = max(tMin, tMax);

    float dstNear = max(max(t1.x, t1.y), t1.z);
    float dstFar  = min(min(t2.x, t2.y), t2.z);

    // intersection exists if dstFar >= dstNear
    if (dstFar >= dstNear) {
        // box intersects the ray line. We want hits in front of the origin:
        if (dstFar < 0.0f) {
            // whole interval is behind the ray origin -> no hit
            return -1.0f;
        }
        // if origin is inside box, dstNear < 0, but we still want to visit it -> clamp to 0
        return (dstNear < 0.0f) ? 0.0f : dstNear;
    }
    return -1.0f;
}

float getDist(vec3 a, vec3 b) { return distance(a, b); }

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
    float t = 1 - n;

    vec3 d0 = length(diffuseDir) > 0.0 ? normalize(diffuseDir) : vec3(0.0);
    vec3 d1 = length(specularDir) > 0.0 ? normalize(specularDir) : vec3(0.0);
    return normalize(mix(d0, d1, t));
}

vec3 getEnvironmentLight(Ray ray) {
    return skyColor * skyBrightness;
}

Hit raycast(Ray ray) {
    float closestT = 1e30f;
    Hit closest; closest.didHit = false;

    const int MAX_STACK = 128;
    int stack[MAX_STACK];
    int sp = 0;
    stack[sp++] = 0; // root

    while (sp > 0) {
        int idx = stack[--sp];

        // defensive: if you want, enable this during debug only
        if (idx < 0 || idx >= boundingBoxCount) continue;

        float tNear = rayBoundingBoxIntersects(ray, boundingBoxes[idx]);
        if (tNear < 0.0f || tNear > closestT) continue;

        BoundingBox box = boundingBoxes[idx];

        if (box.childA == -1) {
            // leaf: iterate triangles
            uint start = box.triangleOffset;
            uint end   = start + box.numTriangles;
            for (uint ti = start; ti < end; ++ti) {
                uint triIdx = triangleIndices[ti];
                // optional debug check: if (triIdx >= trisCount) continue;

                Hit h = rayTriangleIntersects(ray, tris[triIdx]);
                if (h.didHit && h.t < closestT && h.t > 1e-6f) {
                    closestT = h.t;
                    closest  = h;
                }
            }
        } else {
            // internal node: evaluate both children once
            int a = box.childA;
            int b = box.childB;

            float tA = -1.0f, tB = -1.0f;
            if (a != -1) tA = rayBoundingBoxIntersects(ray, boundingBoxes[a]);
            if (b != -1) tB = rayBoundingBoxIntersects(ray, boundingBoxes[b]);

            // push far then near so near is popped first (LIFO)
            // always check sp < MAX_STACK before pushing
            if (tA >= 0.0f && tB >= 0.0f) {
                if (tA < tB) {
                    if (sp + 2 <= MAX_STACK) { stack[sp++] = b; stack[sp++] = a; }
                    else if (sp + 1 <= MAX_STACK) { stack[sp++] = a; } // fallback: push nearer
                } else {
                    if (sp + 2 <= MAX_STACK) { stack[sp++] = a; stack[sp++] = b; }
                    else if (sp + 1 <= MAX_STACK) { stack[sp++] = b; }
                }
            } else if (tA >= 0.0f) {
                if (sp < MAX_STACK) stack[sp++] = a;
            } else if (tB >= 0.0f) {
                if (sp < MAX_STACK) stack[sp++] = b;
            }
        }
    }
    return closest;
}

vec3 raytrace(Ray ray, int bounces) {
    vec3 incomingLight = vec3(0, 0, 0);
    vec3 rayColor = vec3(1, 1, 1);

    for (int i=0; i < bounces; i++) {
        Hit hit = raycast(ray);

        if (hit.didHit) {
            vec3 diffuseDir = diffuse(hit.normal);
            vec3 specularDir = reflect(ray.dir, hit.normal);

            vec3 dir = lerp(diffuseDir, specularDir, hit.roughness);

            ray.dir = dir;
            ray.origin = hit.hit_point + hit.normal * 1e-4;

            vec3 emittedLight = hit.emission_color * hit.emission;

            if (lambertian) {
                emittedLight *= 2;
            }

            incomingLight += emittedLight * rayColor;
            rayColor *= hit.color;

            ray.bounces++;

        } else {
            incomingLight += getEnvironmentLight(ray) * rayColor;

            break;
        }
    }

    return incomingLight;
}

vec3 trace(Ray ray, int bounces, int rays) {
    vec3 origin = ray.origin;
    vec3 dir = ray.dir;

    vec3 color = vec3(0, 0, 0);

    for (int i=0; i < rays; i++) {
        color = color + raytrace(ray, bounces);

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
    // Compute pixel index
    int px = int(gl_FragCoord.x);
    int py = int(gl_FragCoord.y);

    // Check if this pixel belongs to the current tile
    if ((px / tileSizeX) % numTilesX != tileX || (py / tileSizeY) % numTilesY != tileY) {
        color = texture(prevFrame, uv);

        return;
    }

    skyColor = vec3(0.1, 0.6, 0.92);

    seed = uint(gl_FragCoord.x) * 1973u ^ uint(gl_FragCoord.y) * 9277u ^ uint(frameNumber) * 1664525u;

    RandomValue(seed);
    RandomValue(seed);
    RandomValue(seed);

    vec3 dir = getDir(uv.x, uv.y);

    dir += (camRight * RandomValue(seed) + camUp * RandomValue(seed)) * jitterAmount;

    dir = normalize(dir);

    Ray ray;
    ray.origin = camPos;
    ray.dir = dir;
    ray.bounces = 0;

    vec3 currColor = trace(ray, nBounces, rays_per_pixel);

    vec3 prevColor = (frameNumber > 0) ? texture(prevFrame, uv).rgb : texture(prevFrame, uv).rgb;

    // Progressive average
    vec3 colorOut = (prevColor * float(frameNumber) + currColor) / float(frameNumber + 1);

    color = vec4(colorOut, 1.0);
}