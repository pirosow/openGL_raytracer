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
};

struct BoundingBox {
    uint numTriangles;
    uint triangleOffset;

    uint childA;
    uint childB;

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

    return hit;
}

bool rayBoundingBoxIntersects(Ray ray, BoundingBox box) {
    float tmin = -1e30;
    float tmax =  1e30;
    const float EPS = 1e-8;

    // X axis
    float d = ray.dir.x;
    if (abs(d) < EPS) {
        if (ray.origin.x < box.posMin.x || ray.origin.x > box.posMax.x) return false;
    } else {
        float invD = 1.0 / d;
        float t0 = (box.posMin.x - ray.origin.x) * invD;
        float t1 = (box.posMax.x - ray.origin.x) * invD;
        if (t0 > t1) { float tmp = t0; t0 = t1; t1 = tmp; }
        tmin = max(tmin, t0);
        tmax = min(tmax, t1);
        if (tmax < tmin) return false;
    }

    // Y axis
    d = ray.dir.y;
    if (abs(d) < EPS) {
        if (ray.origin.y < box.posMin.y || ray.origin.y > box.posMax.y) return false;
    } else {
        float invD = 1.0 / d;
        float t0 = (box.posMin.y - ray.origin.y) * invD;
        float t1 = (box.posMax.y - ray.origin.y) * invD;
        if (t0 > t1) { float tmp = t0; t0 = t1; t1 = tmp; }
        tmin = max(tmin, t0);
        tmax = min(tmax, t1);
        if (tmax < tmin) return false;
    }

    // Z axis
    d = ray.dir.z;
    if (abs(d) < EPS) {
        if (ray.origin.z < box.posMin.z || ray.origin.z > box.posMax.z) return false;
    } else {
        float invD = 1.0 / d;
        float t0 = (box.posMin.z - ray.origin.z) * invD;
        float t1 = (box.posMax.z - ray.origin.z) * invD;
        if (t0 > t1) { float tmp = t0; t0 = t1; t1 = tmp; }
        tmin = max(tmin, t0);
        tmax = min(tmax, t1);
        if (tmax < tmin) return false;
    }

    return true;
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
    float closestDist = 1e30;
    Hit closest;
    closest.didHit = false;

    //check bounding boxes
    for (int i=0; i < boundingBoxCount; i++) {
        BoundingBox boundingBox = boundingBoxes[i];

        if (rayBoundingBoxIntersects(ray, boundingBox)) {
            uint start = boundingBox.triangleOffset;
            uint range = boundingBox.numTriangles + start;

            //check triangles
            for (uint j=start; j < range; j++) {
                Hit h = rayTriangleIntersects(ray, tris[triangleIndices[j]]);

                if (h.didHit) {
                    float d = getDist(ray.origin, h.hit_point);

                    if (d < closestDist) {
                        closestDist = d;
                        closest = h;
                    }
                }
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
            incomingLight += skyColor * skyBrightness; //getEnvironmentLight(ray) * rayColor;

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

    Ray ray;
    ray.origin = camPos;
    ray.dir = dir;
    ray.bounces = 0;

    vec3 currColor = trace(ray, nBounces, rays_per_pixel);

    vec3 prevColor = (frameNumber > 0) ? texture(prevFrame, uv).rgb : currColor;

    // Progressive average
    vec3 colorOut = (prevColor * float(frameNumber) + currColor) / float(frameNumber + 1);

    color = vec4(colorOut, 1.0);
}