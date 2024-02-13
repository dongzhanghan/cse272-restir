#pragma once
#include "scene.h"
#include "pcg.h"
// The simplest volumetric renderer: 
// single absorption only homogeneous volume
// only handle directly visible light sources
Spectrum vol_path_tracing_1(const Scene &scene,
                            int x, int y, /* pixel coordinates */
                            pcg32_state &rng) {
    int w = scene.camera.width, h = scene.camera.height;
    Vector2 screen_pos((x + next_pcg32_real<Real>(rng)) / w,
                       (y + next_pcg32_real<Real>(rng)) / h);
    Ray ray = sample_primary(scene.camera, screen_pos);
    //disable ray differentials
    RayDifferential ray_diff = RayDifferential{Real(0), Real(0)};

    std::optional<PathVertex> vertex_ = intersect(scene, ray, ray_diff);
    if (!vertex_) {
        return make_zero_spectrum();
    }
    Spectrum radiance = make_zero_spectrum();
    PathVertex vertex = *vertex_;
    Spectrum sigma_a = get_sigma_a(scene.media[vertex.exterior_medium_id], vertex.position);
    Real t = distance(vertex.position,ray.org);
    Spectrum transmittance = exp(- sigma_a * t);
    Spectrum Le = make_zero_spectrum();
    if(is_light(scene.shapes[vertex.shape_id])) {
        Le = emission(vertex, -ray.dir, scene);
    }
    radiance = Le * transmittance;
    return radiance;

}

// The second simplest volumetric renderer: 
// single monochromatic homogeneous volume with single scattering,
// no need to handle surface lighting, only directly visible light source
Spectrum vol_path_tracing_2(const Scene &scene,
                            int x, int y, /* pixel coordinates */
                            pcg32_state &rng) {
    int w = scene.camera.width, h = scene.camera.height;
    Vector2 screen_pos((x + next_pcg32_real<Real>(rng)) / w,
        (y + next_pcg32_real<Real>(rng)) / h);
    Ray ray = sample_primary(scene.camera, screen_pos);
    //disable ray differentials
    RayDifferential ray_diff = RayDifferential{ Real(0), Real(0) };

    std::optional<PathVertex> vertex_ = intersect(scene, ray, ray_diff);
    Real t_hit;
    Spectrum sigma_t,sigma_s,sigma_a;
    PhaseFunction phase;
    if (!vertex_) {
        t_hit = infinity<Real>();
        sigma_a = get_sigma_a(scene.media[scene.camera.medium_id], ray.org);
        sigma_s = get_sigma_s(scene.media[scene.camera.medium_id], ray.org);
        sigma_t = sigma_a + sigma_s;
        phase = get_phase_function(scene.media[scene.camera.medium_id]);
    }
    else {      
        PathVertex vertex = *vertex_;
        sigma_a = get_sigma_a(scene.media[vertex.exterior_medium_id], vertex.position);
        sigma_s = get_sigma_s(scene.media[vertex.exterior_medium_id], vertex.position);
        sigma_t = sigma_a + sigma_s;       
        t_hit = distance(vertex.position, ray.org);
        phase = get_phase_function(scene.media[vertex.exterior_medium_id]);
    }
    Real u = next_pcg32_real<Real>(rng);
    Real t = -log(1 - u) / sigma_t.x;
    if (t < t_hit) {
        Spectrum trans_pdf = exp(-sigma_t * t) * sigma_t;
        Spectrum transmittance = exp(-sigma_t * t);
        Vector3 p = ray.org + t * ray.dir;
        Vector2 light_uv{ next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng) };
        Real light_w = next_pcg32_real<Real>(rng);
        Real shape_w = next_pcg32_real<Real>(rng);
        int light_id = sample_light(scene, light_w);
        const Light& light = scene.lights[light_id];
        PointAndNormal point_on_light = sample_point_on_light(light, p, light_uv, shape_w, scene);
        Real L_s1_pdf = light_pmf(scene, light_id) *
            pdf_point_on_light(light, point_on_light, p, scene);
        Spectrum L_s1_estimate;
        Vector3 dir_light = normalize(point_on_light.position - p);
        Real G = 0;
        Ray shadow_ray{ p, dir_light,
                               get_shadow_epsilon(scene),
                               (1 - get_shadow_epsilon(scene)) *
                                   distance(point_on_light.position, p) };
        if (!occluded(scene, shadow_ray)) {
            // geometry term is cosine at v_{i+1} divided by distance squared
            // this can be derived by the infinitesimal area of a surface projected on
            // a unit sphere -- it's the Jacobian between the area measure and the solid angle
            // measure.
            G = max(-dot(dir_light, point_on_light.normal), Real(0)) /
                distance_squared(point_on_light.position, p);
        }
        L_s1_estimate = eval(phase, -ray.dir, dir_light) * emission(light, -dir_light, Real(0), point_on_light, scene) * exp(-sigma_t * length(point_on_light.position - p))*G;
        return (transmittance / trans_pdf) * sigma_s * (L_s1_estimate / L_s1_pdf);
    }
    else {
        PathVertex vertex = *vertex_;
        Spectrum trans_pdf = exp(-sigma_t * t_hit);
        Spectrum transmittance = exp(-sigma_t * t_hit);
        Spectrum Le = make_zero_spectrum();
        if (is_light(scene.shapes[vertex.shape_id])) {
            Le = emission(vertex, -ray.dir, scene);
        }
        return (transmittance/trans_pdf)*Le;
    }

}

// The third volumetric renderer (not so simple anymore): 
// multiple monochromatic homogeneous volumes with multiple scattering
// no need to handle surface lighting, only directly visible light source
Spectrum vol_path_tracing_3(const Scene &scene,
                            int x, int y, /* pixel coordinates */
                            pcg32_state &rng) {
    // Homework 2: implememt this!
    return make_zero_spectrum();
}

// The fourth volumetric renderer: 
// multiple monochromatic homogeneous volumes with multiple scattering
// with MIS between next event estimation and phase function sampling
// still no surface lighting
Spectrum vol_path_tracing_4(const Scene &scene,
                            int x, int y, /* pixel coordinates */
                            pcg32_state &rng) {
    // Homework 2: implememt this!
    return make_zero_spectrum();
}

// The fifth volumetric renderer: 
// multiple monochromatic homogeneous volumes with multiple scattering
// with MIS between next event estimation and phase function sampling
// with surface lighting
Spectrum vol_path_tracing_5(const Scene &scene,
                            int x, int y, /* pixel coordinates */
                            pcg32_state &rng) {
    // Homework 2: implememt this!
    return make_zero_spectrum();
}

// The final volumetric renderer: 
// multiple chromatic heterogeneous volumes with multiple scattering
// with MIS between next event estimation and phase function sampling
// with surface lighting
Spectrum vol_path_tracing(const Scene &scene,
                          int x, int y, /* pixel coordinates */
                          pcg32_state &rng) {
    // Homework 2: implememt this!
    return make_zero_spectrum();
}


