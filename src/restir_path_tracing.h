#pragma once
#include "scene.h"
#include "pcg.h"


/*Using RIS to combine NEE and bsdf sampling*/
Spectrum restir_path_tracing_1(const Scene& scene,
    int x, int y, /* pixel coordinates */
    pcg32_state& rng) {
    int w = scene.camera.width, h = scene.camera.height;
    Vector2 screen_pos((x + next_pcg32_real<Real>(rng)) / w,
        (y + next_pcg32_real<Real>(rng)) / h);
    Ray ray = sample_primary(scene.camera, screen_pos);
    RayDifferential ray_diff = init_ray_differential(w, h);
    Real p1, p2;
    Spectrum C1, C2;
    std::optional<PathVertex> vertex_ = intersect(scene, ray, ray_diff);
    if (!vertex_) {
        // Hit background. Account for the environment map if needed.
        if (has_envmap(scene)) {
            const Light& envmap = get_envmap(scene);
            return emission(envmap,
                -ray.dir, // pointing outwards from light
                ray_diff.spread,
                PointAndNormal{}, // dummy parameter for envmap
                scene);
        }
        return make_zero_spectrum();
    }
    PathVertex vertex = *vertex_;
    Spectrum radiance = make_zero_spectrum();
    Spectrum current_path_throughput = fromRGB(Vector3{ 1, 1, 1 });
    Real eta_scale = Real(1);
    if (is_light(scene.shapes[vertex.shape_id])) {
        radiance += current_path_throughput *
            emission(vertex, -ray.dir, scene);
    }
    int max_depth = scene.options.max_depth;
    for (int num_vertices = 3; max_depth == -1 || num_vertices <= max_depth + 1; num_vertices++) {
        const Material& mat = scene.materials[vertex.material_id];
        // First, we sample a point on the light source.
        // We do this by first picking a light source, then pick a point on it.
        Vector2 light_uv{ next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng) };
        Real light_w = next_pcg32_real<Real>(rng);
        Real shape_w = next_pcg32_real<Real>(rng);
        int light_id = sample_light(scene, light_w);
        const Light& light = scene.lights[light_id];
        PointAndNormal point_on_light =
            sample_point_on_light(light, vertex.position, light_uv, shape_w, scene);

        C1 = make_zero_spectrum();
      
        {
            Real G = 0;
            Vector3 dir_light;
            if (!is_envmap(light)) {
                dir_light = normalize(point_on_light.position - vertex.position);
              
                Ray shadow_ray{ vertex.position, dir_light,
                               get_shadow_epsilon(scene),
                               (1 - get_shadow_epsilon(scene)) *
                                   distance(point_on_light.position, vertex.position) };
                if (!occluded(scene, shadow_ray)) {
                   
                    G = max(-dot(dir_light, point_on_light.normal), Real(0)) /
                        distance_squared(point_on_light.position, vertex.position);
                }
            }
            else {
               
                dir_light = -point_on_light.normal;
               
                Ray shadow_ray{ vertex.position, dir_light,
                               get_shadow_epsilon(scene),
                               infinity<Real>() /* envmaps are infinitely far away */ };
                if (!occluded(scene, shadow_ray)) {

                    G = 1;
                }
            }

          
            p1 = light_pmf(scene, light_id) *
                pdf_point_on_light(light, point_on_light, vertex.position, scene);
            // We don't need to continue the computation if G is 0.
            // Also sometimes there can be some numerical issue such that we generate
            // a light path with probability zero
            if (G > 0 && p1 > 0) {
                Vector3 dir_view = -ray.dir;
                assert(vertex.material_id >= 0);
                Spectrum f = eval(mat, dir_view, dir_light, vertex, scene.texture_pool);

              
                Spectrum L = emission(light, -dir_light, Real(0), point_on_light, scene);

                C1 = G * f * L;

            }
        }

        // Let's do the hemispherical sampling next.
        C2 = make_zero_spectrum();
        Vector3 dir_view = -ray.dir;
        Vector2 bsdf_rnd_param_uv{ next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng) };
        Real bsdf_rnd_param_w = next_pcg32_real<Real>(rng);
        std::optional<BSDFSampleRecord> bsdf_sample_ =
            sample_bsdf(mat,
                dir_view,
                vertex,
                scene.texture_pool,
                bsdf_rnd_param_uv,
                bsdf_rnd_param_w);
        if (!bsdf_sample_) {
            // BSDF sampling failed. Abort the loop.
            break;
        }
        const BSDFSampleRecord& bsdf_sample = *bsdf_sample_;
        Vector3 dir_bsdf = bsdf_sample.dir_out;
        // Update ray differentials & eta_scale
        if (bsdf_sample.eta == 0) {
            ray_diff.spread = reflect(ray_diff, vertex.mean_curvature, bsdf_sample.roughness);
        }
        else {
            ray_diff.spread = refract(ray_diff, vertex.mean_curvature, bsdf_sample.eta, bsdf_sample.roughness);
            eta_scale /= (bsdf_sample.eta * bsdf_sample.eta);
        }

        // Trace a ray towards bsdf_dir. Note that again we have
        // to have an "epsilon" tnear to prevent self intersection.
        Ray bsdf_ray{ vertex.position, dir_bsdf, get_intersection_epsilon(scene), infinity<Real>() };
        std::optional<PathVertex> bsdf_vertex = intersect(scene, bsdf_ray);

        // To update current_path_throughput
        // we need to multiply G(v_{i}, v_{i+1}) * f(v_{i-1}, v_{i}, v_{i+1}) to it
        // and divide it with the pdf for getting v_{i+1} using hemisphere sampling.
        Real G;
        if (bsdf_vertex) {
            G = fabs(dot(dir_bsdf, bsdf_vertex->geometric_normal)) /
                distance_squared(bsdf_vertex->position, vertex.position);
        }
        else {
            // We hit nothing, set G to 1 to account for the environment map contribution.
            G = 1;
        }

        Spectrum f = eval(mat, dir_view, dir_bsdf, vertex, scene.texture_pool);
        p2 = pdf_sample_bsdf(mat, dir_view, dir_bsdf, vertex, scene.texture_pool);
        if (p2 <= 0) {
            // Numerical issue -- we generated some invalid rays.
            break;
        }

        // Remember to convert p2 to area measure!
        p2 *= G;
        // note that G cancels out in the division f/p, but we still need
        // G later for the calculation of w2.
        if (bsdf_vertex && is_light(scene.shapes[bsdf_vertex->shape_id])) {
            // G & f are already computed.
            Spectrum L = emission(*bsdf_vertex, -dir_bsdf, scene);
            C2 = G * f * L;
        }
        else if (!bsdf_vertex && has_envmap(scene)) {
            // G & f are already computed.
            const Light& light = get_envmap(scene);
            Spectrum L = emission(light,
                -dir_bsdf, // pointing outwards from light
                ray_diff.spread,
                PointAndNormal{}, // dummy parameter for envmap
                scene);
            C2 = G * f * L;
        }
        
        //do RIS
        Real W;
        Real m1 = p1 / (p1 + p2);
        Real m2 = p2 / (p1 + p2);
        Real w1 = m1 / p1 * ((C1.x+C1.y+C1.z)/3);
        Real w2 = m2 / p2 * ((C2.x + C2.y + C2.z) / 3);
        if (w1 > w2) {
            W = (w1 + w2) / ((C1.x + C1.y + C1.z) / 3);
            radiance = radiance+current_path_throughput * C1 * W;
            
        }
        else {
            if (w2 > 0) {
                W = (w1 + w2) / ((C2.x + C2.y + C2.z) / 3);
                radiance = radiance + current_path_throughput * C2 * W;
            }
            
        }
        
        

        if (!bsdf_vertex) {
            // Hit nothing -- can't continue tracing.
            break;
        }

        // Update rays/intersection/current_path_throughput/current_pdf
        // Russian roulette heuristics
        Real rr_prob = 1;
        if (num_vertices - 1 >= scene.options.rr_depth) {
            rr_prob = min(max((1 / eta_scale) * current_path_throughput), Real(0.95));
            if (next_pcg32_real<Real>(rng) > rr_prob) {
                // Terminate the path
                break;
            }
        }

        ray = bsdf_ray;
        vertex = *bsdf_vertex;
        current_path_throughput = current_path_throughput * (G * f) / (p2 * rr_prob);
    }
    return radiance;
}