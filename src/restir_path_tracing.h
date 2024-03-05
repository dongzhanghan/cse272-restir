#pragma once
#include "scene.h"
#include "pcg.h"
#include "reservoir.h"


void pickSpatialNeighbors(const Scene& scene,pcg32_state& rng, int x, int y,int R, int N, ImageReservoir& imgReservoir, std::vector<Reservoir>& reservoirs) {
    Real w = scene.camera.width;
    Real h = scene.camera.height;
    for (int i = 0; i < N; i++) {
        Real u = next_pcg32_real<Real>(rng);
        Real r = std::sqrt(u);  // Radius
        Real theta = 2 * c_PI * u;  // Angle in radians

        // Convert polar coordinates to Cartesian coordinates
        Real delta_x = r * std::cos(theta);
        Real delta_y = r * std::sin(theta);

        int x1 = std::round(max(min(x+ R*delta_x, w), (Real)0));
        int y1 = std::round(max(min(y+ R*delta_y, h), (Real)0));
        reservoirs.push_back(imgReservoir(x1, y1));
    }

}

Spectrum target_function(const Scene& scene,const PathVertex& vertex,const Ray& ray, const Sample& sample) {
    Spectrum C1 = make_zero_spectrum();
    Real G = 0;
    Vector3 dir_light;
    const Light& light = scene.lights[sample.light_id];
    
    if (!is_envmap(light)) {
        dir_light = normalize(sample.point_on_light.position - vertex.position);

        Ray shadow_ray{ vertex.position, dir_light,
                        get_shadow_epsilon(scene),
                        (1 - get_shadow_epsilon(scene)) *
                            distance(sample.point_on_light.position, vertex.position) };
        if (!occluded(scene, shadow_ray)) {

            G = max(-dot(dir_light, sample.point_on_light.normal), Real(0)) /
                distance_squared(sample.point_on_light.position, vertex.position);
        }
    }
    else {

        dir_light = -sample.point_on_light.normal;

        Ray shadow_ray{ vertex.position, dir_light,
                        get_shadow_epsilon(scene),
                        infinity<Real>() /* envmaps are infinitely far away */ };
        if (!occluded(scene, shadow_ray)) {

            G = 1;
        }
    };
    Real p1 = light_pmf(scene, sample.light_id) *
        pdf_point_on_light(light, sample.point_on_light, vertex.position, scene);
    
    if (G > 0 && p1 > 0) {
        
        assert(vertex.material_id >= 0);
        const Material& mat = scene.materials[vertex.material_id];
       
        Vector3 dir_view = -ray.dir;
        
        Spectrum f = eval(mat, dir_view, dir_light, vertex, scene.texture_pool);


        Spectrum L = emission(light, -dir_light, Real(0), sample.point_on_light, scene);

        C1 = G * f * L;
    }
    return C1;
}

void RIS(const Scene& scene,
    pcg32_state& rng,const PathVertex& vertex, Ray& ray, Reservoir& r) {
    int M = scene.options.reservoirNumber;
    const Material& mat = scene.materials[vertex.material_id];
    Real total_p1 = 0;
    for (int i = 0; i < M; i++) {
        Vector2 light_uv{ next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng) };
        Real light_w = next_pcg32_real<Real>(rng);
        Real shape_w = next_pcg32_real<Real>(rng);
        int light_id = sample_light(scene, light_w);
        assert(light_id >= 0);
        const Light& light = scene.lights[light_id];
        
        PointAndNormal point_on_light =
            sample_point_on_light(light, vertex.position, light_uv, shape_w, scene);

        Spectrum C = target_function(scene, vertex, ray, {point_on_light, light_id});
        Real p1 = light_pmf(scene, light_id) *
            pdf_point_on_light(light, point_on_light, vertex.position, scene);
        total_p1 += p1;
        
        Real w = luminance(C);
        //Real w = luminance(C) / (M * p1);
        Sample x = { point_on_light, light_id };
        if (w > 0) {
            r.update(x, w);
        }
            
    }
    if (r.M > 0) {
        Spectrum C = target_function(scene, vertex, ray, r.sample);
        r.W = r.w_sum / (luminance(C)* total_p1);
        //r.W = r.w_sum / luminance(C);
    }
}

void combineReservoirs(const Scene& scene,
    PathVertex & vertex,
    Ray& ray,std::vector<Reservoir> &reservoirs, Reservoir& s) {
    for (Reservoir& r : reservoirs) {  
        if (r.W != 0)
            s.update(r.sample, luminance(target_function(scene, vertex, ray, r.sample)) * r.W * r.M);         
    }
    s.M = 0;
    for (Reservoir& r : reservoirs) { 
         s.M += r.M;                 
    }
    
    if (s.sample.light_id >= 0 ) {
        s.W = s.w_sum / (s.M * luminance(target_function(scene, vertex, ray, s.sample)));
    }  
}


void combineReservoirsUnbiased(const Scene& scene,
    PathVertex& vertex,
    Ray& ray, std::vector<Reservoir>& reservoirs, Reservoir& s) {
    for (Reservoir& r : reservoirs) {
        if (r.W != 0) {
            s.update(r.sample, luminance(target_function(scene, vertex, ray, r.sample)) * r.W * r.M);

        }


    }
    s.M = 0;
    Real z = 0;
    for (Reservoir& r : reservoirs) {
        s.M += r.M;
        if (r.W > 0)
            z += r.M;
    }

    if (s.sample.light_id >= 0 ) {
        s.W = s.w_sum / (z * luminance(target_function(scene, vertex, ray, s.sample)));
    }
}




Spectrum restir_path_tracing_1(const Scene& scene,
    int x, int y, std::optional<PathVertex> vertex_, Ray& ray, /* pixel coordinates */
    pcg32_state& rng, ImageReservoir& imgReservoir) {
    int w = scene.camera.width, h = scene.camera.height;
    RayDifferential ray_diff = init_ray_differential(w, h); 
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
        if (num_vertices == 3) {
            std::vector<Reservoir> reservoirs;
            reservoirs.push_back(imgReservoir(x, y));
            pickSpatialNeighbors(scene, rng, x, y, 30, 3, imgReservoir, reservoirs);
            for (Reservoir& r_near : reservoirs) {
                int light_id = r_near.sample.light_id;
                PointAndNormal point_on_light = r_near.sample.point_on_light;
                if (light_id == -1 ||luminance(target_function(scene, vertex, ray, r_near.sample)) == 0)
                {
                    r_near.W = 0;
                }
            }
            Reservoir r(next_pcg32_real<Real>(rng));

            //combineReservoirs(scene, vertex, ray, reservoirs, r);

            //combineReservoirsUnbiased(scene, vertex, ray, reservoirs, r);

            RIS(scene, rng, vertex, ray, r);
            Spectrum C1 = make_zero_spectrum();
            Real w1 = 0;
            Real G = 0;
            Vector3 dir_light;
            int light_id = r.sample.light_id;
            PointAndNormal point_on_light = r.sample.point_on_light;
            if (light_id >= 0) {
                Light light = scene.lights[light_id];
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
                
                Real p1 = light_pmf(scene, light_id) *
                    pdf_point_on_light(light, r.sample.point_on_light, vertex.position, scene);

                if (G > 0 && p1 > 0) {
                    Vector3 dir_view = -ray.dir;
                    assert(vertex.material_id >= 0);
                    Spectrum f = eval(mat, dir_view, dir_light, vertex, scene.texture_pool);
                    
                    Spectrum L = emission(light, -dir_light, Real(0), r.sample.point_on_light, scene);

                    C1 = G * f * L;
                    Real p2 = pdf_sample_bsdf(
                        mat, dir_view, dir_light, vertex, scene.texture_pool);
                    p2 *= G;

                    w1 = (p1 * p1) / (p1 * p1 + p2 * p2);
                    radiance += current_path_throughput * C1 * r.W * w1;
                }
            }
            
        }
        else {
            Vector2 light_uv{ next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng) };
            Real light_w = next_pcg32_real<Real>(rng);
            Real shape_w = next_pcg32_real<Real>(rng);
            int light_id = sample_light(scene, light_w);
            const Light& light = scene.lights[light_id];
            PointAndNormal point_on_light =
                sample_point_on_light(light, vertex.position, light_uv, shape_w, scene);
            Spectrum C1 = make_zero_spectrum();
            Real w1 = 0;
        
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
                Real p1 = light_pmf(scene, light_id) *
                    pdf_point_on_light(light, point_on_light, vertex.position, scene);

                if (G > 0 && p1 > 0) {
                    Vector3 dir_view = -ray.dir;
                    assert(vertex.material_id >= 0);
                    Spectrum f = eval(mat, dir_view, dir_light, vertex, scene.texture_pool);
                    Spectrum L = emission(light, -dir_light, Real(0), point_on_light, scene);

                    C1 = G * f * L;

                    Real p2 = pdf_sample_bsdf(
                        mat, dir_view, dir_light, vertex, scene.texture_pool);

                    p2 *= G;

                    w1 = (p1 * p1) / (p1 * p1 + p2 * p2);
                    C1 /= p1;
                    }
                }
            radiance += current_path_throughput * C1 * w1;
        }


        // Let's do the hemispherical sampling next.
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
            break;
        }
        const BSDFSampleRecord& bsdf_sample = *bsdf_sample_;
        Vector3 dir_bsdf = bsdf_sample.dir_out;
        if (bsdf_sample.eta == 0) {
            ray_diff.spread = reflect(ray_diff, vertex.mean_curvature, bsdf_sample.roughness);
        }
        else {
            ray_diff.spread = refract(ray_diff, vertex.mean_curvature, bsdf_sample.eta, bsdf_sample.roughness);
            eta_scale /= (bsdf_sample.eta * bsdf_sample.eta);
        }
        Ray bsdf_ray{ vertex.position, dir_bsdf, get_intersection_epsilon(scene), infinity<Real>() };
        std::optional<PathVertex> bsdf_vertex = intersect(scene, bsdf_ray);

        Real G = 0;
        if (bsdf_vertex) {
            G = fabs(dot(dir_bsdf, bsdf_vertex->geometric_normal)) /
                distance_squared(bsdf_vertex->position, vertex.position);
        }
        else {
            G = 1;
        }

        Spectrum f = eval(mat, dir_view, dir_bsdf, vertex, scene.texture_pool);
        Real p2 = pdf_sample_bsdf(mat, dir_view, dir_bsdf, vertex, scene.texture_pool);
        if (p2 <= 0) {
            break;
        }

        // Remember to convert p2 to area measure!
        p2 *= G;
        if (bsdf_vertex && is_light(scene.shapes[bsdf_vertex->shape_id])) {

            Spectrum L = emission(*bsdf_vertex, -dir_bsdf, scene);
            Spectrum C2 = G * f * L;
            int light_id = get_area_light_id(scene.shapes[bsdf_vertex->shape_id]);
            assert(light_id >= 0);
            const Light& light = scene.lights[light_id];
            PointAndNormal light_point{ bsdf_vertex->position, bsdf_vertex->geometric_normal };
            Real p1 = light_pmf(scene, light_id) *
                pdf_point_on_light(light, light_point, vertex.position, scene);
            Real w2 = (p2 * p2) / (p1 * p1 + p2 * p2);

            C2 /= p2;
            radiance += current_path_throughput * C2 * w2;
        }
        else if (!bsdf_vertex && has_envmap(scene)) {
            const Light& light = get_envmap(scene);
            Spectrum L = emission(light,
                -dir_bsdf, // pointing outwards from light
                ray_diff.spread,
                PointAndNormal{}, // dummy parameter for envmap
                scene);
            Spectrum C2 = G * f * L;
            PointAndNormal light_point{ Vector3{0, 0, 0}, -dir_bsdf }; // pointing outwards from light
            Real p1 = light_pmf(scene, scene.envmap_light_id) *
                pdf_point_on_light(light, light_point, vertex.position, scene);
            Real w2 = (p2 * p2) / (p1 * p1 + p2 * p2);

            C2 /= p2;
            radiance += current_path_throughput * C2 * w2;
        }

        if (!bsdf_vertex) {
            break;
        }

        
        // Update rays/intersection/current_path_throughput/current_pdf
        // Russian roulette heuristics
        Real rr_prob = 1;
        if (num_vertices - 1 >= scene.options.rr_depth) {
            rr_prob = min(max((1 / eta_scale) * current_path_throughput), Real(0.95));
            if (next_pcg32_real<Real>(rng) > rr_prob) {
                break;
            }
        }

        ray = bsdf_ray;
        vertex = *bsdf_vertex;
        current_path_throughput = current_path_throughput * (G * f) / (p2 * rr_prob);
    }
    return radiance;
}