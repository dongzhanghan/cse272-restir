#pragma once
#include "scene.h"
#include "pcg.h"
// The simplest volumetric renderer: 
// single absorption only homogeneous volume
// only handle directly visible light sources

int update_medium(Ray ray, PathVertex vertex, int medium) {
    if (vertex.interior_medium_id != vertex.exterior_medium_id) {
        if (dot(ray.dir, vertex.geometric_normal) > 0) {
            medium = vertex.exterior_medium_id;
        }
        else {
            medium = vertex.interior_medium_id;
        }

    }
    return medium;
}

Spectrum next_event_estimation(const Scene& scene, Vector3 p,int current_medium, pcg32_state& rng, int bounces, Ray ray, bool is_phase, std::optional<PathVertex>  ori_vertex_, RayDifferential ray_diff) {
    Vector2 light_uv{ next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng) };
    Real light_w = next_pcg32_real<Real>(rng);
    Real shape_w = next_pcg32_real<Real>(rng);
    int light_id = sample_light(scene, light_w);
    const Light& light = scene.lights[light_id];
    PointAndNormal p_prime = sample_point_on_light(light, p, light_uv, shape_w, scene);
    Spectrum T_light = fromRGB(Vector3{ 1, 1, 1 });
    int shadow_medium = current_medium;
    int shadow_bounces = 0;
    Spectrum p_trans_dir = fromRGB(Vector3{ 1, 1, 1 });
    Vector3 dir_light = normalize(p_prime.position - p);
    while (true) {
        Ray shadow_ray{ p, dir_light, get_shadow_epsilon(scene),
            (1 - get_shadow_epsilon(scene)) *
            distance(p_prime.position, p)};
        std::optional<PathVertex> vertex_ = intersect(scene, shadow_ray, ray_diff);
        Real next_t = distance(p, p_prime.position);
        if (vertex_) {
            PathVertex vertex = *vertex_;
            next_t = distance(p, vertex.position);
        }
        if (shadow_medium != -1) {
            Spectrum sigma_a = get_sigma_a(scene.media[shadow_medium], shadow_ray.org);
            Spectrum sigma_s = get_sigma_s(scene.media[shadow_medium], shadow_ray.org);
            Spectrum sigma_t = sigma_a + sigma_s;
            T_light *= exp(-sigma_t * next_t);
            p_trans_dir *= exp(-sigma_t * next_t);
        }
        if (!vertex_) {
            break;
        }
        else {
            PathVertex vertex = *vertex_;
            if (vertex.material_id >= 0) {
                return make_zero_spectrum();
            }
            shadow_bounces += 1;
            if ((scene.options.max_depth != -1) && (bounces + shadow_bounces + 1 >= scene.options.max_depth)) return make_zero_spectrum();
            shadow_medium = update_medium(shadow_ray, vertex, shadow_medium);
            p = p + next_t * dir_light;
        }
        
    }
    if (T_light.x > 0) {
        Real G = max(-dot(dir_light, p_prime.normal), Real(0)) /
            distance_squared(p_prime.position, ray.org);
        Spectrum Le = emission(light, -dir_light, Real(0), p_prime, scene);
        if (is_phase) {
            PhaseFunction phase = get_phase_function(scene.media[current_medium]);
            Spectrum rho = eval(phase, -ray.dir, dir_light);
            Real pdf_nee = light_pmf(scene, light_id) *
                pdf_point_on_light(light, p_prime, ray.org, scene);
            Spectrum contrib = T_light * G * rho * Le / pdf_nee;
            Spectrum pdf_phase = pdf_sample_phase(phase, -ray.dir, dir_light) * G * p_trans_dir;
            Spectrum w = (pdf_nee * pdf_nee) / (pdf_nee * pdf_nee + pdf_phase * pdf_phase);
            return  contrib*w;
        }
        else {
            PathVertex vertex = *ori_vertex_;
            const Material& mat = scene.materials[vertex.material_id];
            Spectrum f = eval(mat, -ray.dir, dir_light, vertex, scene.texture_pool);
            Real pdf_nee = light_pmf(scene, light_id) * pdf_point_on_light(light, p_prime, ray.org, scene);
            Spectrum contrib = T_light * G * f * Le / pdf_nee;
            Spectrum pdf_bsdf = pdf_sample_bsdf(
                mat, -ray.dir, dir_light, vertex, scene.texture_pool) * G * p_trans_dir;
            Spectrum w = (pdf_nee * pdf_nee) / (pdf_nee * pdf_nee + pdf_bsdf * pdf_bsdf);
            return  contrib * w;
        }
        
    }
    return make_zero_spectrum();
}

Spectrum next_event_estimation_2(const Scene& scene, Vector3 p, int current_medium, pcg32_state& rng, int bounces, Ray ray, bool is_phase, std::optional<PathVertex>  ori_vertex_, RayDifferential ray_diff) {
    Vector2 light_uv{ next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng) };
    Real light_w = next_pcg32_real<Real>(rng);
    Real shape_w = next_pcg32_real<Real>(rng);
    int light_id = sample_light(scene, light_w);
    const Light& light = scene.lights[light_id];
    PointAndNormal p_prime = sample_point_on_light(light, p, light_uv, shape_w, scene);
    Spectrum T_light = make_const_spectrum(1);
    int shadow_medium = current_medium;
    int shadow_bounces = 0;
    Spectrum p_trans_nee = make_const_spectrum(1);
    Spectrum p_trans_dir = make_const_spectrum(1);
    Vector3 dir_light = normalize(p_prime.position - p);
    while (true) {
        Ray shadow_ray{ p, dir_light, get_shadow_epsilon(scene),
            (1 - get_shadow_epsilon(scene)) *
            distance(p_prime.position, p) };
        std::optional<PathVertex> vertex_ = intersect(scene, shadow_ray, ray_diff);
        Real next_t = distance(p, p_prime.position);
        if (vertex_) {
            PathVertex vertex = *vertex_;
            next_t = distance(p, vertex.position);
            if (shadow_medium != -1) {
                Spectrum majorant = get_majorant(scene.media[shadow_medium], shadow_ray);     
                Real u = next_pcg32_real<Real>(rng);
                int channel = std::clamp(int(u * 3), 0, 2);
                Real accum_t = 0;
                int iteration = 0;
                while (true) {
                    
                    if (majorant[channel] <= 0) break;
                    if (iteration >= scene.options.max_null_collisions) break;
                    
                    Real t = -log(1 - next_pcg32_real<Real>(rng)) / majorant[channel];
                    Real dt = next_t - accum_t;
                    accum_t = min(accum_t + t, next_t);
                    if (t < dt) {
                        Spectrum sigma_a = get_sigma_a(scene.media[shadow_medium], shadow_ray.org + accum_t * shadow_ray.dir);
                        Spectrum sigma_s = get_sigma_s(scene.media[shadow_medium], shadow_ray.org + accum_t * shadow_ray.dir);
                        Spectrum sigma_t = sigma_a + sigma_s;
                        Spectrum sigma_n = majorant - sigma_t;
                        T_light *= exp(-majorant * t) * sigma_n / max(majorant);
                        p_trans_nee *= exp(-majorant * t) * majorant / max(majorant);
                        Spectrum real_prob = sigma_t / majorant;
                        p_trans_dir *= exp(-majorant * t) * majorant * (1 - real_prob) / max(majorant);
                        if (max(T_light) <= 0) return make_zero_spectrum();

                    }
                    else {
                        T_light *= exp(-majorant * dt);
                        p_trans_nee *= exp(-majorant * dt);
                        p_trans_dir *= exp(-majorant * dt);
                        break;
                    }
                    iteration++;
                }
            }
        }
        
        if (!vertex_) {
            break;
        }
        else {
            PathVertex vertex = *vertex_;
            if (vertex.material_id >= 0) {
                return make_zero_spectrum();
            }
            shadow_bounces += 1;
            if ((scene.options.max_depth != -1) && (bounces + shadow_bounces + 1 >= scene.options.max_depth)) return make_zero_spectrum();
            shadow_medium = update_medium(shadow_ray, vertex, shadow_medium);
            p = p + next_t * dir_light;
        }

    }
    if (max(T_light) > 0) {
        Real G = max(-dot(dir_light, p_prime.normal), Real(0)) /
            distance_squared(p_prime.position, ray.org);
        Spectrum Le = emission(light, -dir_light, Real(0), p_prime, scene);
        if (is_phase ) {
            PhaseFunction phase = get_phase_function(scene.media[current_medium]);
            Spectrum rho = eval(phase, -ray.dir, dir_light);
            Spectrum pdf_nee = p_trans_nee*light_pmf(scene, light_id) *
                pdf_point_on_light(light, p_prime, ray.org, scene);
            Spectrum contrib = T_light * G * rho * Le / avg(pdf_nee);
            Spectrum pdf_phase = pdf_sample_phase(phase, -ray.dir, dir_light) * G * p_trans_dir;
            Spectrum w = (pdf_nee * pdf_nee) / (pdf_nee * pdf_nee + pdf_phase * pdf_phase);
            return  contrib * w;
        }
        else {
            PathVertex vertex = *ori_vertex_;
            const Material& mat = scene.materials[vertex.material_id];
            Spectrum f = eval(mat, -ray.dir, dir_light, vertex, scene.texture_pool);
            Spectrum pdf_nee = p_trans_nee*light_pmf(scene, light_id) * pdf_point_on_light(light, p_prime, ray.org, scene);
            Spectrum contrib = T_light * G * f * Le / avg(pdf_nee);
            Spectrum pdf_bsdf = pdf_sample_bsdf(
                mat, -ray.dir, dir_light, vertex, scene.texture_pool) * G * p_trans_dir;
            Spectrum w = (pdf_nee * pdf_nee) / (pdf_nee * pdf_nee + pdf_bsdf * pdf_bsdf);
            return  contrib * w;
        }

    }
    return make_zero_spectrum();
}
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
    //sigma is uniform around the space
    Spectrum sigma_a = get_sigma_a(scene.media[scene.camera.medium_id], ray.org);
    Spectrum sigma_s = get_sigma_s(scene.media[scene.camera.medium_id], ray.org);
    Spectrum sigma_t = sigma_a + sigma_s;
    PhaseFunction phase = get_phase_function(scene.media[scene.camera.medium_id]);
    if (!vertex_) {
        t_hit = infinity<Real>();      
    }
    else {      
        PathVertex vertex = *vertex_;
        t_hit = distance(vertex.position, ray.org);
    }
    Real u = next_pcg32_real<Real>(rng);
    Real t = -log(1 - u) / sigma_t.x;
    if (t < t_hit) {
        Spectrum trans_pdf = exp(-sigma_t * t) * sigma_t;
        Spectrum transmittance = exp(-sigma_t * t);
        Vector3 p = ray.org + t * ray.dir;

        //randomly sample a light source
        Vector2 light_uv{ next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng) };
        Real light_w = next_pcg32_real<Real>(rng);
        Real shape_w = next_pcg32_real<Real>(rng);
        int light_id = sample_light(scene, light_w);
        const Light& light = scene.lights[light_id];
        PointAndNormal point_on_light = sample_point_on_light(light, p, light_uv, shape_w, scene);
        //the pdf of sampling the light source
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
            //evaluate the geometry termS
            G = max(-dot(dir_light, point_on_light.normal), Real(0)) /
                distance_squared(point_on_light.position, p);
        }
        L_s1_estimate = eval(phase, -ray.dir, dir_light) * emission(light, -dir_light, Real(0), point_on_light, scene) * exp(-sigma_t * length(point_on_light.position - p))*G;
        return (transmittance / trans_pdf) * sigma_s * (L_s1_estimate / L_s1_pdf);
    }
    else {
        //sampling distance > t_hit, add surface lighting
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
    int w = scene.camera.width, h = scene.camera.height;
    Vector2 screen_pos((x + next_pcg32_real<Real>(rng)) / w,
        (y + next_pcg32_real<Real>(rng)) / h);
    Ray ray = sample_primary(scene.camera, screen_pos);
    //disable ray differentials
    RayDifferential ray_diff = RayDifferential{ Real(0), Real(0) };
    //start from the camera, the medium is outer medium initially
    int current_medium = scene.camera.medium_id;
    Spectrum current_path_throughput = fromRGB(Vector3{ 1, 1, 1 });
    Spectrum radiance = make_zero_spectrum();
    int bounces = 0;
    int max_depth = scene.options.max_depth;
    while (true){
        bool scatter = false;
        std::optional<PathVertex> vertex_ = intersect(scene, ray, ray_diff);
        Spectrum transmittance = fromRGB(Vector3{ 1, 1, 1 });
        Spectrum trans_pdf = fromRGB(Vector3{ 1, 1, 1 });
        Real t_hit;
        if (current_medium != -1) {
            //we do transmittance sampling if the medium is not vaccum
            Spectrum sigma_a = get_sigma_a(scene.media[current_medium], ray.org);
            Spectrum sigma_s = get_sigma_s(scene.media[current_medium], ray.org);
            Spectrum sigma_t = sigma_a + sigma_s;
            Real u = next_pcg32_real<Real>(rng);
            Real t = -log(1 - u) / sigma_t.x;
            trans_pdf = exp(-sigma_t * t) * sigma_t;
            transmittance = exp(-sigma_t * t);          
            if (!vertex_) 
                t_hit = infinity<Real>();
            else {
                PathVertex vertex = *vertex_;
                t_hit = distance(vertex.position, ray.org);
            }
            if (t < t_hit) {
                scatter = true;
                //update the ray origin if we hit a particle
                ray.org = ray.org + t * ray.dir;
            }
            else {
                trans_pdf = exp(-sigma_t * t_hit);
                transmittance = exp(-sigma_t * t_hit);
                //update the ray origin if we hit a surface/index-matching surface 
                ray.org = ray.org + t_hit * (1 + get_intersection_epsilon(scene))*ray.dir;
            }         
            //redefine the ray to avoid self intersection
            ray = Ray{ ray.org, ray.dir, get_intersection_epsilon(scene), infinity<Real>() };
        }
        else if (t_hit != infinity<Real>()) {
            //if we are in a vacuum but hit something, need to update the ray origin
            ray.org = (*vertex_).position;
            ray = Ray{ ray.org, ray.dir, get_intersection_epsilon(scene), infinity<Real>() };
        }
        current_path_throughput *= (transmittance / trans_pdf);
        if (!scatter && vertex_) {
            //if we hit something except particle(surface/index-matching surface)
            Spectrum Le = make_zero_spectrum();
            PathVertex vertex = *vertex_;
            if (is_light(scene.shapes[vertex.shape_id])) {
                Le = emission(vertex, -ray.dir, scene); //add the surface light if hit a surface
            }
            radiance += current_path_throughput * Le;
        }
        if ((bounces >= max_depth - 1) && (max_depth != -1)) break;
        if (!scatter && vertex_) {
            PathVertex vertex = *vertex_;
            if (vertex.material_id == -1) {
                //if we hit an index-matching surface, update the current medium
                current_medium = update_medium(ray, vertex, current_medium);
                //add the bounces
                bounces += 1;
                //in the above code,we add (1 + get_intersection_epsilon(scene)) to t_hit, without this line
                //we may be traped into an infinite loop if max-depth = -1, every time we sample distance
                // and hit this index-matching surface and we continue
                ray = Ray{ (*vertex_).position, ray.dir, get_intersection_epsilon(scene), infinity<Real>() };
                continue;
            }
        }
        if (scatter) {
            //if we hit a particle, get the phase function
            Spectrum sigma_s = get_sigma_s(scene.media[current_medium], ray.org);
            PhaseFunction phase = get_phase_function(scene.media[current_medium]);
            Vector2 rand_uv{ next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng) };
            //Then we random sample a direction from phase function
            std::optional<Vector3> next_dir_ = sample_phase_function(phase, -ray.dir, rand_uv);
            if (next_dir_) {
                Vector3 next_dir = *next_dir_;
                current_path_throughput *=
                    (eval(phase, -ray.dir, next_dir) / pdf_sample_phase(phase, -ray.dir, next_dir)) * sigma_s;
                ray.dir = next_dir;
            }
            else break;
            
        }
        else break;  //if we hit a surface, terminate 

        Real rr_prob = 1;
        if (bounces >= scene.options.rr_depth) {
            rr_prob = min(max(current_path_throughput), 0.95);
            if (next_pcg32_real<Real>(rng) > rr_prob) {
                break;
            }
            else {
                current_path_throughput /= rr_prob;
            }
        }      
        bounces += 1;
                   
    }
    return radiance;
}

    
// The fourth volumetric renderer: 
// multiple monochromatic homogeneous volumes with multiple scattering
// with MIS between next event estimation and phase function sampling
// still no surface lighting
Spectrum vol_path_tracing_4(const Scene &scene,
                            int x, int y, /* pixel coordinates */
                            pcg32_state &rng) {
    int w = scene.camera.width, h = scene.camera.height;
    Vector2 screen_pos((x + next_pcg32_real<Real>(rng)) / w,
        (y + next_pcg32_real<Real>(rng)) / h);
    Ray ray = sample_primary(scene.camera, screen_pos);
    //disable ray differentials
    RayDifferential ray_diff = RayDifferential{ Real(0), Real(0) };

    int current_medium = scene.camera.medium_id;
    Spectrum current_path_throughput = fromRGB(Vector3{ 1, 1, 1 });
    Spectrum radiance = make_zero_spectrum();
    int bounces = 0;
    Real dir_pdf = 0;
    Vector3 nee_p_cache = Vector3{0,0,0};
    Spectrum multi_trans_pdf = fromRGB(Vector3{ 1, 1, 1 });
    bool never_scatter = true;
    int max_depth = scene.options.max_depth;
    while (true) {
        bool scatter = false;
        RayDifferential ray_diff = RayDifferential{ Real(0), Real(0) };
        std::optional<PathVertex> vertex_ = intersect(scene, ray, ray_diff);
        Spectrum transmittance = fromRGB(Vector3{ 1, 1, 1 });
        Spectrum trans_pdf = fromRGB(Vector3{ 1, 1, 1 });
        Real t_hit;
        if (!vertex_) {
            t_hit = infinity<Real>();
        }
        else {
            PathVertex vertex = *vertex_;
            t_hit = distance(vertex.position, ray.org);
        }
        if (current_medium != -1) {
            Spectrum sigma_a = get_sigma_a(scene.media[current_medium], ray.org);
            Spectrum sigma_s = get_sigma_s(scene.media[current_medium], ray.org);
            Spectrum sigma_t = sigma_a + sigma_s;
            Real u = next_pcg32_real<Real>(rng);
            Real t = -log(1 - u) / sigma_t.x;
            trans_pdf = exp(-sigma_t * t) * sigma_t;
            transmittance = exp(-sigma_t * t);

            
            if (t < t_hit) {
                scatter = true;
                ray.org = ray.org + t * ray.dir;
                never_scatter = false;
            }
            else {
                trans_pdf = exp(-sigma_t * t_hit);
                transmittance = exp(-sigma_t * t_hit);
                ray.org = ray.org + t_hit * (1 + get_intersection_epsilon(scene)) * ray.dir;
            }
            ray = Ray{ ray.org, ray.dir, get_intersection_epsilon(scene), infinity<Real>() };
            
        }
        else if (t_hit != infinity<Real>()) {
            ray.org = (*vertex_).position;
            ray = Ray{ ray.org, ray.dir, get_intersection_epsilon(scene), infinity<Real>() };
        }
        multi_trans_pdf *= trans_pdf;
        current_path_throughput *= (transmittance / trans_pdf);
        if (!scatter && vertex_) {
            PathVertex vertex = *vertex_;
            if (vertex.material_id == -1) {
                current_medium = update_medium(ray, vertex, current_medium);
                bounces += 1;
                ray = Ray{ (*vertex_).position, ray.dir, get_intersection_epsilon(scene), infinity<Real>() };
                continue;
            }
        }
        if (!scatter && vertex_) {
            PathVertex vertex = *vertex_;
            if (is_light(scene.shapes[vertex.shape_id])) {
                if (never_scatter) {
                    Spectrum Le = emission(vertex, -ray.dir, scene);
                    radiance += current_path_throughput * Le;
                }
                else {
                    PointAndNormal light_point;
                    light_point.position = vertex.position;
                    light_point.normal = vertex.geometric_normal;
                    int light_id = get_area_light_id(scene.shapes[vertex.shape_id]);
                    assert(light_id >= 0);
                    const Light& light = scene.lights[light_id];
                    Real pdf_nee = pdf_point_on_light(light, light_point, nee_p_cache, scene);
                    Real G = max(-dot(ray.dir, light_point.normal), Real(0)) /
                        distance_squared(light_point.position, nee_p_cache);
                    Spectrum dir_pdf_ = dir_pdf * multi_trans_pdf * G;
                    Spectrum w = (dir_pdf_ * dir_pdf_) / (dir_pdf_ * dir_pdf_ + pdf_nee * pdf_nee);
                    radiance += current_path_throughput * emission(vertex, -ray.dir, scene) * w;

                }
            }
            else {
                Vector3 p = ray.org; // current position
                nee_p_cache = p;
                never_scatter = false;
                Spectrum nee = next_event_estimation(scene, p,  current_medium, rng, bounces, ray,true,vertex_,ray_diff);
                radiance += current_path_throughput * nee;
            }
                     
        }
        if ((bounces == max_depth - 1) && (max_depth != -1)) break;
       
        if (scatter) {
            never_scatter = false;
            nee_p_cache = ray.org;
            Spectrum nee = next_event_estimation(scene,ray.org,current_medium, rng, bounces,ray,true, vertex_,ray_diff);
            Spectrum sigma_s = get_sigma_s(scene.media[current_medium], ray.org);
            radiance += current_path_throughput * nee * sigma_s;
            PhaseFunction phase = get_phase_function(scene.media[current_medium]);         
            Vector2 rand_uv{ next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng) };
            std::optional<Vector3> next_dir_ = sample_phase_function(phase, -ray.dir, rand_uv);
            if (next_dir_) {
                Vector3 next_dir = *next_dir_;
                dir_pdf = pdf_sample_phase(phase, -ray.dir, next_dir);
                current_path_throughput *= (eval(phase, -ray.dir, next_dir) / dir_pdf) * sigma_s;
                ray.dir = next_dir;
                multi_trans_pdf = make_const_spectrum(1);
            }
            else {

                break;
            }

        }
        else {
            break;
        }
        Real rr_prob = 1;
        if (bounces >= scene.options.rr_depth) {
            rr_prob = min(max(current_path_throughput), 0.95);
            if (next_pcg32_real<Real>(rng) > rr_prob) {
                break;
            }
            else {
                current_path_throughput /= rr_prob;
            }
        }
        bounces += 1;

    }
    return radiance;
}

// The fifth volumetric renderer: 
// multiple monochromatic homogeneous volumes with multiple scattering
// with MIS between next event estimation and phase function sampling
// with surface lighting
Spectrum vol_path_tracing_5(const Scene& scene,
    int x, int y, /* pixel coordinates */
    pcg32_state& rng) {
    int w = scene.camera.width, h = scene.camera.height;
    Vector2 screen_pos((x + next_pcg32_real<Real>(rng)) / w,
        (y + next_pcg32_real<Real>(rng)) / h);
    Ray ray = sample_primary(scene.camera, screen_pos);
    //disable ray differentials
    RayDifferential ray_diff = RayDifferential{ Real(0), Real(0) };

    int current_medium = scene.camera.medium_id;
    Spectrum current_path_throughput = fromRGB(Vector3{ 1, 1, 1 });
    Spectrum radiance = make_zero_spectrum();
    int bounces = 0;
    Real dir_pdf = 0;
    Vector3 nee_p_cache = Vector3{ 0,0,0 };
    Spectrum multi_trans_pdf = fromRGB(Vector3{ 1, 1, 1 });
    bool never_scatter = true;
    int max_depth = scene.options.max_depth;
    Real eta_scale = 1;
    while (true) {
        bool scatter = false;
        std::optional<PathVertex> vertex_ = intersect(scene, ray, ray_diff);
        Spectrum transmittance = fromRGB(Vector3{ 1, 1, 1 });
        Spectrum trans_pdf = fromRGB(Vector3{ 1, 1, 1 });
        Real t_hit;
        if (!vertex_) {
            t_hit = infinity<Real>();
        }
        else {
            PathVertex vertex = *vertex_;
            t_hit = distance(vertex.position, ray.org);
        }
        if (current_medium != -1) {
            Spectrum sigma_a = get_sigma_a(scene.media[current_medium], ray.org);
            Spectrum sigma_s = get_sigma_s(scene.media[current_medium], ray.org);
            Spectrum sigma_t = sigma_a + sigma_s;
            Real u = next_pcg32_real<Real>(rng);
            Real t = -log(1 - u) / sigma_t.x;
            trans_pdf = exp(-sigma_t * t) * sigma_t;
            transmittance = exp(-sigma_t * t);


            if (t < t_hit) {
                scatter = true;
                ray.org = ray.org + t * ray.dir;
                never_scatter = false;
            }
            else {
                trans_pdf = exp(-sigma_t * t_hit);
                transmittance = exp(-sigma_t * t_hit);
                ray.org = ray.org + t_hit * (1 + get_intersection_epsilon(scene)) * ray.dir;
            }
            ray = Ray{ ray.org, ray.dir, get_intersection_epsilon(scene), infinity<Real>() };

        }
        else if (t_hit != infinity<Real>()) {
            ray.org = (*vertex_).position;
            ray = Ray{ ray.org, ray.dir, get_intersection_epsilon(scene), infinity<Real>() };
        }
        multi_trans_pdf *= trans_pdf;
        current_path_throughput *= (transmittance / trans_pdf);
        if (!scatter && vertex_) {
            PathVertex vertex = *vertex_;
            if (vertex.material_id == -1) {
                current_medium = update_medium(ray, vertex, current_medium);
                bounces += 1;
                ray = Ray{ (*vertex_).position, ray.dir, get_intersection_epsilon(scene), infinity<Real>() };
                continue;
            }
        }
        if (!scatter && vertex_) {
            PathVertex vertex = *vertex_;
            if (is_light(scene.shapes[vertex.shape_id])) {
                if (never_scatter) {
                    Spectrum Le = emission(vertex, -ray.dir, scene);
                    radiance += current_path_throughput * Le;
                }
                else {
                    PointAndNormal light_point;
                    light_point.position = vertex.position;
                    light_point.normal = vertex.geometric_normal;
                    int light_id = get_area_light_id(scene.shapes[vertex.shape_id]);
                    assert(light_id >= 0);
                    const Light& light = scene.lights[light_id];
                    Real pdf_nee = pdf_point_on_light(light, light_point, nee_p_cache, scene);
                    Real G = max(-dot(ray.dir, light_point.normal), Real(0)) /
                        distance_squared(light_point.position, nee_p_cache);
                    Spectrum dir_pdf_ = dir_pdf * multi_trans_pdf * G;
                    Spectrum w = (dir_pdf_ * dir_pdf_) / (dir_pdf_ * dir_pdf_ + pdf_nee * pdf_nee);
                    radiance += current_path_throughput * emission(vertex, -ray.dir, scene) * w;

                }
            }
            

        }
        if ((bounces >= max_depth - 1) && (max_depth != -1)) break;

        if (scatter) {
            never_scatter = false;
            nee_p_cache = ray.org;
            Spectrum nee = next_event_estimation(scene, ray.org, current_medium, rng, bounces, ray, true, vertex_,ray_diff);
            Spectrum sigma_s = get_sigma_s(scene.media[current_medium], ray.org);
            radiance += current_path_throughput * nee * sigma_s;
            PhaseFunction phase = get_phase_function(scene.media[current_medium]);
            Vector2 rand_uv{ next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng) };
            std::optional<Vector3> next_dir_ = sample_phase_function(phase, -ray.dir, rand_uv);
            if (next_dir_) {
                Vector3 next_dir = *next_dir_;
                dir_pdf = pdf_sample_phase(phase, -ray.dir, next_dir);
                current_path_throughput *= (eval(phase, -ray.dir, next_dir) / dir_pdf) * sigma_s;
                ray.dir = next_dir;
                ray = Ray{ ray.org, ray.dir, get_intersection_epsilon(scene), infinity<Real>() };
                multi_trans_pdf = make_const_spectrum(1);
            }
            else {

                break;
            }

        }
        else if (vertex_){
            PathVertex vertex = *vertex_;
            Vector3 p = vertex.position; // current position
            nee_p_cache = p;
            never_scatter = false;
            Spectrum nee = next_event_estimation(scene, p, current_medium, rng, bounces, ray, false, vertex_, ray_diff);
            radiance += current_path_throughput * nee;
            Vector3 dir_view = -ray.dir;
            Vector2 bsdf_rnd_param_uv{ next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng) };
            Real bsdf_rnd_param_w = next_pcg32_real<Real>(rng);
            Material mat = scene.materials[vertex.material_id];
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
                current_medium = update_medium(ray, vertex, current_medium);
            }
            // Trace a ray towards bsdf_dir. Note that again we have
            // to have an "epsilon" tnear to prevent self intersection.
            dir_pdf = pdf_sample_bsdf(mat, dir_view, dir_bsdf, vertex, scene.texture_pool);
            Spectrum bsdf_f = eval(mat, dir_view, dir_bsdf, vertex, scene.texture_pool);
            current_path_throughput *= bsdf_f / dir_pdf;
            ray.dir = dir_bsdf;
            ray = Ray{ vertex.position, dir_bsdf, get_intersection_epsilon(scene), infinity<Real>() };
            multi_trans_pdf = make_const_spectrum(1);
        }
        Real rr_prob = 1;
        if (bounces >= scene.options.rr_depth) {
            rr_prob = min(max((1 / eta_scale) * current_path_throughput), 0.95);
            if (next_pcg32_real<Real>(rng) > rr_prob) {
                break;
            }
            else {
                current_path_throughput /= rr_prob;
            }
        }
        bounces += 1;

    }
    return radiance;
}

// The final volumetric renderer: 
// multiple chromatic heterogeneous volumes with multiple scattering
// with MIS between next event estimation and phase function sampling
// with surface lighting
Spectrum vol_path_tracing(const Scene& scene,
    int x, int y, /* pixel coordinates */
    pcg32_state& rng) {
    int w = scene.camera.width, h = scene.camera.height;
    Vector2 screen_pos((x + next_pcg32_real<Real>(rng)) / w,
        (y + next_pcg32_real<Real>(rng)) / h);
    Ray ray = sample_primary(scene.camera, screen_pos);
    //std::cout << scene.options.max_null_collisions << std::endl;
    //disable ray differentials
    int current_medium = scene.camera.medium_id;
    Spectrum current_path_throughput = make_const_spectrum(1);
    Spectrum radiance = make_zero_spectrum();
    int bounces = 0;
    Real dir_pdf = 0;
    Vector3 nee_p_cache = Vector3{ 0,0,0 };
    Spectrum multi_trans_pdf = make_const_spectrum(1);
    bool never_scatter = true;
    int max_depth = scene.options.max_depth;
    Real eta_scale = 1;
    RayDifferential ray_diff = RayDifferential{ Real(0), Real(0) };
    while (true) {
        
        bool scatter = false;
       
        std::optional<PathVertex> vertex_ = intersect(scene, ray, ray_diff);
        Spectrum transmittance = make_const_spectrum(1);
        Spectrum trans_dir_pdf = make_const_spectrum(1);
        Spectrum trans_nee_pdf = make_const_spectrum(1);
        Real t_hit;
        if (!vertex_) {
            t_hit = infinity<Real>();
        }
        else {
            PathVertex vertex = *vertex_;
            t_hit = distance(vertex.position, ray.org);
        }
        if (current_medium != -1) {
            Spectrum majorant = get_majorant(scene.media[current_medium], ray);
           
            Real u = next_pcg32_real<Real>(rng);
            int channel = std::clamp(int(u * 3), 0, 2);
            Real accum_t = 0;
            int iteration = 0;

            while (true) {
                if (majorant[channel] <= 0) break;
                if (iteration >= scene.options.max_null_collisions) break;
                
                Real t = -log(1 - next_pcg32_real<Real>(rng)) / majorant[channel];
                Real dt = t_hit - accum_t;
                //Update accumulated distance
                accum_t = min(accum_t + t, t_hit);
                if (t < dt) {
                    Spectrum sigma_a = get_sigma_a(scene.media[current_medium], ray.org + accum_t * ray.dir);
                    Spectrum sigma_s = get_sigma_s(scene.media[current_medium], ray.org + accum_t * ray.dir);
                    Spectrum sigma_t = sigma_a + sigma_s;
                    Spectrum sigma_n = majorant - sigma_t;
                    Spectrum real_prob = sigma_t / majorant;
                    if (next_pcg32_real<Real>(rng) < real_prob[channel]) {
                        scatter = true;
                        never_scatter = false;
                        transmittance *= exp(-majorant * t) / max(majorant);
                        trans_dir_pdf *= exp(-majorant * t) * majorant * real_prob / max(majorant);
                        ray.org = ray.org + accum_t * ray.dir;
                        ray = Ray{ ray.org, ray.dir, get_intersection_epsilon(scene), infinity<Real>() };
                        break;
                    }
                    else {
                        transmittance *= exp(-majorant * t) * sigma_n / max(majorant);
                        trans_dir_pdf *= exp(-majorant * t) * majorant * (1 - real_prob) / max(majorant);
                        trans_nee_pdf *= exp(-majorant * t) * majorant / max(majorant);
                    }
                }
                else {
                    transmittance *= exp(-majorant * dt);
                    trans_dir_pdf *= exp(-majorant * dt);
                    trans_nee_pdf *= exp(-majorant * dt);
                    ray.org = ray.org + t_hit * ray.dir;
                    ray = Ray{ ray.org, ray.dir, get_intersection_epsilon(scene), infinity<Real>() };
                    break;
                }
                iteration += 1;
            }       

        }
        else if (t_hit != infinity<Real>()) {
            ray.org = (*vertex_).position;
            ray = Ray{ ray.org, ray.dir, get_intersection_epsilon(scene), infinity<Real>() };
        }
        multi_trans_pdf *= trans_dir_pdf;
        current_path_throughput *= (transmittance / avg(trans_dir_pdf));
        if ((bounces >= max_depth - 1) && (max_depth != -1)) break;
        if (!scatter && vertex_) {
            PathVertex vertex = *vertex_;
            if (vertex.material_id == -1) {
                current_medium = update_medium(ray, vertex, current_medium);
                ray = Ray{ (*vertex_).position, ray.dir, get_intersection_epsilon(scene), infinity<Real>() };
                bounces += 1;
                continue;
            }
        }
        
        if (!scatter && vertex_) {
            PathVertex vertex = *vertex_;
            if (is_light(scene.shapes[vertex.shape_id])) {
                if (never_scatter) {
                    Spectrum Le = emission(vertex, -ray.dir, scene);
                    radiance += current_path_throughput * Le;
                }
                else {
                    PointAndNormal light_point;
                    light_point.position = vertex.position;
                    light_point.normal = vertex.geometric_normal;
                    int light_id = get_area_light_id(scene.shapes[vertex.shape_id]);
                    assert(light_id >= 0);
                    const Light& light = scene.lights[light_id];
                    Spectrum pdf_nee = trans_nee_pdf*pdf_point_on_light(light, light_point, nee_p_cache, scene);
                    Real G = max(-dot(ray.dir, light_point.normal), Real(0)) /
                        distance_squared(light_point.position, nee_p_cache);
                    Spectrum dir_pdf_ = dir_pdf * multi_trans_pdf * G;
                    Spectrum w = (dir_pdf_ * dir_pdf_) / (dir_pdf_ * dir_pdf_ + pdf_nee * pdf_nee);
                    radiance += current_path_throughput * emission(vertex, -ray.dir, scene) * w;

                }
            }


        }
        

        if (scatter) {
            never_scatter = false;
            nee_p_cache = ray.org;
            Spectrum nee = next_event_estimation_2(scene, ray.org, current_medium, rng, bounces, ray, true, vertex_, ray_diff);
            Spectrum sigma_s = get_sigma_s(scene.media[current_medium], ray.org);
            radiance += current_path_throughput * nee * sigma_s;
            PhaseFunction phase = get_phase_function(scene.media[current_medium]);
            Vector2 rand_uv{ next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng) };
            std::optional<Vector3> next_dir_ = sample_phase_function(phase, -ray.dir, rand_uv);
            if (next_dir_) {
                Vector3 next_dir = *next_dir_;
                dir_pdf = pdf_sample_phase(phase, -ray.dir, next_dir);
                current_path_throughput *= (eval(phase, -ray.dir, next_dir) / dir_pdf) * sigma_s;
                ray.dir = next_dir;
                ray = Ray{ ray.org, ray.dir, get_intersection_epsilon(scene), infinity<Real>() };
                multi_trans_pdf = make_const_spectrum(1);
            }
            else {

                break;
            }

        }
        else if (vertex_) {
            PathVertex vertex = *vertex_;
            Vector3 p = vertex.position; // current position
            nee_p_cache = p;
            never_scatter = false;
            Spectrum nee = next_event_estimation_2(scene, p, current_medium, rng, bounces, ray, false, vertex_, ray_diff);
            radiance += current_path_throughput * nee;
            Vector3 dir_view = -ray.dir;
            Vector2 bsdf_rnd_param_uv{ next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng) };
            Real bsdf_rnd_param_w = next_pcg32_real<Real>(rng);
            Material mat = scene.materials[vertex.material_id];
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
                current_medium = update_medium(ray, vertex, current_medium);
            }
            // Trace a ray towards bsdf_dir. Note that again we have
            // to have an "epsilon" tnear to prevent self intersection.
            dir_pdf = pdf_sample_bsdf(mat, dir_view, dir_bsdf, vertex, scene.texture_pool);
            if (dir_pdf <= 0) {
                // Numerical issue -- we generated some invalid rays.
                break;
            }
            Spectrum bsdf_f = eval(mat, dir_view, dir_bsdf, vertex, scene.texture_pool);
            current_path_throughput *= bsdf_f / dir_pdf;
            ray.dir = dir_bsdf;
            ray = Ray{ vertex.position, dir_bsdf, get_intersection_epsilon(scene), infinity<Real>() };
            multi_trans_pdf = make_const_spectrum(1);
        }
        Real rr_prob = 1;
        if (bounces >= scene.options.rr_depth) {
            rr_prob = min(max((1 / eta_scale) * current_path_throughput), (Real)0.95);
            if (next_pcg32_real<Real>(rng) > rr_prob) {
                break;
            }
            else {
                current_path_throughput /= rr_prob;
            }
        }
        bounces += 1;

    }
    return radiance;
}


