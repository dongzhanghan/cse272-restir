#include "../microfacet.h"

Spectrum eval_op::operator()(const DisneyGlass &bsdf) const {
    bool reflect = dot(vertex.geometric_normal, dir_in) *
                   dot(vertex.geometric_normal, dir_out) > 0;
    // If we are going into the surface, then we use normal eta
    // (internal/external), otherwise we use external/internal.
    Real eta = dot(vertex.geometric_normal, dir_in) > 0 ? bsdf.eta : 1 / bsdf.eta;
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) * dot(vertex.geometric_normal, dir_in) < 0) {
        frame = -frame;
    }  
    // Homework 1: implement this!
    Spectrum base_color = eval(
        bsdf.base_color, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real anisotropic = eval(
        bsdf.anisotropic, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real roughness = eval(
        bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    Vector3 h;
    if (reflect) {
        h = normalize(dir_in + dir_out);
    }
    else {
        // "Generalized half-vector" from Walter et al.
        // See "Microfacet Models for Refraction through Rough Surfaces"
        h = normalize(dir_in + dir_out * eta);
    }
    // Flip half-vector if it's below surface
    if (dot(h, frame.n) < 0) {
        h = -h;
    }
    Real h_dot_in = dot(h, dir_in);
    Real F_g = fresnel_dielectric(h_dot_in, eta);

    Real alpha_min = 0.0001;
    Real aspect = sqrt(1 - 0.9 * anisotropic);
    Real alpha_x = max(alpha_min, roughness * roughness / aspect);
    Real alpha_y = max(alpha_min, roughness * roughness * aspect);

    Real Delta_in = 0.5 * (-1 + sqrt(1 + 1 / (pow(to_local(frame, dir_in).z, 2)) * (pow(to_local(frame, dir_in).x * alpha_x, 2) + pow(to_local(frame, dir_in).y * alpha_y, 2))));
    Real Delta_out = 0.5 * (-1 + sqrt(1 + 1 / (pow(to_local(frame, dir_out).z, 2)) * (pow(to_local(frame, dir_out).x * alpha_x, 2) + pow(to_local(frame, dir_out).y * alpha_y, 2))));
    Real G_in = 1 / (1 + Delta_in);
    Real G_out = 1 / (1 + Delta_out);
    Real G_g = G_in * G_out;

    Real D_g = 1 / (c_PI * alpha_x * alpha_y * pow((pow((to_local(frame, h).x) / alpha_x, 2) + pow((to_local(frame, h).y) / alpha_y, 2) + pow(to_local(frame, h).z, 2)), 2));

    Spectrum f_glass_reflected = base_color * F_g * D_g * G_g / (4 * abs(dot(frame.n, dir_in)));
    Spectrum f_glass_refracted = sqrt(base_color) * (1 - F_g) * D_g * G_g * abs(dot(h, dir_in) * dot(h, dir_out)) / (abs(dot(frame.n, dir_in)) * pow(dot(h,dir_in)+eta*dot(h,dir_out), 2));
    if (reflect) {
        return f_glass_reflected;
    }
    // Burley propose to take the square root of the base color to preserve albedo
    return f_glass_refracted;
}

Real pdf_sample_bsdf_op::operator()(const DisneyGlass &bsdf) const {
    bool reflect = dot(vertex.geometric_normal, dir_in) *
                   dot(vertex.geometric_normal, dir_out) > 0;
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) * dot(vertex.geometric_normal, dir_in) < 0) {
        frame = -frame;
    }
    // If we are going into the surface, then we use normal eta
   // (internal/external), otherwise we use external/internal.
    Real eta = dot(vertex.geometric_normal, dir_in) > 0 ? bsdf.eta : 1 / bsdf.eta;

    Vector3 half_vector;
    if (reflect) {
        half_vector = normalize(dir_in + dir_out);
    }
    else {
        // "Generalized half-vector" from Walter et al.
        // See "Microfacet Models for Refraction through Rough Surfaces"
        half_vector = normalize(dir_in + dir_out * eta);
    }

    // Flip half-vector if it's below surface, why ?????????
    if (dot(half_vector, frame.n) < 0) {
        half_vector = -half_vector;
    }

    Real roughness = eval(
        bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    // Clamp roughness to avoid numerical issues.
    roughness = std::clamp(roughness, Real(0.01), Real(1));
    Real anisotropic = eval(
        bsdf.anisotropic, vertex.uv, vertex.uv_screen_size, texture_pool);
    // We sample the visible normals, also we use F to determine
    // whether to sample reflection or refraction
    // so PDF ~ F * D * G_in for reflection, PDF ~ (1 - F) * D * G_in for refraction.
    Real h_dot_in = dot(half_vector, dir_in);
    Real F_g = fresnel_dielectric(h_dot_in, eta);

    Real alpha_min = 0.0001;
    Real aspect = sqrt(1 - 0.9 * anisotropic);
    Real alpha_x = max(alpha_min, roughness * roughness / aspect);
    Real alpha_y = max(alpha_min, roughness * roughness * aspect);
    Real Delta_in = 0.5 * (-1 + sqrt(1 + 1 / (pow(to_local(frame, dir_in).z, 2)) * (pow(to_local(frame, dir_in).x * alpha_x, 2) + pow(to_local(frame, dir_in).y * alpha_y, 2))));
    Real G_in = 1 / (1 + Delta_in);
    Real D_g = 1 / (c_PI * alpha_x * alpha_y * pow((pow((to_local(frame, half_vector).x) / alpha_x, 2) + pow((to_local(frame, half_vector).y) / alpha_y, 2) + pow(to_local(frame, half_vector).z, 2)), 2));
    if (reflect) {
        return (F_g * D_g * G_in) / (4 * fabs(dot(frame.n, dir_in)));
    }
    else {
        Real h_dot_out = dot(half_vector, dir_out);
        Real sqrt_denom = h_dot_in + eta * h_dot_out;
        Real dh_dout = eta * eta * h_dot_out / (sqrt_denom * sqrt_denom);
        return (1 - F_g) * D_g * G_in * fabs(dh_dout * h_dot_in / dot(frame.n, dir_in));
    }

}

std::optional<BSDFSampleRecord>
        sample_bsdf_op::operator()(const DisneyGlass &bsdf) const {
    // Flip the shading frame if it is inconsistent with the geometry normal
    Real eta = dot(vertex.geometric_normal, dir_in) > 0 ? bsdf.eta : 1 / bsdf.eta;
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) * dot(vertex.geometric_normal, dir_in) < 0) {
        frame = -frame;
    }
    // Homework 1: implement this!
    Real alpha_min = 0.0001;
    Real anisotropic = eval(
        bsdf.anisotropic, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real roughness = eval(
        bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    // Clamp roughness to avoid numerical issues.
    roughness = std::clamp(roughness, Real(0.01), Real(1));
    // Sample a micro normal and transform it to world space -- this is our half-vector.
    Real aspect = sqrt(1 - 0.9 * anisotropic);
    Real alpha_x = max(alpha_min, roughness * roughness / aspect);
    Real alpha_y = max(alpha_min, roughness * roughness * aspect);
    Vector3 local_dir_in = to_local(frame, dir_in);
    Vector3 local_micro_normal =
        sample_visible_normals(local_dir_in, Vector2{alpha_x,alpha_y}, rnd_param_uv);

    Vector3 half_vector = to_world(frame, local_micro_normal);
    // Flip half-vector if it's below surface
    if (dot(half_vector, frame.n) < 0) {
        half_vector = -half_vector;
    }

    // Now we need to decide whether to reflect or refract.
    // We do this using the Fresnel term.
    Real h_dot_in = dot(half_vector, dir_in);
    Real F = fresnel_dielectric(h_dot_in, eta);

    if (rnd_param_w <= F) {
        // Reflection
        Vector3 reflected = normalize(-dir_in + 2 * dot(dir_in, half_vector) * half_vector);
        // set eta to 0 since we are not transmitting
        return BSDFSampleRecord{ reflected, Real(0) /* eta */, roughness };
    }
    else {
        // Refraction
        // https://en.wikipedia.org/wiki/Snell%27s_law#Vector_form
        // (note that our eta is eta2 / eta1, and l = -dir_in)
        Real h_dot_out_sq = 1 - (1 - h_dot_in * h_dot_in) / (eta * eta);
        if (h_dot_out_sq <= 0) {
            // Total internal reflection
            // This shouldn't really happen, as F will be 1 in this case.
            return {};
        }
        // flip half_vector if needed
        if (h_dot_in < 0) {
            half_vector = -half_vector;
        }
        Real h_dot_out = sqrt(h_dot_out_sq);
        Vector3 refracted = -dir_in / eta + (fabs(h_dot_in) / eta - h_dot_out) * half_vector;
        return BSDFSampleRecord{ refracted, eta, roughness };
    }
}

TextureSpectrum get_texture_op::operator()(const DisneyGlass &bsdf) const {
    return bsdf.base_color;
}
