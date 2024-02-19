#include "../microfacet.h"

Spectrum eval_op::operator()(const DisneyClearcoat &bsdf) const {
    if (dot(vertex.geometric_normal, dir_in) < 0 ||
            dot(vertex.geometric_normal, dir_out) < 0) {
        // No light below the surface
        return make_zero_spectrum();
    }
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) < 0) {
        frame = -frame;
    }
    //why ??????
    Vector3 half_vector = normalize(dir_in + dir_out);
    Real n_dot_h = dot(frame.n, half_vector);
    Real n_dot_out = dot(frame.n, dir_out);
    Real h_dot_out = dot(half_vector, dir_out);
    if (n_dot_out <= 0 || n_dot_h <= 0 || h_dot_out <= 0) {
        return make_zero_spectrum();
    }

    Real clearcoat_gloss = eval(
        bsdf.clearcoat_gloss, vertex.uv, vertex.uv_screen_size, texture_pool);

    // They use a hardcoded IOR 1.5 -> R0 = 0.04
    Real R_0 = 1.0 / 25;
    Vector3 h = normalize(dir_in + dir_out);
    Real F_c = R_0 + (1 - R_0) * pow(1 - abs(dot(h, dir_out)), 5);
    // Generalized Trowbridge-Reitz distribution
    Real alpha_g = (1 - clearcoat_gloss) * 0.1 + clearcoat_gloss * 0.001;
    Real D_c = (alpha_g * alpha_g - 1) / (c_PI * log(alpha_g * alpha_g) * (1 + (alpha_g * alpha_g - 1) * pow(to_local(frame, h).z, 2)));

    // SmithG with fixed alpha
    Real Delta_in = 0.5 * (-1 + sqrt(1 + 1 / (pow(to_local(frame, dir_in).z, 2)) * (pow(to_local(frame, dir_in).x * 0.25, 2) + pow(to_local(frame, dir_in).y * 0.25, 2))));
    Real Delta_out = 0.5 * (-1 + sqrt(1 + 1 / (pow(to_local(frame, dir_out).z, 2)) * (pow(to_local(frame, dir_out).x * 0.25, 2) + pow(to_local(frame, dir_out).y * 0.25, 2))));
    Real G_in = 1 / (1 + Delta_in);
    Real G_out = 1 / (1 + Delta_out);
    Real G_c = G_in * G_out;
    return Spectrum{ F_c,F_c,F_c } *D_c * G_c / (4 * abs(dot(frame.n, dir_in)));
}

Real pdf_sample_bsdf_op::operator()(const DisneyClearcoat &bsdf) const {
    if (dot(vertex.geometric_normal, dir_in) < 0 ||
            dot(vertex.geometric_normal, dir_out) < 0) {
        // No light below the surface
        return 0;
    }
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) < 0) {
        frame = -frame;
    }
    Vector3 half_vector = normalize(dir_in + dir_out);
    Real n_dot_h = dot(frame.n, half_vector);
    Real n_dot_out = dot(frame.n, dir_out);
    Real h_dot_out = dot(half_vector, dir_out);
    if (n_dot_out <= 0 || n_dot_h <= 0 || h_dot_out <= 0) {
        return 0;
    }
    // Homework 1: implement this!
    Vector3 h = normalize(dir_in + dir_out);
    Real clearcoat_gloss = eval(
        bsdf.clearcoat_gloss, vertex.uv, vertex.uv_screen_size, texture_pool);
    clearcoat_gloss = std::clamp(clearcoat_gloss, Real(0.01), Real(1));
    Real alpha_g = (1 - clearcoat_gloss) * 0.1 + clearcoat_gloss * 0.001;
    Real D_c = (alpha_g * alpha_g - 1) / (c_PI * log(alpha_g * alpha_g) * (1 + (alpha_g * alpha_g - 1) * pow(to_local(frame, h).z, 2)));
    return  D_c * abs(dot(frame.n, h)) / (4 * abs(dot(dir_out, frame.n)));
}

std::optional<BSDFSampleRecord>
        sample_bsdf_op::operator()(const DisneyClearcoat &bsdf) const {
    if (dot(vertex.geometric_normal, dir_in) < 0) {
        // No light below the surface
        return {};
    }
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) < 0) {
        frame = -frame;
    }
    // Homework 1: implement this!
    Real clearcoat_gloss = eval(
        bsdf.clearcoat_gloss, vertex.uv, vertex.uv_screen_size, texture_pool);
    clearcoat_gloss = std::clamp(clearcoat_gloss, Real(0.01), Real(1));
    Real alpha_g = (1 - clearcoat_gloss) * 0.1 + clearcoat_gloss * 0.001;

    Real h_elevation = acos(sqrt((1 - pow(alpha_g * alpha_g, 1 - rnd_param_uv.x)) / (1 - alpha_g * alpha_g)));
    Real h_azimuth = 2 * c_PI * rnd_param_uv.y;
    Vector3 h = {sin(h_elevation) * cos(h_azimuth),sin(h_elevation) * sin(h_azimuth), cos(h_elevation)};
    Vector3 half_vector = to_world(frame, h);
    Vector3 reflected = normalize(-dir_in + 2 * dot(dir_in, half_vector) * half_vector);
    return BSDFSampleRecord{
            reflected,
            Real(0) /* eta */, sqrt(alpha_g) /* roughness */  //why sqrt??????
    };
    return {};
}

TextureSpectrum get_texture_op::operator()(const DisneyClearcoat &bsdf) const {
    return make_constant_spectrum_texture(make_zero_spectrum());
}
