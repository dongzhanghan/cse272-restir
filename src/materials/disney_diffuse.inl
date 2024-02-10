#include "../microfacet.h"
Spectrum eval_op::operator()(const DisneyDiffuse &bsdf) const {
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

    Vector3 h = normalize(dir_in + dir_out);
    if (dot(h, frame.n) < 0) {
        h = -h;
    }
    Spectrum base_color = eval(
        bsdf.base_color, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real subsurface = eval(
        bsdf.subsurface, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real roughness = eval(
        bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    roughness = std::clamp(roughness, Real(0.01), Real(1));
    Real F_D90 = 0.5 + 2 * roughness * pow(dot(h,dir_out), 2);
    Real F_Din = (1 + (F_D90 - 1)*pow((1 - abs(dot(dir_in, frame.n))),5));
    Real F_Dout = (1 + (F_D90 - 1) * pow((1 - abs(dot(dir_out, frame.n))), 5));
    Spectrum f_basediffuse = base_color / c_PI * F_Din * F_Dout * abs(dot(dir_out, frame.n));

    Real F_SS90 = roughness * pow(dot(h, dir_out), 2);
    Real F_SSin = (1 + (F_SS90 - 1) * pow((1 - abs(dot(dir_in, frame.n))), 5));
    Real F_SSout = (1 + (F_SS90 - 1) * pow((1 - abs(dot(dir_out, frame.n))), 5));
    Spectrum f_subsurface = 1.25 * base_color / c_PI * (F_SSin * F_SSout * (1 / (abs(dot(frame.n, dir_in)) + abs(dot(frame.n, dir_out))) - 0.5) + 0.5) * abs(dot(frame.n, dir_out));

    // Homework 1: implement this!
    //return fmax(dot(frame.n, dir_out), Real(0)) * eval(bsdf.base_color, vertex.uv, vertex.uv_screen_size, texture_pool) / c_PI;
    return (1-subsurface)*f_basediffuse+subsurface*f_subsurface;
}

Real pdf_sample_bsdf_op::operator()(const DisneyDiffuse &bsdf) const {
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
    
    // Homework 1: implement this!
    return fmax(dot(frame.n, dir_out), Real(0)) / c_PI;
}

std::optional<BSDFSampleRecord> sample_bsdf_op::operator()(const DisneyDiffuse &bsdf) const {
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
    Real roughness = eval(
        bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    roughness = std::clamp(roughness, Real(0.01), Real(1));
    return BSDFSampleRecord{
        to_world(frame, sample_cos_hemisphere(rnd_param_uv)),
        Real(0) /* eta */, roughness /* roughness */ };
}

TextureSpectrum get_texture_op::operator()(const DisneyDiffuse &bsdf) const {
    return bsdf.base_color;
}
