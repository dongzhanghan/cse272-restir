#include "../microfacet.h"

Spectrum eval_op::operator()(const DisneyMetal &bsdf) const {
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
    //get parameters
    Vector3 h = normalize(dir_in + dir_out);
    Real alpha_min = 0.0001;
    Spectrum base_color = eval(
        bsdf.base_color, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real anisotropic = eval(
        bsdf.anisotropic, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real roughness = eval(
        bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);

    Spectrum F_m = base_color + (1 - base_color) * pow((1 - abs(dot(h,dir_out))),5);

    Real aspect = sqrt(1 - 0.9 * anisotropic);
    Real alpha_x = max(alpha_min, roughness * roughness / aspect);
    Real alpha_y = max(alpha_min, roughness * roughness * aspect);

    Real Delta_in = 0.5 * (-1 + sqrt(1 + 1 / (pow(to_local(frame, dir_in).z, 2)) * (pow(to_local(frame, dir_in).x * alpha_x, 2) + pow(to_local(frame, dir_in).y * alpha_y, 2))));
    Real Delta_out = 0.5 * (-1 + sqrt(1 + 1 / (pow(to_local(frame, dir_out).z, 2)) * (pow(to_local(frame, dir_out).x * alpha_x, 2) + pow(to_local(frame, dir_out).y * alpha_y, 2))));
    Real G_in = 1 / (1 + Delta_in);
    Real G_out = 1 / (1 + Delta_out);
    Real G_m = G_in * G_out;

    Real D_m = 1 /(c_PI*alpha_x*alpha_y* pow((pow((to_local(frame, h).x)/alpha_x,2)+ pow((to_local(frame, h).y) / alpha_y, 2)+pow(to_local(frame,h).z, 2)),2));
    return F_m*D_m*G_m/(4*abs(dot(frame.n,dir_in)));
}

Real pdf_sample_bsdf_op::operator()(const DisneyMetal &bsdf) const {
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
    Vector3 h = normalize(dir_in + dir_out);
    Real alpha_min = 0.0001;
    Spectrum base_color = eval(
        bsdf.base_color, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real anisotropic = eval(
        bsdf.anisotropic, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real roughness = eval(
        bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    // Clamp roughness to avoid numerical issues.
    roughness = std::clamp(roughness, Real(0.01), Real(1));
    // We use the reflectance to determine whether to choose specular sampling lobe or diffuse.
    // For the specular lobe, we use the ellipsoidal sampling from Heitz 2018
    // "Sampling the GGX Distribution of Visible Normals"
    // https://jcgt.org/published/0007/04/01/
    // this importance samples smith_masking(cos_theta_in) * GTR2(cos_theta_h, roughness) * cos_theta_out
    Real aspect = sqrt(1 - 0.9 * anisotropic);
    Real alpha_x = max(alpha_min, roughness * roughness / aspect);
    Real alpha_y = max(alpha_min, roughness * roughness * aspect);

    Real Delta_in = 0.5 * (-1 + sqrt(1 + 1 / (pow(to_local(frame, dir_in).z, 2)) * (pow(to_local(frame, dir_in).x * alpha_x, 2) + pow(to_local(frame, dir_in).y * alpha_y, 2))));
    Real G_in = 1 / (1 + Delta_in);

    Real D_m = 1 / (c_PI * alpha_x * alpha_y * pow((pow((to_local(frame, h).x) / alpha_x, 2) + pow((to_local(frame, h).y) / alpha_y, 2) + pow(to_local(frame, h).z, 2)), 2));
    // We use visible normal sampling, so the PDF ~ (G_in * D) / (4 * n_dot_in)
    Real spec_prob = (G_in * D_m) / (4 * abs(dot(dir_in,frame.n)));

    return spec_prob;
}

std::optional<BSDFSampleRecord>
        sample_bsdf_op::operator()(const DisneyMetal &bsdf) const {
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

    // Convert the incoming direction to local coordinates
    Vector3 local_dir_in = to_local(frame, dir_in);
    Real roughness = eval(
        bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real anisotropic = eval(
        bsdf.anisotropic, vertex.uv, vertex.uv_screen_size, texture_pool);
    // Clamp roughness to avoid numerical issues.
    roughness = std::clamp(roughness, Real(0.01), Real(1));
    Real alpha_min = 0.0001;
    Real aspect = sqrt(1 - 0.9 * anisotropic);
    Real alpha_x = max(alpha_min, roughness * roughness / aspect);
    Real alpha_y = max(alpha_min, roughness * roughness * aspect);
    Vector3 local_micro_normal =
        sample_visible_normals(local_dir_in, Vector2{alpha_x,alpha_y}, rnd_param_uv);

    // Transform the micro normal to world space
    Vector3 half_vector = to_world(frame, local_micro_normal);
    // Reflect over the world space normal
    Vector3 reflected = normalize(-dir_in + 2 * dot(dir_in, half_vector) * half_vector);
    return BSDFSampleRecord{
            reflected,
            Real(0) /* eta */, roughness /* roughness */
    };
}

TextureSpectrum get_texture_op::operator()(const DisneyMetal &bsdf) const {
    return bsdf.base_color;
}
