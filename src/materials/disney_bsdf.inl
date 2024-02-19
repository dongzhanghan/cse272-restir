#include "../microfacet.h"

Spectrum eval_op::operator()(const DisneyBSDF &bsdf) const {  
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) * dot(vertex.geometric_normal, dir_in) < 0) {
        frame = -frame;
    }
    Spectrum base_color = eval(bsdf.base_color, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real specular_transmission = eval(bsdf.specular_transmission, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real metallic = eval(bsdf.metallic, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real subsurface = eval(bsdf.subsurface, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real specular = eval(bsdf.specular, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real roughness = eval(bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    roughness = std::clamp(roughness, Real(0.01), Real(1));
    Real specular_tint = eval(bsdf.specular_tint, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real anisotropic = eval(bsdf.anisotropic, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real sheen = eval(bsdf.sheen, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real sheen_tint = eval(bsdf.sheen_tint, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real clearcoat = eval(bsdf.clearcoat, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real clearcoat_gloss = eval(bsdf.clearcoat_gloss, vertex.uv, vertex.uv_screen_size, texture_pool);

    Real eta = dot(vertex.geometric_normal, dir_in) > 0 ? bsdf.eta : 1 / bsdf.eta;

    //* calculate weight for each lobes
    Real w_diffuse = (1 - specular_transmission) * (1 - metallic);
    Real w_sheen = (1 - metallic) * sheen;
    Real w_metal = 1 - specular_transmission * (1 - metallic);
    Real w_clearcoat = 0.25 * clearcoat;
    Real w_glass = (1 - metallic) * specular_transmission;

    //* some operations on vectors
    bool reflect = dot(vertex.geometric_normal, dir_in) *
        dot(vertex.geometric_normal, dir_out) >= 0;
    Vector3 half_vector = normalize(dir_in + dir_out);

    Spectrum f_glass = eval_op::operator()(DisneyGlass{ bsdf.base_color, bsdf.roughness, bsdf.anisotropic, bsdf.eta });
    Spectrum f_diffuse = eval_op::operator()(DisneyDiffuse{ bsdf.base_color, bsdf.subsurface, bsdf.roughness });

    //* ============================== 
    //* calculate f_metal
    //* ============================== 
    Real R0 = (eta - 1) * (eta - 1) / ((eta + 1) * (eta + 1));
    Spectrum c_tint = luminance(base_color) > 0 ? base_color / luminance(base_color) : make_const_spectrum(1.);
    Spectrum k_s = (1 - specular_tint) + specular_tint * c_tint;
    Spectrum base_color_0 = specular * R0 * (1 - metallic) * k_s + metallic * base_color;

    Spectrum F_m = base_color_0 + (1 - base_color_0) * pow(1 - dot(half_vector, dir_out), 5);
    Real alpha_min = 0.0001;
    Real aspect = sqrt(1 - 0.9 * anisotropic);
    Real alpha_x = max(alpha_min, roughness * roughness / aspect);
    Real alpha_y = max(alpha_min, roughness * roughness * aspect);

    Real Delta_in = 0.5 * (-1 + sqrt(1 + 1 / (pow(to_local(frame, dir_in).z, 2)) * (pow(to_local(frame, dir_in).x * alpha_x, 2) + pow(to_local(frame, dir_in).y * alpha_y, 2))));
    Real Delta_out = 0.5 * (-1 + sqrt(1 + 1 / (pow(to_local(frame, dir_out).z, 2)) * (pow(to_local(frame, dir_out).x * alpha_x, 2) + pow(to_local(frame, dir_out).y * alpha_y, 2))));
    Real G_in = 1 / (1 + Delta_in);
    Real G_out = 1 / (1 + Delta_out);
    Real G_m = G_in * G_out;

    Real D_m = 1 / (c_PI * alpha_x * alpha_y * pow((pow((to_local(frame, half_vector).x) / alpha_x, 2) + pow((to_local(frame, half_vector).y) / alpha_y, 2) + pow(to_local(frame, half_vector).z, 2)), 2));
    Spectrum f_metal = F_m * D_m * G_m / (4 * abs(dot(frame.n, dir_in)));

    Spectrum f_sheen = eval_op::operator()(DisneySheen{ bsdf.base_color, bsdf.sheen_tint });
    Spectrum f_clearcoat = eval_op::operator()(DisneyClearcoat{ bsdf.clearcoat_gloss });

    bool inside = dot(vertex.geometric_normal, dir_in) <= 0;
    if (inside) {
        // Only the glass component if the light is under the surface
        f_diffuse = make_const_spectrum(0);
        f_metal = make_const_spectrum(0);
        f_sheen = make_const_spectrum(0);
        f_clearcoat = make_const_spectrum(0);
    }

    Spectrum f_bsdf;
    if (reflect) {
        f_bsdf = w_diffuse * f_diffuse +
            w_metal * f_metal +
            w_sheen * f_sheen +
            w_clearcoat * f_clearcoat +
            w_glass * f_glass;
    }
    else {
        // Only the glass component for refraction
        f_bsdf = w_glass * f_glass;
    }

    return f_bsdf;
}

Real pdf_sample_bsdf_op::operator()(const DisneyBSDF &bsdf) const {
    bool reflect = dot(vertex.geometric_normal, dir_in) *
        dot(vertex.geometric_normal, dir_out) > 0;
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) * dot(vertex.geometric_normal, dir_in) < 0) {
        frame = -frame;
    }

    Spectrum base_color = eval(
        bsdf.base_color, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real subsurface = eval(
        bsdf.subsurface, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real roughness = eval(
        bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    roughness = std::clamp(roughness, Real(0.01), Real(1));
    Real specular_transmission = eval(
        bsdf.specular_transmission, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real metallic = eval(
        bsdf.metallic, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real specular = eval(
        bsdf.specular, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real specular_tint = eval(
        bsdf.specular_tint, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real anisotropic = eval(
        bsdf.anisotropic, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real sheen_tint = eval(
        bsdf.sheen_tint, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real sheen = eval(
        bsdf.sheen, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real clearcoat = eval(
        bsdf.clearcoat, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real clearcoat_gloss = eval(
        bsdf.clearcoat_gloss, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real w_diffuse = (1 - metallic) * (1 - specular_transmission);
    Real w_metal = (1 - specular_transmission * (1 - metallic));
    Real w_glass = (1 - metallic) * specular_transmission;
    Real w_clearcoat = 0.25 * clearcoat;
    Real total = w_diffuse + w_metal + w_glass + w_clearcoat;
    //normalize to let the sum weight to be 1, cannot make them a single line
    w_diffuse /= total;
    w_metal /= total;
    w_clearcoat /= total;
    w_glass /= total;
    bool inside = dot(vertex.geometric_normal, dir_in) <= 0;
    if (inside) {
        // Only the glass component if the light is under the surface
        w_diffuse = 0;
        w_metal = 0;
        w_glass = 1;
        w_clearcoat = 0;
    }
    if (reflect) {  
        struct DisneyDiffuse diffuse = { bsdf.base_color,bsdf.roughness,bsdf.subsurface };
        struct DisneyClearcoat clearcoat = { bsdf.clearcoat_gloss };
        struct DisneyGlass glass = { bsdf.base_color,bsdf.roughness,bsdf.anisotropic,bsdf.eta };
        struct DisneyMetal metal = { bsdf.base_color,bsdf.roughness,bsdf.anisotropic};
        Real pdf = w_diffuse * pdf_sample_bsdf(diffuse, dir_in, dir_out, vertex, texture_pool) +
            w_metal * pdf_sample_bsdf(metal, dir_in, dir_out, vertex, texture_pool) +
            w_glass * pdf_sample_bsdf(glass, dir_in, dir_out, vertex, texture_pool) +
            w_clearcoat * pdf_sample_bsdf(clearcoat, dir_in, dir_out, vertex, texture_pool);     
        return pdf;
    }
    else {
        // Only the glass component for refraction
        struct DisneyGlass glass = { bsdf.base_color,bsdf.roughness,bsdf.anisotropic,bsdf.eta };
        return pdf_sample_bsdf(glass, dir_in, dir_out, vertex, texture_pool);
    }
}

std::optional<BSDFSampleRecord>
        sample_bsdf_op::operator()(const DisneyBSDF &bsdf) const {
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) * dot(vertex.geometric_normal, dir_in) < 0) {
        frame = -frame;
    }
    Spectrum base_color = eval(
        bsdf.base_color, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real subsurface = eval(
        bsdf.subsurface, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real roughness = eval(
        bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    roughness = std::clamp(roughness, Real(0.01), Real(1));
    Real specular_transmission = eval(
        bsdf.specular_transmission, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real metallic = eval(
        bsdf.metallic, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real specular = eval(
        bsdf.specular, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real specular_tint = eval(
        bsdf.specular_tint, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real anisotropic = eval(
        bsdf.anisotropic, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real sheen_tint = eval(
        bsdf.sheen_tint, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real sheen = eval(
        bsdf.sheen, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real clearcoat = eval(
        bsdf.clearcoat, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real clearcoat_gloss = eval(
        bsdf.clearcoat_gloss, vertex.uv, vertex.uv_screen_size, texture_pool);

    bool inside = dot(vertex.geometric_normal, dir_in) <= 0;

    if (!inside) {
        Real w_diffuse = (1 - metallic) * (1 - specular_transmission);
        Real w_metal = 1 - specular_transmission * (1 - metallic);
        Real w_glass = (1 - metallic) * specular_transmission;
        Real w_clearcoat = 0.25 * clearcoat;
        Real total = w_diffuse + w_metal + w_glass + w_clearcoat;
        w_diffuse /= total;
        w_metal /= total;
        w_clearcoat /= total;
        w_glass /= total;
        //rescale your random number w for selecting reflection / refraction.
        if (rnd_param_w <= w_diffuse) {
            struct DisneyDiffuse diffuse = { bsdf.base_color,bsdf.roughness,bsdf.subsurface };
            return sample_bsdf(diffuse, dir_in, vertex, texture_pool, rnd_param_uv, rnd_param_w/w_diffuse);
        }else if(rnd_param_w <= w_diffuse+w_metal){
            struct DisneyMetal metal = { bsdf.base_color,bsdf.roughness,bsdf.anisotropic };
            return sample_bsdf(metal, dir_in, vertex, texture_pool, rnd_param_uv, (rnd_param_w - w_diffuse) / w_metal);
        }
        else if (rnd_param_w <= w_diffuse + w_metal+w_glass) {
            struct DisneyGlass glass = { bsdf.base_color,bsdf.roughness,bsdf.anisotropic,bsdf.eta };
            return sample_bsdf(glass, dir_in, vertex, texture_pool, rnd_param_uv, (rnd_param_w - w_diffuse-w_metal) / w_glass);
        }
        else {
            struct DisneyClearcoat clearcoat = { bsdf.clearcoat_gloss };
            return sample_bsdf(clearcoat, dir_in, vertex, texture_pool, rnd_param_uv, (rnd_param_w - w_diffuse - w_metal- w_glass) / w_clearcoat);
        }
    }
    else {
        // Only the glass component if the light is under the surface
        struct DisneyGlass glass = { bsdf.base_color,bsdf.roughness,bsdf.anisotropic,bsdf.eta };
        return sample_bsdf(glass,dir_in,vertex,texture_pool,rnd_param_uv,rnd_param_w);
    }
}

TextureSpectrum get_texture_op::operator()(const DisneyBSDF &bsdf) const {
    return bsdf.base_color;
}
