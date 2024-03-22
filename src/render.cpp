#include "render.h"
#include "intersection.h"
#include "material.h"
#include "parallel.h"
#include "path_tracing.h"
#include "vol_path_tracing.h"
#include "pcg.h"
#include "progress_reporter.h"
#include "scene.h"
#include "restir_path_tracing.h"

/// Render auxiliary buffers e.g., depth.
Image3 aux_render(const Scene &scene) {
    int w = scene.camera.width, h = scene.camera.height;
    Image3 img(w, h);
    constexpr int tile_size = 16;
    int num_tiles_x = (w + tile_size - 1) / tile_size;
    int num_tiles_y = (h + tile_size - 1) / tile_size;

    parallel_for([&](const Vector2i &tile) {
        int x0 = tile[0] * tile_size;
        int x1 = min(x0 + tile_size, w);
        int y0 = tile[1] * tile_size;
        int y1 = min(y0 + tile_size, h);
        for (int y = y0; y < y1; y++) {
            for (int x = x0; x < x1; x++) {
                Ray ray = sample_primary(scene.camera, Vector2((x + Real(0.5)) / w, (y + Real(0.5)) / h));
                RayDifferential ray_diff = init_ray_differential(w, h);
                if (std::optional<PathVertex> vertex = intersect(scene, ray, ray_diff)) {
                    Real dist = distance(vertex->position, ray.org);
                    Vector3 color{0, 0, 0};
                    if (scene.options.integrator == Integrator::Depth) {
                        color = Vector3{dist, dist, dist};
                    } else if (scene.options.integrator == Integrator::ShadingNormal) {
                        // color = (vertex->shading_frame.n + Vector3{1, 1, 1}) / Real(2);
                        color = vertex->shading_frame.n;
                    } else if (scene.options.integrator == Integrator::MeanCurvature) {
                        Real kappa = vertex->mean_curvature;
                        color = Vector3{kappa, kappa, kappa};
                    } else if (scene.options.integrator == Integrator::RayDifferential) {
                        color = Vector3{ray_diff.radius, ray_diff.spread, Real(0)};
                    } else if (scene.options.integrator == Integrator::MipmapLevel) {
                        const Material &mat = scene.materials[vertex->material_id];
                        const TextureSpectrum &texture = get_texture(mat);
                        auto *t = std::get_if<ImageTexture<Spectrum>>(&texture);
                        if (t != nullptr) {
                            const Mipmap3 &mipmap = get_img3(scene.texture_pool, t->texture_id);
                            Vector2 uv{modulo(vertex->uv[0] * t->uscale, Real(1)),
                                       modulo(vertex->uv[1] * t->vscale, Real(1))};
                            // ray_diff.radius stores approximatedly dpdx,
                            // but we want dudx -- we get it through
                            // dpdx / dpdu
                            Real footprint = vertex->uv_screen_size;
                            Real scaled_footprint = max(get_width(mipmap), get_height(mipmap)) *
                                                    max(t->uscale, t->vscale) * footprint;
                            Real level = log2(max(scaled_footprint, Real(1e-8f)));
                            color = Vector3{level, level, level};
                        }
                    }
                    img(x, y) = color;
                } else {
                    img(x, y) = Vector3{0, 0, 0};
                }
            }
        }
    }, Vector2i(num_tiles_x, num_tiles_y));

    return img;
}

Image3 path_render(const Scene &scene) {
    int w = scene.camera.width, h = scene.camera.height;
    Image3 img(w, h);

    constexpr int tile_size = 16;
    int num_tiles_x = (w + tile_size - 1) / tile_size;
    int num_tiles_y = (h + tile_size - 1) / tile_size;

    ProgressReporter reporter(num_tiles_x * num_tiles_y);
    parallel_for([&](const Vector2i &tile) {
        // Use a different rng stream for each thread.
        pcg32_state rng = init_pcg32(tile[1] * num_tiles_x + tile[0]);
        int x0 = tile[0] * tile_size;
        int x1 = min(x0 + tile_size, w);
        int y0 = tile[1] * tile_size;
        int y1 = min(y0 + tile_size, h);
        for (int y = y0; y < y1; y++) {
            for (int x = x0; x < x1; x++) {
                Spectrum radiance = make_zero_spectrum();
                int spp = scene.options.samples_per_pixel;
                for (int s = 0; s < spp; s++) {
                    radiance += path_tracing(scene, x, y, rng);
                }
                img(x, y) = radiance / Real(spp);
            }
        }
        reporter.update(1);
    }, Vector2i(num_tiles_x, num_tiles_y));
    reporter.done();
    return img;
}

Image3 vol_path_render(const Scene &scene) {
    int w = scene.camera.width, h = scene.camera.height;
    Image3 img(w, h);

    constexpr int tile_size = 16;
    int num_tiles_x = (w + tile_size - 1) / tile_size;
    int num_tiles_y = (h + tile_size - 1) / tile_size;

    auto f = vol_path_tracing;
    if (scene.options.vol_path_version == 1) {
        f = vol_path_tracing_1;
    } else if (scene.options.vol_path_version == 2) {
        f = vol_path_tracing_2;
    } else if (scene.options.vol_path_version == 3) {
        f = vol_path_tracing_3;
    } else if (scene.options.vol_path_version == 4) {
        f = vol_path_tracing_4;
    } else if (scene.options.vol_path_version == 5) {
        f = vol_path_tracing_5;
    } else if (scene.options.vol_path_version == 6) {
        f = vol_path_tracing;
    }

    ProgressReporter reporter(num_tiles_x * num_tiles_y);
    parallel_for([&](const Vector2i &tile) {
        // Use a different rng stream for each thread.
        pcg32_state rng = init_pcg32(tile[1] * num_tiles_x + tile[0]);
        int x0 = tile[0] * tile_size;
        int x1 = min(x0 + tile_size, w);
        int y0 = tile[1] * tile_size;
        int y1 = min(y0 + tile_size, h);
        for (int y = y0; y < y1; y++) {
            for (int x = x0; x < x1; x++) {
                Spectrum radiance = make_zero_spectrum();
                int spp = scene.options.samples_per_pixel;
                for (int s = 0; s < spp; s++) {
                    Spectrum L = f(scene, x, y, rng);
                    if (isfinite(L)) {
                        // Hacky: exclude NaNs in the rendering.
                        radiance += L;
                    }
                }
                img(x, y) = radiance / Real(spp);
            }
        }
        reporter.update(1);
    }, Vector2i(num_tiles_x, num_tiles_y));
    reporter.done();
    return img;
}


// path 1: sample buffer and reservoir buffer, save some info
// path 2: path tracing
Image3 restir_path_render(const Scene &scene) {
    int w = scene.camera.width, h = scene.camera.height;
    Image3 img(w, h);
    constexpr int tile_size = 2;
    int num_tiles_x = (w + tile_size - 1) / tile_size;
    int num_tiles_y = (h + tile_size - 1) / tile_size;
    int spp = scene.options.samples_per_pixel;

    ProgressReporter reporter(spp);
    for (int i = 0; i < spp; i++) {
        ImageReservoir imgReservoir(w, h);
        ImageRay imgRay(w, h);
        ImagePathVertex imgPathVertex(w, h);
        parallel_for([&](const Vector2i &tile) {
            pcg32_state rng = init_pcg32(tile[1] * num_tiles_x + tile[0] + spp * i);
            int x0 = tile[0] * tile_size;
            int x1 = min(x0 + tile_size, w);
            int y0 = tile[1] * tile_size;
            int y1 = min(y0 + tile_size, h);
            for (int y = y0; y < y1; y++) {
                for (int x = x0; x < x1; x++) {
                    //int w = scene.camera.width, h = scene.camera.height;
                    Vector2 screen_pos((x + next_pcg32_real<Real>(rng)) / w,
                                       (y + next_pcg32_real<Real>(rng)) / h);
                    Ray ray = sample_primary(scene.camera, screen_pos);
                    imgRay(x, y) = ray;
                    RayDifferential ray_diff = init_ray_differential(w, h);
                    std::optional<PathVertex> vertex_ = intersect(scene, ray, ray_diff);
                    imgPathVertex(x, y) = vertex_;
                    if (!vertex_) {
                        break;
                    }
                    PathVertex vertex = *vertex_;
                    RIS(scene, rng, vertex, ray, imgReservoir(x, y));
                }
            }
        }, Vector2i(num_tiles_x, num_tiles_y));
        parallel_for([&](const Vector2i &tile) {
            // Use a different rng stream for each thread.
            pcg32_state rng = init_pcg32(tile[1] * num_tiles_x + tile[0] + spp * i);
            int x0 = tile[0] * tile_size;
            int x1 = min(x0 + tile_size, w);
            int y0 = tile[1] * tile_size;
            int y1 = min(y0 + tile_size, h);


            for (int y = y0; y < y1; y++) {
                for (int x = x0; x < x1; x++) {
                    Spectrum radiance = make_zero_spectrum();
                    //radiance += path_tracing(scene, x, y, rng);

                    // restir path tracing
                    radiance = restir_path_tracing_1(scene, x, y, imgPathVertex(x, y), imgRay(x, y), rng, imgReservoir,imgRay, imgPathVertex);
                    //std::cout << "radiance of pixel (" << x << ", " << y << "): " << radiance << std::endl;
                    img(x, y) += radiance / Real(spp);
                }
            }

        }, Vector2i(num_tiles_x, num_tiles_y));
        reporter.update(1);
    }

    reporter.done();
    return img;
}

//Image3 restir_path_render(const Scene &scene) {
//    int w = scene.camera.width, h = scene.camera.height;
//    Image3 img(w, h);
//    constexpr int tile_size = 16;
//    int num_tiles_x = (w + tile_size - 1) / tile_size;
//    int num_tiles_y = (h + tile_size - 1) / tile_size;
//    int spp = scene.options.samples_per_pixel;
//    ProgressReporter reporter(spp);
//    for (int i = 0; i < spp; i++) {
//        parallel_for([&](const Vector2i &tile) {
//            // Use a different rng stream for each thread.
//            pcg32_state rng = init_pcg32(tile[1] * num_tiles_x + tile[0] +spp*i);
//            int x0 = tile[0] * tile_size;
//            int x1 = min(x0 + tile_size, w);
//            int y0 = tile[1] * tile_size;
//            int y1 = min(y0 + tile_size, h);
//            for (int y = y0; y < y1; y++) {
//                for (int x = x0; x < x1; x++) {
//                    Spectrum radiance = make_zero_spectrum();
//                    radiance += path_tracing(scene, x, y, rng);
//                    img(x, y) += radiance / Real(spp);
//                }
//            }
//
//        }, Vector2i(num_tiles_x, num_tiles_y));
//        reporter.update(1);
//    }
//    reporter.done();
//    return img;
//}

// test variance of restir path tracing
void test_variance_restir_path_render_old(const Scene &scene) {
    // test data
    int x = 1000, y = 1000;
    int test_times = 1000;

    int w = scene.camera.width, h = scene.camera.height;
    int spp = scene.options.samples_per_pixel;
    Image3 variance_res(test_times, 1);
    ProgressReporter reporter(test_times * spp);
    for (int times = 0; times < test_times; times++) {
        Spectrum sum_radiance = make_zero_spectrum();
        for (int i = 0; i < spp; i++) {
            ImageReservoir imgReservoir(w, h);
            ImageRay imgRay(w, h);
            ImagePathVertex imgPathVertex(w, h);
            pcg32_state rng = init_pcg32(times + i);
            Vector2 screen_pos((x + next_pcg32_real<Real>(rng)) / w,
                               (y + next_pcg32_real<Real>(rng)) / h);
            Ray ray = sample_primary(scene.camera, screen_pos);
            imgRay(x, y) = ray;
            RayDifferential ray_diff = init_ray_differential(w, h);
            std::optional<PathVertex> vertex_ = intersect(scene, ray, ray_diff);
            imgPathVertex(x, y) = vertex_;
            if (vertex_) {
                PathVertex vertex = *vertex_;
                RIS(scene, rng, vertex, ray, imgReservoir(x, y));
            }
            Spectrum radiance = make_zero_spectrum();
            if (vertex_) {
                radiance = restir_path_tracing_1(scene, x, y, imgPathVertex(x, y), imgRay(x, y), rng, imgReservoir,imgRay,imgPathVertex);
                if (isfinite(radiance)) {
                    sum_radiance += radiance;
                }
            }
            reporter.update(1);
        }
        variance_res(times, 0) = sum_radiance / Real(spp);
    }
    reporter.done();

    // calculate mean
    Spectrum mean = make_zero_spectrum();
    for (int i = 0; i < test_times; i++) {
        mean += variance_res(i, 0);
    }
    mean /= Real(test_times);

    // calculate variance
    Spectrum variance = make_zero_spectrum();
    for (int i = 0; i < test_times; i++) {
        variance += (variance_res(i, 0) - mean) * (variance_res(i, 0) - mean);
    }
    variance /= Real(test_times);
    std::cout << "unnormalized variance of pixel (" << x << ", " << y << "): " << variance << std::endl;

    Spectrum normalized_variance = make_zero_spectrum();
    if (mean[0] != 0) {
        normalized_variance = variance / mean;
    }
    std::cout << "normalized variance of pixel (" << x << ", " << y << "): " << normalized_variance << std::endl;
}

//int x_min = 260, x_max = 550, y_min = 415, y_max = 900; // surrounding around box1.
//Image3 box_location_finder(const Scene &scene) {
//    // only render pixel in the box:
//    int x_min = 260, x_max = 550, y_min = 415, y_max = 900; // surrounding around box1.
//    int test_times = 1;
//
//    int w = scene.camera.width, h = scene.camera.height;
//    Image3 img(w, h);
//    // no need to use parallel_for because we only render pixels in the box
//    int spp = scene.options.samples_per_pixel;
//    ProgressReporter reporter(spp * test_times);
//    // first make all the image black
//    for (int i = 0; i < w; i++) {
//        for (int j = 0; j < h; j++) {
//            img(i, j) = Spectrum(0, 0, 0);
//        }
//    }
//    // calculate the variance of each pixel
//    Image3 variance_res(test_times, 1);
//    for (int times = 0; times < test_times; times++) {
//        for (int i = 0; i < spp; i++) {
//            ImageReservoir imgReservoir(w, h);
//            ImageRay imgRay(w, h);
//            ImagePathVertex imgPathVertex(w, h);
//            pcg32_state rng = init_pcg32(times + i);
//            for (int y = 0; y < h; y++) {
//                for (int x = 0; x < w; x++) {
//                    if (x < x_min || x > x_max || y < y_min || y > y_max) {
//                        continue;
//                    }
//                    Vector2 screen_pos((x + next_pcg32_real<Real>(rng)) / w,
//                                       (y + next_pcg32_real<Real>(rng)) / h);
//                    Ray ray = sample_primary(scene.camera, screen_pos);
//                    imgRay(x, y) = ray;
//                    RayDifferential ray_diff = init_ray_differential(w, h);
//                    std::optional<PathVertex> vertex_ = intersect(scene, ray, ray_diff);
//                    imgPathVertex(x, y) = vertex_;
//                    if (vertex_) {
//                        PathVertex vertex = *vertex_;
//                        RIS(scene, rng, vertex, ray, imgReservoir(x, y));
//                    }
//                    Spectrum radiance = make_zero_spectrum();
//                    if (vertex_) {
//                        radiance = restir_path_tracing_1(scene, x, y, imgPathVertex(x, y), imgRay(x, y), rng,
//                                                         imgReservoir);
//                        if (isfinite(radiance)) {
//                            img(x, y) += radiance / Real(spp) / Real(test_times);
//                        }
//                    }
//                }
//            }
//            reporter.update(1);
//        }
//
//    }
//    reporter.done();
//
//    return img;
//}
//
//void test_variance_restir_path_render(const Scene &scene) {
//    // test data
//    int x_min = 270, x_max = 540, y_min = 415, y_max = 900; // surrounding around box1.
//    int test_times = 1;
//
//    int w = scene.camera.width, h = scene.camera.height;
//    int spp = scene.options.samples_per_pixel;
//    int tile_size = 16;
//    int num_tiles_x = ((x_max - x_min) + tile_size - 1) / tile_size;
//    int num_tiles_y = ((y_max - y_min) + tile_size - 1) / tile_size;
//
//    Image3 variance_res(test_times, 1);
//    ProgressReporter reporter(test_times * spp * (x_max - x_min) * (y_max - y_min));
//    double sum_unnormalized_variance = 0;
//    double sum_normalized_variance = 0;
//
//    parallel_for([&](const Vector2i &tile) {
//        int x0 = max(tile[0] * tile_size + x_min, x_min);
//        int x1 = min((tile[0] + 1) * tile_size + x_min, x_max);
//        int y0 = max(tile[1] * tile_size + y_min, y_min);
//        int y1 = min((tile[1] + 1) * tile_size + y_min, y_max);
//
//        for (int x = x0; x < x1; x++) {
//            for (int y = y0; y < y1; y++) {
//                for (int times = 0; times < test_times; times++) {
//                    Spectrum sum_radiance = make_zero_spectrum();
//                    for (int i = 0; i < spp; i++) {
//                        ImageReservoir imgReservoir(w, h);
//                        ImageRay imgRay(w, h);
//                        ImagePathVertex imgPathVertex(w, h);
//                        pcg32_state rng = init_pcg32(times + i);
//                        Vector2 screen_pos((x + next_pcg32_real<Real>(rng)) / w,
//                                           (y + next_pcg32_real<Real>(rng)) / h);
//                        Ray ray = sample_primary(scene.camera, screen_pos);
//                        imgRay(x, y) = ray;
//                        RayDifferential ray_diff = init_ray_differential(w, h);
//                        std::optional<PathVertex> vertex_ = intersect(scene, ray, ray_diff);
//                        imgPathVertex(x, y) = vertex_;
//                        if (vertex_) {
//                            PathVertex vertex = *vertex_;
//                            RIS(scene, rng, vertex, ray, imgReservoir(x, y));
//                        }
//                        Spectrum radiance = make_zero_spectrum();
//                        if (vertex_) {
//                            radiance = restir_path_tracing_1(scene, x, y, imgPathVertex(x, y), imgRay(x, y), rng,
//                                                             imgReservoir);
//                            if (isfinite(radiance)) {
//                                sum_radiance += radiance;
//                            }
//                        }
//                        reporter.update(1);
//                    }
//                    variance_res(times, 0) = sum_radiance / Real(spp);
//                }
//                // calculate mean
//                Spectrum mean = make_zero_spectrum();
//                for (int i = 0; i < test_times; i++) {
//                    mean += variance_res(i, 0);
//                }
//                mean /= Real(test_times);
//                //printf("mean of pixel (%d, %d): %f\n", x, y, mean[0]);
//
//                // calculate variance
//                Spectrum variance = make_zero_spectrum();
//                for (int i = 0; i < test_times; i++) {
//                    variance += (variance_res(i, 0) - mean) * (variance_res(i, 0) - mean);
//                    // print 3 dimensions of variance
////                printf("variance of pixel (%d, %d): %f\n", x, y, variance[0]);
////                printf("variance of pixel (%d, %d): %f\n", x, y, variance[1]);
////                printf("variance of pixel (%d, %d): %f\n", x, y, variance[2]);
//                }
//                variance /= Real(test_times);
//                std::cout << "\nunnormalized variance of pixel (" << x << ", " << y << "): " << variance << std::endl;
//                sum_unnormalized_variance += variance[0] + variance[1] + variance[2];
//                Spectrum normalized_variance = make_zero_spectrum();
//                if (mean[0] != 0) {
//                    normalized_variance = variance / mean;
//                }
//                std::cout << "normalized variance of pixel (" << x << ", " << y << "): " << normalized_variance << std::endl;
//                sum_normalized_variance += normalized_variance[0] + normalized_variance[1] + normalized_variance[2];
//            }
//        }
//
//    }, Vector2i(num_tiles_x, num_tiles_y));
//    reporter.done();
//    // print sum_unnormalized_variance and sum_normalized_variance
//    printf("sum_unnormalized_variance: %f\n", sum_unnormalized_variance);
//    printf("sum_normalized_variance: %f\n", sum_normalized_variance);
//    return;
//}


Image3 render(const Scene &scene) {
    if (scene.options.integrator == Integrator::Depth ||
        scene.options.integrator == Integrator::ShadingNormal ||
        scene.options.integrator == Integrator::MeanCurvature ||
        scene.options.integrator == Integrator::RayDifferential ||
        scene.options.integrator == Integrator::MipmapLevel) {
        return aux_render(scene);
    } else if (scene.options.integrator == Integrator::Path) {
        return path_render(scene);
    } else if (scene.options.integrator == Integrator::VolPath) {
        return vol_path_render(scene);
    } else if (scene.options.integrator == Integrator::RestirPath) {
        return restir_path_render(scene);
//    } else if (scene.options.integrator == Integrator::TestVarianceRestirPath) {
////        test_variance_restir_path_render(scene);
////        return Image3();
//        return box_location_finder(scene);
    } else {
        assert(false);
        return Image3();
    }
}



