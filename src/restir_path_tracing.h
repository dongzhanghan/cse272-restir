#pragma once
#include "scene.h"
#include "pcg.h"


/*The simplest restir, only include RIS+ WAS*/
Spectrum restir_path_tracing_1(const Scene& scene,
    int x, int y, /* pixel coordinates */
    pcg32_state& rng) {
    int w = scene.camera.width, h = scene.camera.height;
    Vector2 screen_pos((x + next_pcg32_real<Real>(rng)) / w,
        (y + next_pcg32_real<Real>(rng)) / h);
    Ray ray = sample_primary(scene.camera, screen_pos);
    return make_zero_spectrum();
}