#pragma once
#include "point_and_normal.h"

struct Sample {
    PointAndNormal point_on_light;
    int light_id;
};

class Reservoir {
public:
    int M;
    Real w_sum;
    Real W;
    Sample sample;
    Real u;

    // Constructor
    Reservoir() : M(0), w_sum(0), W(0),
        sample(Sample{ PointAndNormal{Vector3{0,0,0}, Vector3{0,0,0}},-1 }), u(0) {}
    Reservoir(Real w_u) : M(0), w_sum(0), W(0),
        sample(Sample{ PointAndNormal{Vector3{0,0,0}, Vector3{0,0,0}},-1 }), u(w_u) {}

    // Update method for reservoir sampling
    void update(Sample& x, Real w) {
        w_sum += w;
        M++;
        if (u < (w / w_sum) && w_sum > 0) {
            sample.light_id = x.light_id;
            sample.point_on_light = x.point_on_light;
        }
    }
};