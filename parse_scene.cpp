#include "parse_scene.h"
#include "flexception.h"
#include "load_serialized.h"
#include "parse_obj.h"
#include "pugixml.hpp"
#include "transform.h"
#include <cctype>
#include <map>
#include <regex>

const Real c_default_fov = 45.0;
const int c_default_res = 256;
const std::string c_default_filename = "image.pfm";
const std::string c_default_material_name = "___default_white";

struct ParsedSampler {
    int sample_count = 4;
};

struct ParsedTexture {
    fs::path filename;
    Real uscale = 1, vscale = 1;
};

enum class FovAxis {
    X,
    Y,
    DIAGONAL,
    SMALLER,
    LARGER
};

std::vector<std::string> split_string(const std::string &str, const std::regex &delim_regex) {
    std::sregex_token_iterator first{begin(str), end(str), delim_regex, -1}, last;
    std::vector<std::string> list{first, last};
    return list;
}

Vector3 parse_vector3(const std::string &value) {
    std::vector<std::string> list = split_string(value, std::regex("(,| )+"));
    Vector3 v;
    if (list.size() == 1) {
        v[0] = std::stof(list[0]);
        v[1] = std::stof(list[0]);
        v[2] = std::stof(list[0]);
    } else if (list.size() == 3) {
        v[0] = std::stof(list[0]);
        v[1] = std::stof(list[1]);
        v[2] = std::stof(list[2]);
    } else {
        Error("parse_vector3 failed");
    }
    return v;
}

std::vector<std::pair<Real, Real>> parse_spectrum(const std::string &value) {
    std::vector<std::string> list = split_string(value, std::regex("(,| )+"));
    std::vector<std::pair<Real, Real>> s;
    if (list.size() == 1 && list[0].find(":") == std::string::npos) {
        // a single uniform value for all wavelength
        s.push_back(std::make_pair(Real(-1), stof(list[0])));
    } else {
        for (auto val_str : list) {
            std::vector<std::string> pair = split_string(val_str, std::regex(":"));
            if (pair.size() < 2) {
                Error("parse_spectrum failed");
            }
            s.push_back(std::make_pair(Real(stof(pair[0])), Real(stof(pair[1]))));
        }
    }
    return s;
}

Matrix4x4 parse_matrix4x4(const std::string &value) {
    std::vector<std::string> list = split_string(value, std::regex("(,| )+"));
    if (list.size() != 16) {
        Error("parse_matrix4x4 failed");
    }

    Matrix4x4 m;
    int k = 0;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            m(i, j) = std::stof(list[k++]);
        }
    }
    return m;
}

Matrix4x4 parse_transform(pugi::xml_node node) {
    Matrix4x4 tform = Matrix4x4::identity();
    for (auto child : node.children()) {
        std::string name = to_lowercase(child.name());
        if (name == "scale") {
            Real x = 1.0;
            Real y = 1.0;
            Real z = 1.0;
            if (!child.attribute("x").empty())
                x = std::stof(child.attribute("x").value());
            if (!child.attribute("y").empty())
                y = std::stof(child.attribute("y").value());
            if (!child.attribute("z").empty())
                z = std::stof(child.attribute("z").value());
            tform = scale(Vector3{x, y, z}) * tform;
        } else if (name == "translate") {
            Real x = 0.0;
            Real y = 0.0;
            Real z = 0.0;
            if (!child.attribute("x").empty())
                x = std::stof(child.attribute("x").value());
            if (!child.attribute("y").empty())
                y = std::stof(child.attribute("y").value());
            if (!child.attribute("z").empty())
                z = std::stof(child.attribute("z").value());
            tform = translate(Vector3{x, y, z}) * tform;
        } else if (name == "rotate") {
            Real x = 0.0;
            Real y = 0.0;
            Real z = 0.0;
            Real angle = 0.0;
            if (!child.attribute("x").empty())
                x = std::stof(child.attribute("x").value());
            if (!child.attribute("y").empty())
                y = std::stof(child.attribute("y").value());
            if (!child.attribute("z").empty())
                z = std::stof(child.attribute("z").value());
            if (!child.attribute("angle").empty())
                angle = std::stof(child.attribute("angle").value());
            tform = rotate(angle, Vector3(x, y, z)) * tform;
        } else if (name == "lookat") {
            Vector3 pos = parse_vector3(child.attribute("origin").value());
            Vector3 target = parse_vector3(child.attribute("target").value());
            Vector3 up = parse_vector3(child.attribute("up").value());
            tform = look_at(pos, target, up) * tform;
        } else if (name == "matrix") {
            Matrix4x4 trans = parse_matrix4x4(std::string(child.attribute("value").value()));
            tform = trans * tform;
        }
    }
    return tform;
}

Texture<Spectrum> parse_spectrum_texture(
        pugi::xml_node node,
        const std::map<std::string /* name id */, ParsedTexture> &texture_map,
        TexturePool &texture_pool) {
    std::string type = node.name();
    if (type == "spectrum") {
        std::vector<std::pair<Real, Real>> spec =
            parse_spectrum(node.attribute("value").value());
        if (spec.size() > 1) {
            Vector3 xyz = integrate_XYZ(spec);
            return make_constant_spectrum_texture(fromRGB(XYZ_to_RGB(xyz)));
        } else if (spec.size() == 1) {
            return make_constant_spectrum_texture(fromRGB(Vector3{1, 1, 1}));
        } else {
            return make_constant_spectrum_texture(fromRGB(Vector3{0, 0, 0}));
        }
    } else if (type == "rgb") {
        return make_constant_spectrum_texture(
            fromRGB(parse_vector3(node.attribute("value").value())));
    } else if (type == "ref") {
        // referencing a texture
        std::string ref_id = node.attribute("id").value();
        auto t_it = texture_map.find(ref_id);
        if (t_it == texture_map.end()) {
            Error(std::string("Texture not found. ID = ") + ref_id);
        }
        const ParsedTexture t = t_it->second;
        return make_image_spectrum_texture(
            ref_id, t.filename, texture_pool, t.uscale, t.vscale);
    } else {
        Error(std::string("Unknown spectrum texture type:") + type);
        return ConstantTexture<Spectrum>{make_zero_spectrum()};
    }
}

RenderOptions parse_integrator(pugi::xml_node node) {
    RenderOptions options;
    std::string type = node.attribute("type").value();
    if (type == "path") {
        options.integrator = Integrator::Path;
        for (auto child : node.children()) {
            std::string name = child.attribute("name").value();
            if (name == "maxDepth") {
                options.max_depth = std::stoi(child.attribute("value").value());
            }
        }
    } else if (type == "direct") {
        options.integrator = Integrator::Path;
        options.max_depth = 2;
    } else if (type == "depth") {
        options.integrator = Integrator::Depth;
    } else if (type == "meanCurvature") {
        options.integrator = Integrator::MeanCurvature;
    } else if (type == "rayDifferential") {
        options.integrator = Integrator::RayDifferential;
    } else if (type == "mipmapLevel") {
        options.integrator = Integrator::MipmapLevel;
    } else {
        Error(std::string("Unsupported integrator: ") + type);
    }
    return options;
}

std::tuple<int /* width */, int /* height */, std::string /* filename */>
parse_film(pugi::xml_node node) {
    int width = c_default_res, height = c_default_res;
    std::string filename = c_default_filename;

    for (auto child : node.children()) {
        std::string type = child.name();
        std::string name = child.attribute("name").value();
        if (name == "width") {
            width = std::stoi(child.attribute("value").value());
        } else if (name == "height") {
            height = std::stoi(child.attribute("value").value());
        } else if (name == "filename") {
            filename = std::string(child.attribute("value").value());
        }
        if (type == "rfilter") {
            std::string filter_type = child.attribute("type").value();
            if (filter_type != "box") {
                std::cerr << "Warning: the renderer currently only supports box filters." << std::endl;
            }
        }
    }
    return std::make_tuple(width, height, filename);
}

std::tuple<Camera, std::string /* output filename */, ParsedSampler>
parse_sensor(pugi::xml_node node) {
    Real fov = c_default_fov;
    Matrix4x4 to_world = Matrix4x4::identity();
    int width = c_default_res, height = c_default_res;
    std::string filename = c_default_filename;
    FovAxis fov_axis = FovAxis::X;
    ParsedSampler sampler;

    std::string type = node.attribute("type").value();
    if (type == "perspective") {
        for (auto child : node.children()) {
            std::string name = child.attribute("name").value();
            if (name == "fov") {
                fov = std::stof(child.attribute("value").value());
            } else if (name == "toWorld") {
                to_world = parse_transform(child);
            } else if (name == "fovAxis") {
                std::string value = child.attribute("value").value();
                if (value == "x") {
                    fov_axis = FovAxis::X;
                } else if (value == "y") {
                    fov_axis = FovAxis::Y;
                } else if (value == "diagonal") {
                    fov_axis = FovAxis::DIAGONAL;
                } else if (value == "smaller") {
                    fov_axis = FovAxis::SMALLER;
                } else if (value == "larger") {
                    fov_axis = FovAxis::LARGER;
                } else {
                    Error(std::string("Unknown fovAxis value: ") + value);
                }
            }
        }
    } else {
        Error(std::string("Unsupported sensor: ") + type);
    }

    for (auto child : node.children()) {
        if (std::string(child.name()) == "film") {
            std::tie(width, height, filename) = parse_film(child);
        } else if (std::string(child.name()) == "sampler") {
            std::string name = child.attribute("name").value();
            if (name != "independent") {
                std::cerr << "Warning: the renderer currently only supports independent samplers." << std::endl;
            }
            for (auto grand_child : child.children()) {
                std::string name = grand_child.attribute("name").value();
                if (name == "sampleCount") {
                    sampler.sample_count = std::stoi(grand_child.attribute("value").value());
                }
            }
        }
    }

    // convert to fovX (code taken from 
    // https://github.com/mitsuba-renderer/mitsuba/blob/master/src/librender/sensor.cpp)
    if (fov_axis == FovAxis::Y ||
            (fov_axis == FovAxis::SMALLER && height < width) ||
            (fov_axis == FovAxis::LARGER && width < height)) {
        Real aspect = width / Real(height);
        fov = degrees(2 * atan(
            tan(radians(fov) / 2) * aspect));
    } else if (fov_axis == FovAxis::DIAGONAL) {
        Real aspect = width / Real(height);
        Real diagonal = 2 * tan(radians(fov) / 2);
        Real width = diagonal / sqrt(1 + 1 / (aspect * aspect));
        fov = degrees(2 * atan(width / 2));
    }

    return std::make_tuple(Camera(to_world, fov, width, height), filename, sampler);
}

std::tuple<std::string /* ID */, Material> parse_bsdf(
        pugi::xml_node node,
        const std::map<std::string /* name id */, ParsedTexture> &texture_map,
        TexturePool &texture_pool) {
    std::string type = node.attribute("type").value();
    std::string id;
    if (!node.attribute("id").empty()) {
        id = node.attribute("id").value();
    }
    if (type == "diffuse") {
        Texture<Spectrum> reflectance = make_constant_spectrum_texture(fromRGB(Vector3{0.5, 0.5, 0.5}));
        for (auto child : node.children()) {
            std::string name = child.attribute("name").value();
            if (name == "reflectance") {
                reflectance = parse_spectrum_texture(child, texture_map, texture_pool);
            }
        }
        return std::make_tuple(id, Lambertian{reflectance});
    } else if (type == "roughplastic") {
        Texture<Spectrum> diffuse_reflectance = make_constant_spectrum_texture(fromRGB(Vector3{0.5, 0.5, 0.5}));
        Texture<Spectrum> specular_reflectance = make_constant_spectrum_texture(fromRGB(Vector3{1, 1, 1}));
        Texture<Real> roughness = make_constant_float_texture(Real(0.1));
        Real intIOR = 1.49;
        Real extIOR = 1.000277;
        for (auto child : node.children()) {
            std::string name = child.attribute("name").value();
            if (name == "diffuseReflectance") {
                diffuse_reflectance = parse_spectrum_texture(child, texture_map, texture_pool);
            } else if (name == "specularReflectance") {
                specular_reflectance = parse_spectrum_texture(child, texture_map, texture_pool);
            } else if (name == "alpha") {
                std::string type = child.name();
                if (type == "ref") {
                    // referencing a texture
                    std::string ref_id = child.attribute("id").value();
                    auto t_it = texture_map.find(ref_id);
                    if (t_it == texture_map.end()) {
                        Error(std::string("Texture not found. ID = ") + ref_id);
                    }
                    const ParsedTexture t = t_it->second;
                    Image1 alpha = imread1(t.filename);
                    // Convert alpha to roughness.
                    Image1 roughness_img(alpha.width, alpha.height);
                    for (int i = 0; i < alpha.width * alpha.height; i++) {
                        roughness_img.data[i] = sqrt(alpha.data[i]);
                    }
                    roughness = make_image_float_texture(
                        ref_id, roughness_img, texture_pool, t.uscale, t.vscale);
                } else if (type == "float") {
                    Real alpha = std::stof(child.attribute("value").value());
                    roughness = make_constant_float_texture(sqrt(alpha));
                } else {
                    Error(std::string("Unknown float texture type:") + type);
                }
            } else if (name == "roughness") {
                std::string type = child.name();
                if (type == "ref") {
                    // referencing a texture
                    std::string ref_id = child.attribute("id").value();
                    auto t_it = texture_map.find(ref_id);
                    if (t_it == texture_map.end()) {
                        Error(std::string("Texture not found. ID = ") + ref_id);
                    }
                    const ParsedTexture t = t_it->second;
                    roughness = make_image_float_texture(
                        ref_id, t.filename, texture_pool, t.uscale, t.vscale);
                } else if (type == "float") {
                    roughness = make_constant_float_texture(std::stof(child.attribute("value").value()));
                } else {
                    Error(std::string("Unknown float texture type:") + type);
                }
            } else if (name == "intIOR") {
                intIOR = std::stof(child.attribute("value").value()); 
            } else if (name == "extIOR") {
                extIOR = std::stof(child.attribute("value").value()); 
            }
        }
        return std::make_tuple(id, RoughPlastic{
            diffuse_reflectance, specular_reflectance, roughness, intIOR / extIOR});
    } else {
        Error(std::string("Unknown BSDF: ") + type);
    }
    return std::make_tuple("", Material{});
}

Shape parse_shape(pugi::xml_node node,
                  std::vector<Material> &materials,
                  std::map<std::string, int> &material_map,
                  const std::map<std::string /* name id */, ParsedTexture> &texture_map,
                  TexturePool &texture_pool,
                  std::vector<Light> &lights,
                  const std::vector<Shape> &shapes) {
    int material_id = -1;
    for (auto child : node.children()) {
        std::string name = child.name();
        if (name == "ref") {
            pugi::xml_attribute id = child.attribute("id");
            if (!id.empty()) {
                auto it = material_map.find(id.value());
                if (it == material_map.end()) {
                    Error("ref not found");
                }
                material_id = it->second;
                break;
            } else {
                Error("ref not specified");
            }
        } else if (name == "bsdf") {
            Material m;
            std::string material_name;
            std::tie(material_name, m) = parse_bsdf(child, texture_map, texture_pool);
            if (!material_name.empty()) {
                material_map[material_name] = materials.size();
            }
            material_id = materials.size();
            materials.push_back(m);
            break;
        }
    }
    if (material_id == -1) {
        if (material_map.find(c_default_material_name) == material_map.end()) {
            material_map[c_default_material_name] = materials.size();
            materials.push_back(Lambertian{
                make_constant_spectrum_texture(fromRGB(Vector3{0.5, 0.5, 0.5}))});
        }
        material_id = material_map[c_default_material_name];
    }

    Shape shape;
    std::string type = node.attribute("type").value();
    if (type == "obj") {
        std::string filename;
        Matrix4x4 to_world = Matrix4x4::identity();
        for (auto child : node.children()) {
            std::string name = child.attribute("name").value();
            if (name == "filename") {
                filename = child.attribute("value").value();
            } else if (name == "toWorld") {
                if (std::string(child.name()) == "transform") {
                    to_world = parse_transform(child);
                }
            }
        }
        shape = parse_obj(filename, to_world);
    } else if (type == "serialized") {
        std::string filename;
        int shape_index = 0;
        Matrix4x4 to_world = Matrix4x4::identity();
        for (auto child : node.children()) {
            std::string name = child.attribute("name").value();
            if (name == "filename") {
                filename = child.attribute("value").value();
            } else if (name == "toWorld") {
                if (std::string(child.name()) == "transform") {
                    to_world = parse_transform(child);
                }
            } else if (name == "shapeIndex") {
                shape_index = std::stoi(child.attribute("value").value());
            }
        }
        shape = load_serialized(filename, shape_index, to_world);
    } else if (type == "sphere") {
        Vector3 center{0, 0, 0};
        Real radius = 1;
        for (auto child : node.children()) {
            std::string name = child.attribute("name").value();
            if (name == "center") {
                center = Vector3{
                    std::stof(child.attribute("x").value()),
                    std::stof(child.attribute("y").value()),
                    std::stof(child.attribute("z").value())};
            } else if (name == "radius") {
                radius = std::stof(child.attribute("value").value());
            }
        }
        shape = Sphere{-1 /*material_id*/, -1 /*area_light_id*/, center, radius};
    } else {
        Error(std::string("Unknown shape:") + type);
    }
    set_material_id(shape, material_id);

    for (auto child : node.children()) {
        std::string name = child.name();
        if (name == "emitter") {
            Spectrum radiance = fromRGB(Vector3{1, 1, 1});
            for (auto grand_child : child.children()) {
                std::string name = grand_child.attribute("name").value();
                if (name == "radiance") {
                    std::string rad_type = grand_child.name();
                    if (rad_type == "spectrum") {
                        std::vector<std::pair<Real, Real>> spec =
                            parse_spectrum(grand_child.attribute("value").value());
                        Vector3 xyz = integrate_XYZ(spec);
                        radiance = fromRGB(XYZ_to_RGB(xyz));
                    } else if (rad_type == "rgb") {
                        radiance = fromRGB(parse_vector3(grand_child.attribute("value").value()));
                    }
                }
            }
            set_area_light_id(shape, lights.size());
            lights.push_back(DiffuseAreaLight{(int)shapes.size() /* shape ID */, radiance});
        }
    }

    return shape;
}

/// We don't load the images to memory at this stage. Only record their names.
ParsedTexture parse_texture(pugi::xml_node node) {
    std::string type = node.attribute("type").value();
    if (type == "bitmap") {
        std::string filename = "";
        Real uscale = 1;
        Real vscale = 1;
        for (auto child : node.children()) {
            std::string name = child.attribute("name").value();
            if (name == "filename") {
                filename = child.attribute("value").value();
            } else if (name == "uvscale") {
                uscale = vscale = std::stof(child.attribute("value").value());
            } else if (name == "uscale") {
                uscale = std::stof(child.attribute("value").value());
            } else if (name == "vscale") {
                vscale = std::stof(child.attribute("value").value());
            }
        }
        return ParsedTexture{fs::path(filename), uscale, vscale};
    }
    Error("Unknown texture type");
    return ParsedTexture{};
}

Scene parse_scene(pugi::xml_node node, const RTCDevice &embree_device) {
    RenderOptions options;
    Camera camera(Matrix4x4::identity(), c_default_fov, c_default_res, c_default_res);
    std::string filename = c_default_filename;
    std::vector<Material> materials;
    std::map<std::string /* name id */, int /* index id */> material_map;
    TexturePool texture_pool;
    std::map<std::string /* name id */, ParsedTexture> texture_map;
    std::vector<Shape> shapes;
    std::vector<Light> lights;
    for (auto child : node.children()) {
        std::string name = child.name();
        if (name == "integrator") {
            options = parse_integrator(child);
        } else if (name == "sensor") {
            ParsedSampler sampler;
            std::tie(camera, filename, sampler) = parse_sensor(child);
            options.samples_per_pixel = sampler.sample_count;
        } else if (name == "bsdf") {
            std::string material_name;
            Material m;
            std::tie(material_name, m) = parse_bsdf(child, texture_map, texture_pool);
            if (!material_name.empty()) {
                material_map[material_name] = materials.size();
                materials.push_back(m);
            }
        } else if (name == "shape") {
            Shape s = parse_shape(child,
                                  materials,
                                  material_map,
                                  texture_map,
                                  texture_pool,
                                  lights,
                                  shapes);
            shapes.push_back(s);
        } else if (name == "texture") {
            std::string id = child.attribute("id").value();
            if (texture_map.find(id) != texture_map.end()) {
                Error(std::string("Duplicated texture ID:") + id);
            }
            texture_map[id] = parse_texture(child);
        }
    }
    return Scene{embree_device, camera, materials, shapes, lights, texture_pool, options, filename};
}

Scene parse_scene(const fs::path &filename, const RTCDevice &embree_device) {
    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file(filename.c_str());
    if (!result) {
        std::cerr << "Error description: " << result.description() << std::endl;
        std::cerr << "Error offset: " << result.offset << std::endl;
        Error("Parse error");
    }
    // back up the current working directory and switch to the parent folder of the file
    fs::path old_path = fs::current_path();
    fs::current_path(filename.parent_path());
    Scene scene = parse_scene(doc.child("scene"), embree_device);
    // switch back to the old current working directory
    fs::current_path(old_path);
    return scene;
}
