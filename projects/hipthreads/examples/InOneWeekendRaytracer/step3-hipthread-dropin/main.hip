//==================================================================================================
// Written in 2016 by Peter Shirley <ptrshrl@gmail.com>
//
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is distributed
// without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication along
// with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==================================================================================================

#include <functional>
#include <iostream>
#include <thread>
#include <vector>
#include <float.h>
#include <hiprand/hiprand.hpp>
#include "camera.h"
#include "hitable_list.h"
#include "material.h"
#include "random.h"
#include "sphere.h"
#include <hip/thread>
#include <thrust/copy.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/unique_ptr.h>

__host__ __device__ vec3 color(const ray& r, const hittable_list &world, RandState *state) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
    for(int i = 0; i < 50; i++) {
        hit_record rec;
        if (world.hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            vec3 attenuation;
            if(rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else {
                return vec3(0.0,0.0,0.0);
            }
        }
        else {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f * (unit_direction.y() + 1.0f);
            vec3 c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    return vec3(0.0,0.0,0.0); // exceeded recursion
}


__host__ __device__ hittable_list random_scene() {
#ifdef __HIP_DEVICE_COMPILE__
    RandState randState;
    setup_random(0, &randState);
    RandState *randState_ptr = &randState;
#else
    RandState *randState_ptr = nullptr;
#endif
    int n = 500;
    sphere **list = new sphere*[n+1];
    list[0] =  new sphere(vec3(0,-1000,0), 1000, new lambertian(vec3(0.5, 0.5, 0.5)));
    int i = 1;
    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            float choose_mat = random_double(randState_ptr);
            vec3 center(a+0.9*random_double(randState_ptr),0.2,b+0.9*random_double(randState_ptr));
            if ((center-vec3(4,0.2,0)).length() > 0.9) {
                if (choose_mat < 0.8f) {  // diffuse
                    list[i++] = new sphere(
                        center, 0.2,
                        new lambertian(vec3(random_double(randState_ptr)*random_double(randState_ptr),
                                            random_double(randState_ptr)*random_double(randState_ptr),
                                            random_double(randState_ptr)*random_double(randState_ptr)))
                    );
                }
                else if (choose_mat < 0.95f) { // metal
                    list[i++] = new sphere(
                        center, 0.2,
                        new metal(vec3(0.5*(1 + random_double(randState_ptr)),
                                       0.5*(1 + random_double(randState_ptr)),
                                       0.5*(1 + random_double(randState_ptr))),
                                  0.5*random_double(randState_ptr))
                    );
                }
                else {  // glass
                    list[i++] = new sphere(center, 0.2, new dielectric(1.5));
                }
            }
        }
    }

    list[i++] = new sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5));
    list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
    list[i++] = new sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));

    return hittable_list(list,i);
}


void write_image(uint8_t *image, int nx, int ny) {
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; ++i) {
            std::cout << static_cast<int>(image[3*(j*nx+i)+0]) << " ";
            std::cout << static_cast<int>(image[3*(j*nx+i)+1]) << " ";
            std::cout << static_cast<int>(image[3*(j*nx+i)+2]) << "\n";
        }
    }
}


int main() {
    int nx = 1200;
    int ny = 800;
    int ns = 10;

    // --- Tiling parameters ---
    const int tx = 8; // Tile width
    const int ty = 4; // Tile height
    const int tiles_x = (nx + tx - 1) / tx; // Number of tiles in X dimension
    const int tiles_y = (ny + ty - 1) / ty; // Number of tiles in Y dimension
    const int total_tiles = tiles_x * tiles_y;

    thrust::device_ptr<hittable_list> dev_ptr_world = thrust::device_malloc<hittable_list>(1);
    hittable_list *world = thrust::raw_pointer_cast(dev_ptr_world);

    vec3 lookfrom(13,2,3);
    vec3 lookat(0,0,0);
    float dist_to_focus = 10.0;
    float aperture = 0.1;

    camera cam(lookfrom, lookat, vec3(0,1,0), 30, float(nx)/float(ny), aperture, dist_to_focus);

    auto image_d = thrust::make_unique<uint8_t[]>(nx * ny * 3);

    {
        // Use list-initialization {} instead of direct-initialization () here:
        // on Windows the clang HIP front-end mis-parses `Type name(device_lambda)`
        // as a declaration and rejects the __device__ specifier (ROCM-25863).
        hip::wthread world_constructor{[=] __device__ () { new (world) hittable_list(random_scene()); }};
        world_constructor.join();

        const int k_threads = 4096;
        std::vector<hip::wthread> threads(k_threads);
        for (int t = 0; t < k_threads; ++t) {
            // Each thread will start at its index 't' and jump by 'k_threads' to get its next tile.
            threads[t] = hip::wthread([=, image = image_d.get_raw()] __device__ () {
                // Each thread processes a strided range of tiles.
                for (int tile_idx = t; tile_idx < total_tiles; tile_idx += k_threads) {
                    int tile_x = tile_idx % tiles_x;
                    int tile_y = tile_idx / tiles_x;

                    int i_start = tile_x * tx;
                    int j_start = tile_y * ty;

                    // Iterate over pixels within the tile
                    for (int j = j_start; j < j_start + ty && j < ny; ++j) {
                        for (int i = i_start; i < i_start + tx && i < nx; ++i) {
                            hiprandState randState;
                            setup_random(j * nx + i + 1, &randState);
                            vec3 col(0, 0, 0);
                            for (int s=0; s < ns; s++) {
                                float u = float(i + random_double(&randState)) / float(nx);
                                float v = float(j + random_double(&randState)) / float(ny);
                                ray r = cam.get_ray(u, v, &randState);
                                col += color(r, *world, &randState);
                            }
                            col /= float(ns);
                            col = vec3( sqrt(col[0]), sqrt(col[1]), sqrt(col[2]) );

                            image[3*(j*nx+i)+0] = int(255.99*col[0]);
                            image[3*(j*nx+i)+1] = int(255.99*col[1]);
                            image[3*(j*nx+i)+2] = int(255.99*col[2]);
                        }
                    }
                }
            });
        }
        for (int t = 0; t < k_threads; ++t) {
            threads[t].join();
        }

        hip::wthread([=] __device__ () { world->~hittable_list(); }).join();
    }

    thrust::device_free(dev_ptr_world);

    auto image_h = std::make_unique<uint8_t[]>(nx * ny * 3);
    thrust::copy(image_d.get(), image_d.get() + (nx * ny * 3), image_h.get());
    write_image(image_h.get(), nx, ny);
}
