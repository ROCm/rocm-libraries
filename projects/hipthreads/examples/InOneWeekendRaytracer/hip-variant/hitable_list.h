#ifndef HITTABLELISTH
#define HITTABLELISTH

#include "hitable.h"

class hittable_list: public hittable  {
    public:
        __host__ __device__ hittable_list() {}
        __host__ __device__ hittable_list(hittable **l, int n) {list = l; list_size = n; }
        __host__ __device__ hittable_list(hittable_list &&other)
            : list(other.list), list_size(other.list_size) {
            other.list = nullptr;
            other.list_size = 0;
        }
        __host__ __device__ hittable_list& operator=(hittable_list &&other) {
            for (int i = 0; i < list_size; i++) {
                delete list[i];
            }
            delete [] list;
            list = other.list;
            list_size = other.list_size;
            other.list = nullptr;
            other.list_size = 0;
            return *this;
        }
        __host__ __device__ ~hittable_list() {
            for (int i = 0; i < list_size; i++) {
                delete list[i];
            }
            delete [] list;
        }
        __host__ __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
        hittable **list;
        int list_size;
};

__host__ __device__ bool hittable_list::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
        hit_record temp_rec;
        bool hit_anything = false;
        double closest_so_far = t_max;
        for (int i = 0; i < list_size; i++) {
            if (list[i]->hit(r, t_min, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }
        return hit_anything;
}

#endif
