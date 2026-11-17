#include "hip/hip_runtime.h"
#ifndef SPHEREH
#define SPHEREH

#include "hitable.h"

class sphere: public hittable  {
    public:
        __host__ __device__ sphere() {}
        __host__ __device__ sphere(vec3 cen, float r, material *m) : center(cen), radius(r), mat_ptr(m)  {};
        __host__ __device__ sphere(sphere &&other)
            : center(other.center), radius(other.radius), mat_ptr(other.mat_ptr) {
            other.mat_ptr = nullptr;
        }
        __host__ __device__ sphere& operator=(sphere &&other) {
            delete mat_ptr;
            center = other.center;
            radius = other.radius;
            mat_ptr = other.mat_ptr;
            other.mat_ptr = nullptr;
            return *this;
        }
        __host__ __device__ ~sphere() {
            delete mat_ptr;
        }
        __host__ __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
        vec3 center;
        float radius;
        material *mat_ptr;
};

__host__ __device__ bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    vec3 oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float b = dot(oc, r.direction());
    float c = dot(oc, oc) - radius*radius;
    float discriminant = b*b - a*c;
    if (discriminant > 0) {
        float temp = (-b - sqrt(discriminant))/a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center) / radius;
            rec.mat_ptr = mat_ptr;
            return true;
        }
        temp = (-b + sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center) / radius;
            rec.mat_ptr = mat_ptr;
            return true;
        }
    }
    return false;
}


#endif
