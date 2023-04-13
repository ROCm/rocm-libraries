// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <unistd.h>
#include <cuda/std/chrono>
#include <chrono>
#include <random>

#define HIP_CHECK(command)                                                     \
  {                                                                            \
    if (command != hipSuccess) {                                               \
      printf("HIP ERROR: %s", hipGetErrorString(command));                     \
      std::abort();                                                            \
    }                                                                          \
  }

#define divideAndRoundUp(x, y) ((x) / (y) + ((x) % (y) != 0))

constexpr size_t N_TIMESTAMPS = 1000;

template <class Duration>
using time_point = cuda::std::chrono::time_point<cuda::std::chrono::system_clock, Duration>;  

using timestamp_ns = time_point<cuda::std::chrono::nanoseconds>;

enum class datetime_component {
    INVALID = 0,
    YEAR,
    MONTH,
    DAY,
    WEEKDAY,
    HOUR,
    MINUTE,
    SECOND,
    MILLISECOND,
    MICROSECOND,
    NANOSECOND
};

//the implementation of this function is derived from cuDF
template <datetime_component Component>
__host__ __device__ inline int extract_component(timestamp_ns const ts) 
{
    using namespace cuda::std::chrono;

    auto days_since_epoch = floor<days>(ts);

    auto time_since_midnight = ts - days_since_epoch;

    if (time_since_midnight.count() < 0) { time_since_midnight += days(1); }

    auto const hrs_  = [&] { return duration_cast<hours>(time_since_midnight); };
    auto const mins_ = [&] { return duration_cast<minutes>(time_since_midnight) - hrs_(); };
    auto const secs_ = [&] {
      return duration_cast<seconds>(time_since_midnight) - hrs_() - mins_();
    };
    auto const millisecs_ = [&] {
      return duration_cast<milliseconds>(time_since_midnight) - hrs_() - mins_() - secs_();
    };
    auto const microsecs_ = [&] {
      return duration_cast<microseconds>(time_since_midnight) - hrs_() - mins_() - secs_() -
             millisecs_();
    };
    auto const nanosecs_ = [&] {
      return duration_cast<nanoseconds>(time_since_midnight) - hrs_() - mins_() - secs_() -
             millisecs_() - microsecs_();
    };

    switch (Component) {
      case datetime_component::YEAR:
        return static_cast<int>(year_month_day(days_since_epoch).year());
      case datetime_component::MONTH:
        return static_cast<unsigned>(year_month_day(days_since_epoch).month());
      case datetime_component::DAY:
        return static_cast<unsigned>(year_month_day(days_since_epoch).day());
      case datetime_component::WEEKDAY:
        return year_month_weekday(days_since_epoch).weekday().iso_encoding();
      case datetime_component::HOUR: return hrs_().count();
      case datetime_component::MINUTE: return mins_().count();
      case datetime_component::SECOND: return secs_().count();
      case datetime_component::MILLISECOND: return millisecs_().count();
      case datetime_component::MICROSECOND: return microsecs_().count();
      case datetime_component::NANOSECOND: return nanosecs_().count();
      default: return 0;
    }  
};

__global__ void process_timestamps(timestamp_ns *d_start_ts, timestamp_ns *d_end_ts) { 

  using namespace cuda::std::chrono;

  int id = blockIdx.x * blockDim.x + threadIdx.x;

  if(id<N_TIMESTAMPS) {
    if(id%64==0)
      printf("\e[1;35m\t\t\tTThread = %d is processing input timestamps (ns) start = %lld end = %lld\e[0m\n", id, d_start_ts[id].time_since_epoch().count(), d_end_ts[id].time_since_epoch().count());

    int start_year = extract_component<datetime_component::YEAR>(d_start_ts[id]);
    int start_day = extract_component<datetime_component::DAY>(d_start_ts[id]);
    int start_month = extract_component<datetime_component::MONTH>(d_start_ts[id]);
    int start_hour = extract_component<datetime_component::HOUR>(d_start_ts[id]);
    int start_minute = extract_component<datetime_component::MINUTE>(d_start_ts[id]);
  
    int end_year = extract_component<datetime_component::YEAR>(d_end_ts[id]);
    int end_day = extract_component<datetime_component::DAY>(d_end_ts[id]);
    int end_month = extract_component<datetime_component::MONTH>(d_end_ts[id]);
    int end_hour = extract_component<datetime_component::HOUR>(d_end_ts[id]);
    int end_minute = extract_component<datetime_component::MINUTE>(d_end_ts[id]);

    auto duration = d_end_ts[id] - d_start_ts[id];
    auto days_elapsed = duration_cast<days>(duration);
    auto hours_elapsed = duration_cast<hours>(duration-days_elapsed);
    auto mins_elapsed = duration_cast<minutes>(duration-days_elapsed-hours_elapsed);

    if(id%64==0)
      printf("\e[1;36m\t\t\tThread = %d has processed input timestamps (ns) start = %lld end = %lld : converted start date = %02d-%02d-%d %02d:%02d, converted end date = %02d-%02d-%4d %02d:%02d, elapsed time: %d days, %ld hours and %ld minutes\e[0m\n",
              id, d_start_ts[id].time_since_epoch().count(), d_end_ts[id].time_since_epoch().count(), 
              start_month, start_day, start_year, start_hour, start_minute, 
              end_month, end_day, end_year, end_hour, end_minute,
              days_elapsed.count(), hours_elapsed.count(), mins_elapsed.count());
  }
}

void initialize_timestamps_randomly(timestamp_ns *h_start_ts, timestamp_ns *h_end_ts) {
    
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<unsigned long> start_rand(0,50*365*24*60*60*1e9);
  std::uniform_int_distribution<unsigned long> offset_rand(0,365*24*60*60*1e9);
  
  for(int i=0; i<N_TIMESTAMPS; i++) {
     h_start_ts[i] = timestamp_ns(cuda::std::chrono::nanoseconds(start_rand(gen)));
     h_end_ts[i] = h_start_ts[i] + cuda::std::chrono::nanoseconds(offset_rand(gen));
  }  
}

int main(int argc, char **argv) {

  printf("\e[0;38m\n\n\tA program to test the functionality of LIBHIPCXX time library "
         "in the context of cuDF\e[0m\n\n");
  sleep(1); // for demonstration purposes
     
  int device_id;
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDevice(&device_id));
  HIP_CHECK(hipGetDeviceProperties(&props, device_id));
    

  size_t n_bytes_per_column = N_TIMESTAMPS * sizeof(timestamp_ns);
  
  timestamp_ns *h_start_ts;
  h_start_ts = (timestamp_ns*) malloc(n_bytes_per_column);
  
  timestamp_ns *h_end_ts;
  h_end_ts = (timestamp_ns*) malloc(n_bytes_per_column);
  
  printf("\e[0;38m\n\n\tInitializing dataframe with randomly generated timestamps...\e[0m\n\n");
  initialize_timestamps_randomly(h_start_ts, h_end_ts);

  timestamp_ns *d_start_ts = nullptr;
  timestamp_ns *d_end_ts = nullptr;

  HIP_CHECK(hipMalloc(&d_start_ts, n_bytes_per_column));
  HIP_CHECK(hipMalloc(&d_end_ts, n_bytes_per_column));
 
  printf("\e[0;38m\n\n\tCopying dataframe to the GPU...\e[0m\n\n");
  HIP_CHECK(hipMemcpy(d_start_ts, h_start_ts, n_bytes_per_column, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_end_ts, h_end_ts, n_bytes_per_column, hipMemcpyHostToDevice));
  
  printf("\e[0;38m\t\t(Processing timestamps on GPU %s)\e[0m\n", props.name);
  sleep(3); // for demonstration purposes
  dim3 block(N_TIMESTAMPS, 1, 1);
  dim3 grid(divideAndRoundUp(N_TIMESTAMPS, block.x), 1, 1);
  process_timestamps<<<grid,block>>>(d_start_ts, d_end_ts);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  return 0; 
}
